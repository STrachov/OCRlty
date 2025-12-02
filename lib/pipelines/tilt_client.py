# lib/pipelines/tilt_client.py
from __future__ import annotations

import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Опциональные тяжёлые зависимости: используем ленивый импорт
try:  # Pillow для работы с изображениями
    from PIL import Image  # type: ignore[import]
except Exception:  # pragma: no cover - в рантайме Pillow обязателен
    Image = None  # type: ignore[assignment]

try:  # PDF → изображения
    import pypdfium2 as pdfium  # type: ignore[import]
except Exception:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]

try:  # OCR (CPU)
    from paddleocr import PaddleOCR  # type: ignore[import]
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import]
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


MOCK = os.getenv("MOCK_VLLM", "0") == "1"


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("base_url must be non-empty")
    # приводим к виду http://host:port/v1
    return url if url.endswith("/v1") else (url + "/v1")


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Достаём JSON даже если модель вернула его внутри текста/```json```.

    Логика:
    1) если вся строка — это JSON-объект, просто парсим;
    2) иначе ищем блоки ```json ... ``` и пробуем их;
    3) иначе ищем первую сбалансированную {...} в тексте;
    4) в крайнем случае даём понятную ошибку.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")

    # 1) чистый JSON-объект
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # 2) ```json ... ```
    import re as _re

    blocks = _re.findall(r"```json(.*?)```", text, flags=_re.DOTALL | _re.IGNORECASE)
    for b in blocks:
        try:
            return json.loads(b.strip())
        except Exception:
            pass

    # 3) первая сбалансированная {...}
    stack: List[int] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append(i)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        start = None

    raise ValueError(f"Cannot extract JSON from model response: {text[:200]!r}...")


def _is_pdf(doc_bytes: bytes, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    # сигнатура %PDF
    return doc_bytes.startswith(b"%PDF")


class ArcticTiltClient:
    """Клиент к нашему GPU-серверу TILT (apps.tilt_api:app).

    Делает три вещи:
      1. Превращает bytes (PDF/PNG/JPEG) → список PIL.Image.
      2. Гоняет OCR по каждой странице (PaddleOCR, CPU) → слова + bbox.
      3. Формирует запрос /v1/tilt/generate и парсит JSON-ответ модели.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 10.0,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        ocr_lang: str = "en",
        question: Optional[str] = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.ocr_lang = ocr_lang

        # Вопрос к TILT по умолчанию: извлечение реквизитов чека/квитанции в JSON.
        self.question = question or os.getenv(
            "TILT_KIE_PROMPT",
            (
                "You are an information extraction engine for receipts and invoices. "
                "Given the OCR words with bounding boxes of a single document page, "
                "extract key fields and return ONLY a valid JSON object with the "
                "following structure (use null when value is missing):\n"
                "{"
                "\"seller_name\": str | null,"
                "\"seller_address\": str | null,"
                "\"seller_vat_id\": str | null,"
                "\"invoice_number\": str | null,"
                "\"invoice_date\": str | null,"
                "\"currency\": str | null,"
                "\"subtotal\": float | null,"
                "\"tax_amount\": float | null,"
                "\"total_amount\": float | null,"
                "\"items\": ["
                "{\"description\": str, \"quantity\": float | null, \"unit_price\": float | null, \"line_total\": float | null}"
                "]"
                "}.\n"
                "Do not add explanations or extra text, output JSON only."
            ),
        )

        # HTTP-клиент к tilt_api
        self._cli = httpx.Client(timeout=self.timeout)

        # OCR и связанные ошибки
        self._ocr: Any = None
        self._ocr_err: Optional[Exception] = None

        if not MOCK:
            self._init_ocr()

    # ------------------------------------------------------------------ #
    # Внутренние хелперы
    # ------------------------------------------------------------------ #

    def _init_ocr(self) -> None:
        """Ленивый и безопасный init PaddleOCR."""
        if PaddleOCR is None:
            self._ocr_err = RuntimeError("paddleocr import failed (module not found)")
            return
        if np is None:
            self._ocr_err = RuntimeError("numpy is not installed")
            return
        try:
            self._ocr = PaddleOCR(
                lang=self.ocr_lang,
                use_angle_cls=True,
                #use_gpu=False,
                show_log=False,
            )
        except Exception as exc:
            # здесь важно сохранить ИМЕННО оригинальное исключение,
            # чтобы в логе и в HTTP 500 увидеть реальную причину
            self._ocr_err = exc

    # ----- HTTP к tilt_api -----

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_tilt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/tilt/generate"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._cli.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_exc = e
                # HTTP 4xx повторять смысла нет
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
                    break
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s)

        if last_exc is not None:
            raise RuntimeError(f"Error calling {url}: {last_exc}") from last_exc
        raise RuntimeError(f"Unknown error calling {url}")

    # ----- Bytes → изображения -----

    def _pdf_to_images(self, doc_bytes: bytes) -> List["Image.Image"]:
        if pdfium is None or Image is None:
            raise RuntimeError("pypdfium2 and Pillow are required for PDF support")

        try:
            pdf = pdfium.PdfDocument(io.BytesIO(doc_bytes))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to open PDF: {exc}") from exc

        images: List["Image.Image"] = []
        try:
            page_indices = list(range(len(pdf)))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to enumerate PDF pages: {exc}") from exc

        # Для чеков обычно достаточно небольшого количества страниц,
        # но если их много — можно будет позже ограничить.
        for i in page_indices:
            page = pdf[i]
            try:
                pil_image = page.render(scale=2.0).to_pil()  # type: ignore[no-untyped-call]
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to render PDF page {i}: {exc}") from exc
            images.append(pil_image.convert("RGB"))

        if not images:
            raise RuntimeError("PDF has zero pages")
        return images

    def _image_bytes_to_images(self, doc_bytes: bytes) -> List["Image.Image"]:
        if Image is None:
            raise RuntimeError("Pillow is required for image support")
        try:
            img = Image.open(io.BytesIO(doc_bytes))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to open image: {exc}") from exc

        images: List["Image.Image"] = []
        try:
            n_frames = getattr(img, "n_frames", 1)
        except Exception:
            n_frames = 1

        for frame_idx in range(n_frames):
            try:
                if frame_idx:
                    img.seek(frame_idx)
                images.append(img.convert("RGB"))
            except Exception:
                # в крайнем случае берём только первый кадр
                if not images:
                    images.append(img.convert("RGB"))
                break

        return images

    def _doc_bytes_to_images(self, doc_bytes: bytes, content_type: Optional[str]) -> List["Image.Image"]:
        if _is_pdf(doc_bytes, content_type):
            return self._pdf_to_images(doc_bytes)
        return self._image_bytes_to_images(doc_bytes)

    # ----- OCR → TiltRequest.pages -----

    def _run_ocr(self, image: "Image.Image") -> Tuple[int, int, List[Dict[str, Any]]]:
        if self._ocr_err is not None:
            raise RuntimeError(f"PaddleOCR initialization failed: {self._ocr_err!r}")
        if self._ocr is None or np is None:
            raise RuntimeError("PaddleOCR is not available")

        np_img = np.array(image)
        try:
            result = self._ocr.ocr(np_img, cls=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OCR failed: {exc}") from exc

        width, height = image.size
        words: List[Dict[str, Any]] = []

        # result — это список страниц; мы передали одну картинку,
        # поэтому берём лишь первый элемент
        for page in result or []:
            for det in page:
                # формат: [box, (text, score)]
                try:
                    box, (txt, score) = det
                except Exception:
                    continue
                txt = (txt or "").strip()
                if not txt:
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                words.append({"text": txt, "bbox": bbox})

        return width, height, words

    # ----- парсинг ответа модели -----

    def _parse_response(self, content: str) -> Dict[str, Any]:
        return _extract_json_from_text(content)

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def infer(self, doc_bytes: bytes, content_type: Optional[str] = None) -> Dict[str, Any]:
        """Основной метод: bytes документа → dict с извлечёнными полями.

        :param doc_bytes: содержимое загруженного файла (PDF/JPEG/PNG)
        :param content_type: MIME-тип (для PDF более надёжное определение)
        """
        if MOCK:
            # Упрощённый дубль для локальной отладки без TILT'а
            return {
                "mock": True,
                "length_bytes": len(doc_bytes),
                "content_type": content_type,
            }

        images = self._doc_bytes_to_images(doc_bytes, content_type)
        if not images:
            raise RuntimeError("No pages produced from input document")

        pages_payload: List[Dict[str, Any]] = []
        for img in images:
            w, h, words = self._run_ocr(img)
            ocr_page = {
                "width": int(w),
                "height": int(h),
                "words": words,
            }
            pages_payload.append({"ocr": ocr_page})

        payload: Dict[str, Any] = {
            "question": self.question,
            "pages": pages_payload,
            "model": self.model,
        }

        resp = self._post_tilt(payload)

        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unexpected tilt_api response structure: {e}; got: {resp}") from e

        return self._parse_response(content)

    def close(self) -> None:
        try:
            self._cli.close()
        except Exception:
            pass
