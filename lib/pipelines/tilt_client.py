from __future__ import annotations

import io
import json
import os
import time
import tempfile
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from lib.utils.logging import get_event_logger  # shared structured logger

obs = get_event_logger("tilt_client")

# Опциональные тяжёлые зависимости: используем ленивый импорт
try:  # Pillow для работы с изображениями
    from PIL import Image  # type: ignore[import]
except Exception:  # pragma: no cover - в рантайме Pillow обязателен
    Image = None  # type: ignore[assignment]

try:  # PDF → изображения
    import pypdfium2 as pdfium  # type: ignore[import]
except Exception:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import]
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    # PaddleX 3.x: универсальный inference-пайплайн
    from paddlex import create_pipeline  # type: ignore[import]
except Exception:  # pragma: no cover
    create_pipeline = None  # type: ignore[assignment]


MOCK = os.getenv("MOCK_VLLM", "0") == "1"
CANDIDATES_PLACEHOLDER = "{{}}"

DEFAULT_MAX_CANDIDATES = int(os.getenv("DEFAULT_MAX_CANDIDATES", "8"))
DEFAULT_MAX_NEIGHBOURS = int(os.getenv("DEFAULT_MAX_NEIGHBOURS", "3"))

# Нормализованные анкоры (ты можешь расширять словарь по мере нужды)
FIELD_ANCHOR_TIERS = {
    "total_price": [
        # Tier 1: более специфичные "итоговые" ярлыки (имеют приоритет)
        [("GRAND", "TOTAL"), "GRANDTOTAL", ("AMOUNT", "DUE"), "AMOUNTDUE", ("BALANCE", "DUE"), "BALANCEDUE"],
        # Tier 2: fallback
        ["TOTAL", "T0TAL", "TOTL", "TTL", "TL"],
    ],
    "cash": [
        ["CASH", "TENDER", "TENDERED", "PAID", "RECEIVED", "RCVD"],
    ],
    "change": [
        ["CHANGE", "CHNG", "CHG"],
    ],
}


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("base_url must be non-empty")
    # приводим к виду http://host:port/v1
    return url if url.endswith("/v1") else (url + "/v1")


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Достаём JSON даже если модель вернула его внутри текста/```json```."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response from model")

    # 1) чистый JSON-объект
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # 2) ```json ... ```
    blocks = re.findall(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
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
    return doc_bytes.startswith(b"%PDF")


class ArcticTiltClient:
    """Клиент к GPU-серверу TILT (apps.tilt_api:app).

    Делает три вещи:
      1) bytes (PDF/PNG/JPEG) → список PIL.Image.
      2) OCR (PaddleX, CPU) → слова + bbox.
      3) POST /v1/tilt/generate и возврат результата (как есть) + used_question.
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
        min_confidence: float = 0.3,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.ocr_lang = ocr_lang
        self.min_confidence = min_confidence
        self.question = os.getenv("TILT_KIE_PROMPT")

        obs.log_event(
            "INFO",
            "client.init",
            msg="TILT client initialized",
            tilt={"base_url": self.base_url, "model": self.model, "timeout_s": self.timeout},
        )

        self._cli = httpx.Client(timeout=self.timeout)

        self._ocr: Any = None
        self._ocr_err: Optional[Exception] = None

        if not MOCK:
            self._init_ocr()

    # ------------------------------------------------------------------ #
    # OCR init / ensure
    # ------------------------------------------------------------------ #

    def _init_ocr(self) -> None:
        if create_pipeline is None:
            self._ocr_err = RuntimeError("paddlex is not installed")
            obs.log_event("ERROR", "client.ocr_init_failed", msg="paddlex import failed")
            return
        try:
            self._ocr = create_pipeline(pipeline="OCR")
            obs.log_event("INFO", "client.ocr_init_ok")
        except Exception as exc:  # noqa: BLE001
            self._ocr_err = exc
            obs.log_event("ERROR", "client.ocr_init_failed", msg=str(exc), error={"type": type(exc).__name__})

    def _ensure_ocr(self) -> None:
        if self._ocr is not None or self._ocr_err is not None:
            return
        self._init_ocr()

    # ------------------------------------------------------------------ #
    # HTTP к tilt_api
    # ------------------------------------------------------------------ #

    def _headers(self, request_id: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    def _post_tilt(self, payload: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        url = f"{self.base_url}/tilt/generate"
        last_exc: Optional[Exception] = None
        last_body: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                obs.log_event(
                    "INFO",
                    "client.tilt_request_start",
                    request_id=request_id,
                    tilt={"url": url, "attempt": attempt, "pages": len(payload.get("pages", []))},
                    prompt={"chars": len(payload.get("question") or "")},
                )
                resp = self._cli.post(url, json=payload, headers=self._headers(request_id))
                resp.raise_for_status()
                dt = time.time() - t0
                obs.log_event(
                    "INFO",
                    "client.tilt_request_done",
                    request_id=request_id,
                    duration_ms=round(dt * 1000, 2),
                    http={"status": resp.status_code},
                    tilt={"response_chars": len(resp.text or "")},
                )
                return resp.json()
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                last_exc = e
                body = None
                status = None
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                    status = e.response.status_code
                    try:
                        body = e.response.text
                    except Exception:
                        body = "<failed to read response body>"
                last_body = body

                obs.log_event(
                    "WARNING",
                    "client.tilt_request_failed",
                    request_id=request_id,
                    duration_ms=round(dt * 1000, 2),
                    http={"status": status},
                    error={"type": type(e).__name__, "message": str(e)},
                )

                # HTTP 4xx повторять смысла нет
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None and 400 <= e.response.status_code < 500:
                    break
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s)

        if last_exc is not None:
            raise RuntimeError(f"Error calling {url}: {last_exc} (response_body={last_body})") from last_exc
        raise RuntimeError(f"Unknown error calling {url}")

    # ------------------------------------------------------------------ #
    # Bytes → изображения
    # ------------------------------------------------------------------ #

    def _pdf_to_images(self, doc_bytes: bytes) -> List["Image.Image"]:
        if pdfium is None or Image is None:
            raise RuntimeError("pypdfium2 and Pillow are required for PDF support")

        pdf = pdfium.PdfDocument(io.BytesIO(doc_bytes))  # type: ignore[arg-type]

        images: List["Image.Image"] = []
        page_indices = list(range(len(pdf)))

        for i in page_indices:
            page = pdf[i]
            pil_image = page.render(scale=2.0).to_pil()  # type: ignore[no-untyped-call]
            images.append(pil_image.convert("RGB"))

        if not images:
            raise RuntimeError("PDF has zero pages")
        return images

    def _image_bytes_to_images(self, doc_bytes: bytes) -> List["Image.Image"]:
        if Image is None:
            raise RuntimeError("Pillow is required for image support")

        img = Image.open(io.BytesIO(doc_bytes))
        images: List["Image.Image"] = []
        n_frames = getattr(img, "n_frames", 1)

        for frame_idx in range(n_frames):
            try:
                if frame_idx:
                    img.seek(frame_idx)
                images.append(img.convert("RGB"))
            except Exception:
                if not images:
                    images.append(img.convert("RGB"))
                break

        return images

    def _doc_bytes_to_images(self, doc_bytes: bytes, content_type: Optional[str]) -> List["Image.Image"]:
        if _is_pdf(doc_bytes, content_type):
            return self._pdf_to_images(doc_bytes)
        return self._image_bytes_to_images(doc_bytes)

    # ------------------------------------------------------------------ #
    # OCR → TiltRequest.pages
    # ------------------------------------------------------------------ #

    def _run_ocr(self, img: "Image.Image") -> Tuple[int, int, List[Dict[str, Any]]]:
        self._ensure_ocr()
        if self._ocr_err is not None or self._ocr is None:
            raise RuntimeError(f"OCR initialization failed: {self._ocr_err!r}")
        if Image is None:
            raise RuntimeError("Pillow is required for OCR images")

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size

        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            img.save(tmp_path, format="PNG")
            raw_out_gen = self._ocr.predict(tmp_path, use_doc_orientation_classify=False)
            raw_out = list(raw_out_gen)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if not raw_out:
            return w, h, []

        page0 = raw_out[0]

        if hasattr(page0, "res"):
            res = page0.res
        elif isinstance(page0, dict) and "res" in page0:
            res = page0["res"]
        else:
            res = page0

        if hasattr(res, "__dict__") and not isinstance(res, dict):
            res = res.__dict__

        if not isinstance(res, dict):
            return w, h, []

        def _first_nonempty(keys: Tuple[str, ...]):
            for key in keys:
                if key not in res:
                    continue
                val = res[key]
                if val is None:
                    continue
                try:
                    if np is not None and isinstance(val, np.ndarray):
                        if val.size == 0:
                            continue
                    elif isinstance(val, (list, tuple, str)):
                        if len(val) == 0:
                            continue
                except Exception:
                    pass
                return val
            return None

        boxes = _first_nonempty(("rec_boxes", "dt_polys", "det_boxes", "boxes"))
        texts = _first_nonempty(("rec_texts", "rec_text", "texts"))
        scores = _first_nonempty(("rec_scores", "rec_score", "scores"))

        if boxes is None or texts is None:
            return w, h, []

        if np is not None:
            try:
                boxes_arr = np.array(boxes)
            except Exception:
                boxes_arr = boxes
        else:
            boxes_arr = boxes

        out_words: List[Dict[str, Any]] = []
        for idx, (box, text) in enumerate(zip(boxes_arr, texts)):
            if not text:
                continue

            score = 1.0
            if scores is not None and idx < len(scores):
                try:
                    score = float(scores[idx])
                except Exception:
                    pass

            if score < self.min_confidence:
                continue

            try:
                if np is not None and isinstance(box, np.ndarray):
                    pts = box.reshape(-1, 2)
                    x1 = float(pts[:, 0].min())
                    y1 = float(pts[:, 1].min())
                    x2 = float(pts[:, 0].max())
                    y2 = float(pts[:, 1].max())
                else:
                    if (
                        isinstance(box, (list, tuple))
                        and len(box) == 4
                        and all(isinstance(v, (int, float)) for v in box)
                    ):
                        x1, y1, x2, y2 = map(float, box)
                    else:
                        pts = list(box)
                        xs = [float(p[0]) for p in pts]
                        ys = [float(p[1]) for p in pts]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            except Exception:
                continue

            out_words.append({"text": str(text), "bbox": [x1, y1, x2, y2], "score": float(score)})
        # Stabilize reading order (important for anchor→right-neighbour heuristics)
        out_words.sort(key=lambda d: ((d["bbox"][1] + d["bbox"][3]) * 0.5, d["bbox"][0]))

        return w, h, out_words

    # ------------------------------------------------------------------ #
    # Candidate builder (anchors → value to the right)
    # ------------------------------------------------------------------ #

    def _norm_token(self, s: str) -> str:
        s = (s or "").strip().upper()
        return re.sub(r"[^A-Z0-9]+", "", s)

    def _has_digit(self, s: str) -> bool:
        return any(ch.isdigit() for ch in (s or ""))

    def _bbox_mid_y(self, bbox: List[float]) -> float:
        return (float(bbox[1]) + float(bbox[3])) / 2.0

    def _bbox_h(self, bbox: List[float]) -> float:
        return float(bbox[3]) - float(bbox[1])

    def _same_line(self, bbox1: List[float], bbox2: List[float], y_tol_px: float) -> bool:
        return abs(self._bbox_mid_y(bbox1) - self._bbox_mid_y(bbox2)) <= y_tol_px

    def _is_right_of_anchor(self, anchor_bbox: List[float], cand_bbox: List[float]) -> bool:
        return float(cand_bbox[0]) >= float(anchor_bbox[2]) - 2.0

    def _find_phrase_anchors(self, words: List[Dict[str, Any]], w1: str, w2: str) -> List[int]:
        w1n = self._norm_token(w1)
        w2n = self._norm_token(w2)

        out: List[int] = []
        for i in range(1, len(words)):
            t2 = self._norm_token(words[i].get("text", ""))
            if t2 != w2n:
                continue
            bbox2 = words[i].get("bbox")
            if not bbox2 or len(bbox2) != 4:
                continue

            t1 = self._norm_token(words[i - 1].get("text", ""))
            if t1 != w1n:
                continue
            bbox1 = words[i - 1].get("bbox")
            if not bbox1 or len(bbox1) != 4:
                continue

            y_tol = max(8.0, 0.8 * self._bbox_h(bbox2))
            if self._same_line(bbox1, bbox2, y_tol):
                out.append(i)

        return out

    def _find_anchor_indices_for_tier(self, words: List[Dict[str, Any]], tier: List[Any]) -> List[int]:
        idxs: List[int] = []
        tier_token_norm = {self._norm_token(x) for x in tier if isinstance(x, str)}

        for i, w in enumerate(words):
            t = self._norm_token(w.get("text", ""))
            if t and t in tier_token_norm:
                idxs.append(i)

        for pat in tier:
            if isinstance(pat, tuple) and len(pat) == 2:
                idxs.extend(self._find_phrase_anchors(words, pat[0], pat[1]))

        return sorted(set(idxs))

    def _select_anchor_indices_by_priority(self, words: List[Dict[str, Any]], field_name: str) -> Tuple[List[int], Optional[str]]:
        tiers = FIELD_ANCHOR_TIERS.get(field_name)
        if not tiers:
            return [], None

        for tier_i, tier in enumerate(tiers, start=1):
            idxs = self._find_anchor_indices_for_tier(words, tier)
            if idxs:
                return idxs, str(tier_i)
        return [], None

    def _candidate_from_anchor(self, words: List[Dict[str, Any]], anchor_i: int, max_neighbours: int) -> Optional[str]:
        anchor_bbox = words[anchor_i].get("bbox")
        if not anchor_bbox or len(anchor_bbox) != 4:
            return None

        y_tol = max(8.0, 0.8 * self._bbox_h(anchor_bbox))

        start_j: Optional[int] = None
        for j in range(anchor_i + 1, len(words)):
            t = (words[j].get("text") or "").strip()
            bbox = words[j].get("bbox")
            if not t or not bbox or len(bbox) != 4:
                continue
            if not self._has_digit(t):
                continue
            if not self._same_line(anchor_bbox, bbox, y_tol):
                continue
            if not self._is_right_of_anchor(anchor_bbox, bbox):
                continue
            start_j = j
            break

        if start_j is None:
            return None

        parts: List[str] = []
        for j in range(start_j, min(len(words), start_j + max_neighbours)):
            t = (words[j].get("text") or "").strip()
            bbox = words[j].get("bbox")
            if not t or not bbox or len(bbox) != 4:
                break
            if not self._same_line(anchor_bbox, bbox, y_tol):
                break
            if not self._is_right_of_anchor(anchor_bbox, bbox):
                break
            parts.append(t)

        if not parts:
            return None
        return "".join(parts)

    def _collect_candidates_from_pages(
        self,
        pages_payload: List[Dict[str, Any]],
        field_name: str,
        max_candidates: int,
        max_neighbours: int,
    ) -> Tuple[List[str], Optional[str], int]:
        out: List[str] = []
        remaining = max_candidates
        anchors_total = 0
        tier_selected: Optional[str] = None

        for p in pages_payload:
            ocr = (p or {}).get("ocr") or {}
            words = ocr.get("words") or []
            if not words:
                continue

            anchor_idxs, tier = self._select_anchor_indices_by_priority(words, field_name)
            if tier_selected is None:
                tier_selected = tier
            anchors_total += len(anchor_idxs)

            for ai in anchor_idxs:
                if remaining <= 0:
                    return out, tier_selected, anchors_total
                cand = self._candidate_from_anchor(words, ai, max_neighbours=max_neighbours)
                if cand is not None:
                    out.append(cand)
                    remaining -= 1

        return out, tier_selected, anchors_total

    def _inject_candidates_into_question(self, base_question: str, candidates: List[str]) -> str:
        if not candidates:
            return base_question

        arr = "[" + ",".join(json.dumps(c) for c in candidates) + "]"

        if CANDIDATES_PLACEHOLDER in base_question:
            return base_question.replace(CANDIDATES_PLACEHOLDER, arr)

        return (
            base_question.rstrip()
            + "\nReturn EXACTLY ONE of these strings (copy-paste, no edits):\n"
            + arr
            + "\nOutput only the chosen string."
        )

    def _build_pages_payload(self, images: List["Image.Image"], request_id: Optional[str]) -> Tuple[List[Dict[str, Any]], int]:
        pages_payload: List[Dict[str, Any]] = []
        total_words = 0

        for page_idx, img in enumerate(images):
            t0 = time.time()
            w, h, words = self._run_ocr(img)
            dt = time.time() - t0

            total_words += len(words)
            obs.log_event(
                "INFO",
                "client.ocr_page_done",
                request_id=request_id,
                duration_ms=round(dt * 1000, 2),
                ocr={"page_idx": page_idx, "width": w, "height": h, "words": len(words)},
            )

            ocr_page = {
                "width": int(w),
                "height": int(h),
                "words": [{"text": w_["text"], "bbox": w_["bbox"]} for w_ in words],
            }
            pages_payload.append({"ocr": ocr_page})

        if not total_words:
            raise RuntimeError("OCR produced zero words on all pages")

        return pages_payload, total_words

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def debug_candidates(
        self,
        doc_bytes: bytes,
        content_type: Optional[str] = None,
        question: Optional[str] = None,
        field_name: Optional[str] = None,
        max_candidates: Optional[int] = None,
        max_neighbours: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        images = self._doc_bytes_to_images(doc_bytes, content_type)
        if not images:
            raise RuntimeError("No pages produced from input document")

        pages_payload, total_words = self._build_pages_payload(images, request_id=request_id)

        base_question = question or self.question
        if not base_question:
            raise ValueError("question is required (pass question=... or set TILT_KIE_PROMPT)")

        used_question = base_question
        cands: List[str] = []
        tier_selected: Optional[str] = None
        anchors_total = 0

        if field_name and field_name in FIELD_ANCHOR_TIERS:
            mc = int(max_candidates) if max_candidates is not None else DEFAULT_MAX_CANDIDATES
            mn = int(max_neighbours) if max_neighbours is not None else DEFAULT_MAX_NEIGHBOURS
            cands, tier_selected, anchors_total = self._collect_candidates_from_pages(pages_payload, field_name, mc, mn)
            used_question = self._inject_candidates_into_question(base_question, cands)

        obs.log_event(
            "INFO",
            "client.candidates_built",
            request_id=request_id,
            field={"name": field_name, "tier": tier_selected},
            candidates={"count": len(cands), "anchors_found": anchors_total},
        )

        return {
            "field_name": field_name,
            "candidates": cands,
            "tier": tier_selected,
            "anchors_found": anchors_total,
            "used_question": used_question,
            "pages_payload_preview": {"num_pages": len(pages_payload), "words_total": total_words},
        }

    def infer(
        self,
        doc_bytes: bytes,
        content_type: Optional[str] = None,
        question: Optional[str] = None,
        field_name: Optional[str] = None,
        max_candidates: Optional[int] = None,
        max_neighbours: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if MOCK:
            return {"mock": True, "length_bytes": len(doc_bytes), "content_type": content_type}

        obs.log_event(
            "INFO",
            "client.infer_start",
            request_id=request_id,
            doc={"content_type": content_type, "size_bytes": len(doc_bytes)},
            field={"name": field_name},
            params={
                "max_candidates": max_candidates,
                "max_neighbours": max_neighbours,
            },
            prompt={"provided": bool(question), "chars": len(question or "")},
        )

        try:
            images = self._doc_bytes_to_images(doc_bytes, content_type)
        except Exception as e:
            obs.log_event(
                "ERROR",
                "client._doc_bytes_to_images_failed",
                request_id=request_id,
                doc={"content_type": content_type, "size_bytes": len(doc_bytes)},
                error={"type": type(e).__name__, "message": str(e)},
            )
            raise

        if not images:
            obs.log_event(
                "WARNING",
                "client.doc_no_pages",
                request_id=request_id,
                doc={"content_type": content_type, "size_bytes": len(doc_bytes)},
            )
            raise RuntimeError("No pages produced from input document")

        obs.log_event(
            "INFO",
            "client.doc_loaded",
            request_id=request_id,
            doc={"pages": len(images), "content_type": content_type, "size_bytes": len(doc_bytes)},
        )

        pages_payload, total_words = self._build_pages_payload(images, request_id=request_id)

        base_question = question or self.question
        if not base_question:
            raise ValueError("question is required (pass question=... or set TILT_KIE_PROMPT)")

        used_question = base_question

        cands: List[str] = []
        tier_selected: Optional[str] = None
        anchors_total = 0

        if field_name and field_name in FIELD_ANCHOR_TIERS:
            mc = int(max_candidates) if max_candidates is not None else DEFAULT_MAX_CANDIDATES
            mn = int(max_neighbours) if max_neighbours is not None else DEFAULT_MAX_NEIGHBOURS
            cands, tier_selected, anchors_total = self._collect_candidates_from_pages(pages_payload, field_name, mc, mn)
            used_question = self._inject_candidates_into_question(base_question, cands)

            obs.log_event(
                "INFO",
                "client.candidates_built",
                request_id=request_id,
                field={"name": field_name, "tier": tier_selected},
                candidates={"count": len(cands), "anchors_found": anchors_total},
            )

        # payload includes request_id as a fallback channel (header is primary)
        payload: Dict[str, Any] = {
            "question": used_question,
            "pages": pages_payload,
            "model": self.model,
            "request_id": request_id,
        }

        obs.log_event(
            "INFO",
            "client.payload_ready",
            request_id=request_id,
            doc={"pages": len(pages_payload)},
            ocr={"words_total": total_words},
            field={"name": field_name, "mode": "candidates" if cands else "plain", "tier": tier_selected},
            candidates={"count": len(cands)},
            prompt={"chars": len(used_question or "")},
        )

        resp = self._post_tilt(payload, request_id=request_id)

        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unexpected tilt_api response structure: {e}; got keys={list(resp.keys())}") from e

        return {"response": content, "used_question": used_question, "candidates": cands}

    def close(self) -> None:
        try:
            self._cli.close()
        except Exception:
            pass
