import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from PIL import Image

from vllm import LLM, SamplingParams

# ------------------------------------------------------------
# Логер
# ------------------------------------------------------------
logger = logging.getLogger("tilt_api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ------------------------------------------------------------
# Конфиг TILT / vLLM
# ------------------------------------------------------------

TILT_MODEL_ID = os.getenv("TILT_MODEL_ID", "Snowflake/snowflake-arctic-tilt-v1.3")
TILT_DTYPE = os.getenv("TILT_DTYPE", "float16")
TILT_TP_SIZE = int(os.getenv("TILT_TP_SIZE", "1"))
TILT_GPU_UTIL = float(os.getenv("TILT_GPU_UTILIZATION", "0.9"))
TILT_MAX_MODEL_LEN = int(os.getenv("TILT_MAX_MODEL_LEN", "125000"))
HF_HOME = os.getenv("HF_HOME", "/workspace/cache/hf")

logger.info(
    "TILT config: model=%s, dtype=%s, tp=%d, gpu_util=%s",
    TILT_MODEL_ID,
    TILT_DTYPE,
    TILT_TP_SIZE,
    TILT_GPU_UTIL,
)

# ------------------------------------------------------------
# Pydantic-модели запроса
# ------------------------------------------------------------


class OCRWord(BaseModel):
    text: str
    bbox: List[float] = Field(
        ...,
        description="Bounding box [x0, y0, x1, y1] в пикселях относительно страницы",
        min_items=4,
        max_items=4,
    )

    @validator("bbox")
    def _validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox must have 4 elements: [x0, y0, x1, y1]")
        x0, y0, x1, y1 = v
        if x1 < x0 or y1 < y0:
            # немножко подправим вместо падения
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            return [x0, y0, x1, y1]
        return v


class OCRPage(BaseModel):
    width: int
    height: int
    words: List[OCRWord]


class PageInput(BaseModel):
    # base64 PNG / JPEG; опционально
    image: Optional[str] = Field(
        default=None,
        description="Base64-кодированное изображение страницы (без data: URI префикса)",
    )
    # Результат PaddleOCR (или другого OCR)
    ocr: Optional[OCRPage] = None


class TiltRequest(BaseModel):
    question: str
    pages: List[PageInput]
    temperature: float = 0.0
    max_tokens: int = 64


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------


def _decode_base64_image(b64_data: str) -> Image.Image:
    """Декодируем base64-строку в PIL.Image.

    Ожидается *чистая* base64 строка без 'data:image/...;base64,' префикса.
    """
    try:
        binary = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(binary)).convert("RGB")
        return img
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to decode base64 image: %s", e)
        raise HTTPException(status_code=400, detail="Invalid base64 image") from e


def _normalize_bbox(bbox: List[float]) -> List[float]:
    """Лёгкая нормализация bbox: гарантируем порядок и неотрицательность."""
    x0, y0, x1, y1 = bbox
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), max(0.0, x1), max(0.0, y1)
    return [x0, y0, x1, y1]


def _build_tilt_pages(pages: List[PageInput]) -> List[Dict[str, Any]]:
    """
    Приводим вход в формат, максимально похожий на тот, что использует
    TILT Document/Page внутри официального TiltPreprocessor:

        Page(
            words=[...],
            bboxes=[[x0,y0,x1,y1], ...],
            width=...,
            height=...,
            image=Image или None,
        )
    """
    tilt_pages: List[Dict[str, Any]] = []

    for idx, page in enumerate(pages):
        if page.ocr is None:
            # Если нет OCR, смысла передавать страницу почти нет — пропускаем
            logger.warning("Page %d has no OCR; skipping it", idx)
            continue

        ocr = page.ocr
        words: List[str] = []
        bboxes: List[List[float]] = []

        for w in ocr.words:
            txt = (w.text or "").strip()
            if not txt:
                continue
            words.append(txt)
            bboxes.append(_normalize_bbox(w.bbox))

        if not words:
            logger.warning("Page %d has OCR but no non-empty words; skipping", idx)
            continue

        page_dict: Dict[str, Any] = {
            # ВАЖНО: width / height, а не page_size – это ближе к тому, что
            # есть в TILT Page (width, height поля).
            "words": words,
            "bboxes": bboxes,
            "width": int(ocr.width),
            "height": int(ocr.height),
        }

        # Если изображение реально передано – декодируем и отдадим модели.
        # Если нет – просто не указываем ключ "image": модель должна
        # уметь работать только по OCR-тексту.
        if page.image:
            try:
                img = _decode_base64_image(page.image)
                page_dict["image"] = img
            except HTTPException:
                # Ошибку уже залогировали и вернули бы 400, но на всякий случай:
                logger.warning("Invalid image on page %d; ignoring image.", idx)

        tilt_pages.append(page_dict)

    if not tilt_pages:
        raise HTTPException(
            status_code=400,
            detail="No valid pages with OCR were provided.",
        )

    return tilt_pages


def _build_tilt_inputs(req: TiltRequest) -> List[Dict[str, Any]]:
    """
    Формируем список sample-ов для tilt_generate.
    Каждый элемент – это примерно "document + question".
    """
    pages = _build_tilt_pages(req.pages)

    tilt_input: Dict[str, Any] = {
        "document": {
            "pages": pages,
        },
        "question": req.question,
    }

    return [tilt_input]


# ------------------------------------------------------------
# Инициализация vLLM / TILT
# ------------------------------------------------------------


def _build_llm() -> LLM:
    logger.info("Creating LLM with task=tilt_generate")
    llm = LLM(
        model=TILT_MODEL_ID,
        tokenizer=TILT_MODEL_ID,
        task="tilt_generate",
        dtype=TILT_DTYPE,
        tensor_parallel_size=TILT_TP_SIZE,
        gpu_memory_utilization=TILT_GPU_UTIL,
        max_model_len=TILT_MAX_MODEL_LEN,
        download_dir=HF_HOME,
        trust_remote_code=False,
        enforce_eager=True,
    )
    return llm


llm = _build_llm()

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

app = FastAPI(title="Arctic-TILT doc VQA API")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Arctic-TILT is running"}


@app.post("/v1/tilt/generate")
async def tilt_generate(req: TiltRequest) -> Dict[str, Any]:
    """
    Основной endpoint:

        - принимает вопрос и структуру OCR-страниц,
        - собирает inputs для tilt_generate,
        - вызывает модель Snowflake Arctic TILT,
        - возвращает ответ в стиле chat.completions.
    """
    try:
        tilt_inputs = _build_tilt_inputs(req)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to build TILT inputs: %s", e)
        raise HTTPException(status_code=400, detail="Invalid document structure") from e

    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        # Стоп-токены оставляем дефолтными – TILT сама знает свои маркеры конца.
    )

    try:
        # В TILT-ветке vLLM есть специализированный метод .tilt_generate(...)
        # который умеет принимать "document + question" вместо голого текста.
        outputs = llm.tilt_generate(
            inputs=tilt_inputs,
            sampling_params=sampling_params,
        )
    except AttributeError:
        # На всякий случай – если вдруг окружение без tilt_generate.
        logger.exception("llm.tilt_generate is not available in this vLLM build.")
        raise HTTPException(
            status_code=500,
            detail="TILT-specific generation is not available in this vLLM build.",
        ) from None
    except Exception as e:  # noqa: BLE001
        logger.exception("TILT generation error: %s", e)
        raise HTTPException(status_code=500, detail="TILT generation failed") from e

    # outputs – список RequestOutput; берём первый запрос и первую гипотезу
    if not outputs or not outputs[0].outputs:
        logger.warning("TILT returned no outputs")
        answer_text = ""
    else:
        answer_text = outputs[0].outputs[0].text or ""
        logger.info("TILT output text=%r", answer_text)

    # Возвращаем ответ в формате, похожем на OpenAI chat.completions
    response: Dict[str, Any] = {
        "id": "tiltcmpl-1",
        "object": "chat.completion",
        "model": TILT_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        # Для отладки можно вернуть repr сырого вывода
        "raw": {
            "output_repr": repr(outputs[0].outputs[0]) if outputs and outputs[0].outputs else None
        },
    }
    return response
