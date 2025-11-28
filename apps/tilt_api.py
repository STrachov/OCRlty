"""
FastAPI server for Arctic-TILT on vLLM.

- Логика максимально близка к examples/tilt_example.py.
- Работает с OCR-only входом (слова + bbox). Если нет реальной картинки,
  создаётся белый dummy-Image нужного размера, чтобы TiltPreprocessor был доволен.
- Один основной endpoint: POST /v1/tilt/generate
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image  # type: ignore[import]

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.multimodal.tilt_processor import (
    Document,
    Page,
    Question,
    TiltPreprocessor,
)
from vllm.utils import FlexibleArgumentParser

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

log = logging.getLogger("tilt_api")
if not log.handlers:
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# -------------------------------------------------------------------------
# Конфиг из ENV
# -------------------------------------------------------------------------

MODEL_NAME: str = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")

# В examples/tilt_example.py используют bfloat16. На float16 мы уже ловили NaN.
DTYPE: str = os.getenv("TILT_DTYPE", "bfloat16")  # "bfloat16" по умолчанию

TP_SIZE: int = int(os.getenv("TILT_TP", os.getenv("TILT_TP_SIZE", "1")))
MAX_MODEL_LEN_ENV: Optional[str] = os.getenv("TILT_MAX_MODEL_LEN", None)

GPU_UTIL: float = float(os.getenv("TILT_GPU_UTIL",
                                  os.getenv("VLLM_GPU_UTIL", "0.9")))

HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/cache/hf")
ENFORCE_EAGER: bool = True  # как в примере; для TILT Long рекомендуют eager

DEFAULT_TEMPERATURE: float = float(os.getenv("TILT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("TILT_MAX_TOKENS", "256"))

log.info(
    "TILT config: model=%s, dtype=%s, tp=%s, gpu_util=%s",
    MODEL_NAME,
    DTYPE,
    TP_SIZE,
    GPU_UTIL,
)

# -------------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------------

app = FastAPI(title="Arctic-TILT API", version="1.0")

# -------------------------------------------------------------------------
# vLLM Engine + TiltPreprocessor (как в tilt_example.py, но без CLI)
# -------------------------------------------------------------------------


def _build_llm_engine() -> Tuple[LLMEngine, TiltPreprocessor]:
    """
    Создаём LLMEngine и TiltPreprocessor, подавая аргументы через
    AsyncEngineArgs.add_cli_args + EngineArgs.from_cli_args, как в examples/tilt_example.py.
    """
    parser = FlexibleArgumentParser(
        description="Arctic-TILT vLLM engine (used behind FastAPI)."
    )

    # Добавляем все engine-параметры vLLM (как в примере):
    parser = AsyncEngineArgs.add_cli_args(parser, async_args_only=False)

    # Значения по умолчанию, максимально близкие к tilt_example.py
    parser.set_defaults(
        model=MODEL_NAME,
        task="tilt_generate",
        scheduler_cls="vllm.tilt.scheduler.Scheduler",
        gpu_memory_utilization=GPU_UTIL,
        dtype=DTYPE,
        max_num_seqs=16,
        enforce_eager=ENFORCE_EAGER,
        # Не поддерживается V1-движком, но vLLM сам откатится на V0 и выведет warning.
        disable_async_output_proc=True,
    )

    # Не хотим брать реальные аргументы командной строки uvicorn — парсим пустой список.
    args = parser.parse_args([])

    # Донастраиваем некоторые поля из ENV
    args.tensor_parallel_size = TP_SIZE
    args.download_dir = HF_CACHE_DIR

    if MAX_MODEL_LEN_ENV:
        try:
            args.max_model_len = int(MAX_MODEL_LEN_ENV)
        except ValueError:
            log.warning(
                "Invalid TILT_MAX_MODEL_LEN=%s, ignoring.",
                MAX_MODEL_LEN_ENV,
            )

    engine_args = EngineArgs.from_cli_args(args)
    log.info("Creating LLMEngine with task=%s", engine_args.task)

    llm_engine = LLMEngine.from_engine_args(engine_args)

    tokenizer = llm_engine.get_tokenizer()
    preprocessor = TiltPreprocessor.from_config(
        model_config=llm_engine.model_config.hf_config,
        tokenizer=tokenizer.backend_tokenizer,
    )

    return llm_engine, preprocessor


llm_engine, preprocessor = _build_llm_engine()
_engine_lock = threading.Lock()

# -------------------------------------------------------------------------
# Pydantic-модели входа
# -------------------------------------------------------------------------


class OCRWord(BaseModel):
    text: str = Field(..., description="Recognized token text")
    bbox: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="[x0,y0,x1,y1] in pixels",
    )


class OCRPage(BaseModel):
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")
    words: List[OCRWord] = Field(
        default_factory=list,
        description="Words with absolute bboxes",
    )


class InputPage(BaseModel):
    ocr: Optional[OCRPage] = Field(
        None,
        description="OCR result with word boxes in pixels",
    )
    image_b64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG/JPG of the page (optional).",
    )
    image_path: Optional[str] = Field(
        None,
        description="Path to page image inside container (optional).",
    )


class TiltRequest(BaseModel):
    question: str = Field(..., description="Doc-VQA / KIE question or instruction")
    pages: List[InputPage] = Field(
        ..., description="List of pages with OCR and/or images"
    )
    model: Optional[str] = Field(
        None,
        description="Override model name (optional).",
    )
    temperature: Optional[float] = Field(
        None, description="Overrides default temperature."
    )
    max_tokens: Optional[int] = Field(
        None, description="Overrides default max tokens."
    )


# -------------------------------------------------------------------------
# Helpers: InputPage -> TILT Page / Document / Question
# -------------------------------------------------------------------------


def _decode_image(page: InputPage) -> Image.Image:
    """Получаем PIL.Image из image_b64 или image_path, либо создаём белую заглушку."""
    if page.image_b64:
        try:
            raw = base64.b64decode(page.image_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to decode image_b64: %s", exc)

    if page.image_path:
        try:
            return Image.open(page.image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to open image_path=%s: %s", page.image_path, exc)

    # dummy: белый лист A4-ish
    return Image.new(mode="L", size=(768, 1086), color=255)


def _input_page_to_tilt_page(page: InputPage) -> Page:
    img = _decode_image(page)

    if page.ocr:
        width = page.ocr.width
        height = page.ocr.height
        words = [w.text for w in page.ocr.words]
        bboxes: List[List[float]] = []
        for w in page.ocr.words:
            try:
                bboxes.append([float(v) for v in w.bbox])
            except Exception:  # noqa: BLE001
                continue
    else:
        # Если OCR нет — просто пустая страница с dummy-изображением
        width, height = img.size
        words = []
        bboxes = []

    return Page(
        words=words,
        bboxes=bboxes,
        width=width,
        height=height,
        image=img,
    )


def _build_document(req: TiltRequest) -> Document:
    doc_id = f"api-{int(time.time() * 1000)}"
    pages = [_input_page_to_tilt_page(p) for p in req.pages]
    return Document(ident=doc_id, split=None, pages=pages)


def _build_questions(req: TiltRequest) -> List[Question]:
    q_text = req.question.strip()
    # В examples/tilt_example.py feature_name = key. У нас один вопрос, условно key="user_question".
    return [Question(feature_name="user_question", text=q_text)]


# -------------------------------------------------------------------------
# Core inference: один запрос через LLMEngine (с блокировкой)
# -------------------------------------------------------------------------


def _run_tilt_inference(req: TiltRequest) -> Tuple[str, Optional[str]]:
    """
    Выполняет один запрос TILT через глобальный LLMEngine.

    Возвращает:
    - text: str        — финальный ответ модели (может быть пустым)
    - debug_repr: str? — repr(RequestOutput) для дебага
    """
    if not req.pages:
        raise HTTPException(
            status_code=400,
            detail="Request must contain at least one page.",
        )

    document = _build_document(req)
    questions = _build_questions(req)

    samples = preprocessor.preprocess(document, questions)
    if not samples:
        log.warning("TiltPreprocessor returned no samples for document %s", document.ident)
        return "", None

    sample = samples[0]

    temperature = (
        req.temperature
        if req.temperature is not None
        else DEFAULT_TEMPERATURE
    )
    max_tokens = (
        req.max_tokens
        if req.max_tokens is not None
        else DEFAULT_MAX_TOKENS
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=0,  # TILT task ожидает logprobs
    )

    request_id = f"{document.ident}-q0"
    final_output = None

    # синхронный доступ к движку: один request за раз
    with _engine_lock:
        llm_engine.add_request(
            prompt=sample,
            request_id=request_id,
            params=sampling_params,
        )

        while True:
            request_outputs = llm_engine.step()
            if not request_outputs:
                continue

            for out in request_outputs:
                if out.request_id != request_id:
                    continue
                if not out.finished:
                    continue
                final_output = out
                break

            if final_output is not None:
                break

    if final_output is None:
        log.warning("No RequestOutput for request_id=%s", request_id)
        return "", None

    outputs = getattr(final_output, "outputs", None) or []
    if not outputs:
        log.warning("RequestOutput.outputs is empty for request_id=%s", request_id)
        return "", repr(final_output)

    first = outputs[0]
    debug_repr = repr(first)

    try:
        text = first.text
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read text from outputs[0]: %s", exc)
        return "", debug_repr

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    log.info("TILT output text=%r", text)

    return text, debug_repr


# -------------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------------


@app.post("/v1/tilt/generate")
def tilt_generate(req: TiltRequest = Body(...)) -> Dict[str, Any]:
    """
    Основной endpoint: принимает question + pages[ocr/image] и возвращает
    OpenAI-подобный ответ с полем raw.output_repr для дебага.
    """
    print("Welcome!")
    try:
        text, debug_repr = _run_tilt_inference(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Error during TILT inference: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": "tiltcmpl-1",
        "object": "chat.completion",
        "model": req.model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "raw": {"output_repr": debug_repr} if debug_repr is not None else None,
    }


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "gpu_util": GPU_UTIL,
        "tp_size": TP_SIZE,
    }


# (опционально можно было бы добавить warmup, но с учётом тяжёлой модели
#  лучше не стартовать лишний инференс на импорт модуля)
