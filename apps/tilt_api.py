# apps/tilt_api.py
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
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.multimodal.tilt_processor import (
    Document,
    Page,
    Question,
    TiltPreprocessor,
)

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
# Config from ENV
# -------------------------------------------------------------------------

MODEL_NAME: str = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
DTYPE: str = os.getenv("TILT_DTYPE", "float16")  # "float16" | "bfloat16"
TP_SIZE: int = int(os.getenv("TILT_TP", os.getenv("VLLM_TP_SIZE", "1")))
MAX_MODEL_LEN_ENV: Optional[str] = os.getenv("TILT_MAX_LEN", None)

GPU_UTIL: float = float(os.getenv("VLLM_GPU_UTIL", os.getenv("GPU_UTIL", "0.90")))
HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/cache/hf")
ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in (
    "1",
    "true",
    "yes",
)

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
# vLLM Engine + TiltPreprocessor (через CLI, как в example_tilt.py)
# -------------------------------------------------------------------------


def _build_llm_engine() -> Tuple[LLMEngine, TiltPreprocessor]:
    """Create LLMEngine and TiltPreprocessor configured for TILT."""
    parser = FlexibleArgumentParser(
        description="Arctic-TILT vLLM engine (used behind FastAPI)."
    )

    # Добавляем стандартные engine-аргументы vLLM
    parser = AsyncEngineArgs.add_cli_args(parser, async_args_only=False)

    # Значения по умолчанию для TILT (по мотивам examples/tilt_example.py)
    parser.set_defaults(
        model=MODEL_NAME,
        task="tilt_generate",
        scheduler_cls="vllm.tilt.scheduler.Scheduler",
        gpu_memory_utilization=GPU_UTIL,
        dtype=DTYPE,
        max_num_seqs=16,
        enforce_eager=ENFORCE_EAGER,
        disable_async_output_proc=True,  # Not implemented in TILT scheduler
        disable_log_requests=True,
    )

    # Мы не читаем реальные CLI-аргументы, а используем только дефолты+ENV
    args = parser.parse_args([])

    # Донастраиваем из ENV
    args.tensor_parallel_size = TP_SIZE
    args.download_dir = HF_CACHE_DIR

    if MAX_MODEL_LEN_ENV:
        try:
            args.max_model_len = int(MAX_MODEL_LEN_ENV)
        except ValueError:
            log.warning(
                "Invalid TILT_MAX_LEN=%s, ignoring and using default.",
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
# Pydantic models: low-level TILT request (internal API)
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
    # В реальных запросах ожидаем, что OCR заполнен.
    ocr: Optional[OCRPage] = Field(
        None, description="OCR result with word boxes in pixels"
    )
    # Картинка опциональна — для визуальных признаков.
    image_b64: Optional[str] = Field(
        None, description="Base64-encoded PNG/JPG of the page"
    )
    image_path: Optional[str] = Field(
        None, description="Filesystem path to page image (mounted into container)"
    )


class TiltRequest(BaseModel):
    question: str = Field(..., description="Doc-VQA / KIE question or instruction")
    pages: List[InputPage] = Field(
        ..., description="List of page images and/or OCR data"
    )
    model: Optional[str] = Field(
        None, description="Override model name (optional)"
    )
    temperature: Optional[float] = Field(
        None, description="Overrides default temperature"
    )
    max_tokens: Optional[int] = Field(
        None, description="Overrides default max tokens"
    )


# -------------------------------------------------------------------------
# Helpers: InputPage -> TILT Document/Page
# -------------------------------------------------------------------------


def _decode_image(page: InputPage) -> Image.Image:
    """Получаем PIL.Image из image_b64 или image_path, либо создаём заглушку."""
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

    # Заглушка: белый лист A4
    return Image.new(mode="L", size=(768, 1086), color=255)


def _input_page_to_tilt_page(page: InputPage) -> Page:
    img = _decode_image(page)

    if page.ocr:
        width = page.ocr.width
        height = page.ocr.height
        words = [w.text for w in page.ocr.words]
        bboxes = []
        for w in page.ocr.words:
            try:
                bboxes.append([float(v) for v in w.bbox])
            except Exception:  # noqa: BLE001
                continue
    else:
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


def _run_tilt_inference(req: TiltRequest) -> str:
    """Выполнить один запрос TILT через LLMEngine, синхронно."""

    if not req.pages:
        raise HTTPException(status_code=400, detail="Request must contain at least one page.")

    # Собираем Document и Question
    doc_id = f"api-{int(time.time() * 1000)}"
    tilt_pages = [_input_page_to_tilt_page(p) for p in req.pages]

    document = Document(ident=doc_id, split=None, pages=tilt_pages)
    questions = [Question(feature_name="answer", text=req.question)]

    samples = preprocessor.preprocess(document, questions)
    if not samples:
        log.warning("Preprocessor returned no samples; returning empty answer.")
        return ""

    sample = samples[0]

    # Sampling params (с учётом оверрайдов)
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=0,
    )

    request_id = f"{doc_id}-q0"
    final_output = None

    # Сериализуем доступ к движку
    with _engine_lock:
        llm_engine.add_request(
            prompt=sample,
            request_id=request_id,
            params=sampling_params,
        )

        # Как в tilt_example: крутим step() пока наша заявка не закончится
        while True:
            request_outputs = llm_engine.step()
            for output in request_outputs:
                if output.request_id == request_id and output.finished:
                    final_output = output
                    break
            if final_output is not None:
                break

    if final_output is None:
        log.warning("No RequestOutput from LLMEngine for request_id=%s", request_id)
        return ""

    outputs = getattr(final_output, "outputs", None) or []
    if not outputs:
        log.warning("RequestOutput.outputs is empty for request_id=%s", request_id)
        return ""

    try:
        text = outputs[0].text
        if not isinstance(text, str):
            text = str(text)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to extract text from outputs[0]: %s", exc)
        return ""


# -------------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------------


@app.post("/v1/tilt/generate")
def tilt_generate(req: TiltRequest = Body(...)) -> Dict[str, Any]:
    """
    Низкоуровневый endpoint: напрямую принимает TiltRequest
    (question + pages[ocr/image]) и возвращает OpenAI-like ответ.
    """
    text = _run_tilt_inference(req)

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
        "raw": None,
    }


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME}


# -------------------------------------------------------------------------
# Опциональный warmup (через полный пайплайн, но с try/except)
# -------------------------------------------------------------------------

try:
    dummy_req = TiltRequest(
        question="ping",
        pages=[
            InputPage(
                ocr=OCRPage(width=768, height=1086, words=[]),
            )
        ],
    )
    _ = _run_tilt_inference(dummy_req)
    log.info("Warmup for TILT completed (or returned empty, but no crash).")
except Exception as exc:  # noqa: BLE001
    log.warning("Warmup failed (non-fatal): %s", exc)
