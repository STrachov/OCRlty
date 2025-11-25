# apps/tilt_api.py
from __future__ import annotations

import os
import base64
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
log = logging.getLogger("tilt_api")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# -------------------------------------------------------------------------
# Config (ENV)
# -------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
DTYPE: str = os.getenv("TILT_DTYPE", "float16")  # float16 | bfloat16
TP_SIZE: int = int(os.getenv("TILT_TP", "1"))
MAX_MODEL_LEN: int = int(os.getenv("TILT_MAX_LEN", "16384"))
GPU_UTIL: float = float(os.getenv("VLLM_GPU_UTIL", os.getenv("GPU_UTIL", "0.90")))
HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/cache/hf")
ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "1") in ("1", "true", "True")

# -------------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------------
app = FastAPI(title="Arctic-TILT API", version="1.0")

# -------------------------------------------------------------------------
# vLLM (task=tilt_generate)
# -------------------------------------------------------------------------
# ВАЖНО: под TILT НЕЛЬЗЯ вызывать llm.generate().
# Нужно использовать только llm.tilt_generate(...).
log.info(
    "Starting TILT LLM with model=%s, dtype=%s, tp=%s, max_model_len=%s",
    MODEL_NAME, DTYPE, TP_SIZE, MAX_MODEL_LEN,
)

llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    dtype=DTYPE,  # torch_dtype deprecated
    tensor_parallel_size=TP_SIZE,
    max_model_len=MAX_MODEL_LEN,
    download_dir=HF_CACHE_DIR,
    gpu_memory_utilization=GPU_UTIL,
    enforce_eager=ENFORCE_EAGER,
    task="tilt_generate",  # ← критично
)

# дефолтный sampling
DEFAULT_SP = SamplingParams(
    temperature=float(os.getenv("TILT_TEMPERATURE", "0.0")),
    max_tokens=int(os.getenv("TILT_MAX_TOKENS", "256")),
    stop=[],
)


# -------------------------------------------------------------------------
# Schemas
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
    # Заполняем ИЛИ image_b64, ИЛИ image_path, ИЛИ ocr (или комбинацию)
    image_b64: Optional[str] = Field(
        None, description="Base64-encoded PNG/JPG of the page"
    )
    image_path: Optional[str] = Field(
        None, description="Filesystem path to page image (mounted into container)"
    )
    ocr: Optional[OCRPage] = Field(
        None, description="OCR result with word boxes in pixels"
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
# Helpers: build TILT input
# -------------------------------------------------------------------------
def _normalize_bbox(b: List[float]) -> List[float]:
    # Ensure floats (not ints) and clamp negatives
    return [float(max(0.0, v)) for v in b]


def _page_to_tilt(page: InputPage) -> Dict[str, Any]:
    """
    Convert a single InputPage into TILT's expected per-page dict.
    Keys:
      - 'image_b64' or 'image_path' (optional)
      - 'page_size': [width, height]
      - 'words': [str, ...]
      - 'bboxes': [[x0,y0,x1,y1], ...] (absolute pixel coords)
    """
    page_dict: Dict[str, Any] = {}

    if page.image_b64:
        try:
            base64.b64decode(page.image_b64, validate=True)
        except Exception:
            # Если base64 кривой — просто пропускаем проверку, дальше разрулит TILT
            pass
        page_dict["image_b64"] = page.image_b64

    if page.image_path:
        page_dict["image_path"] = page.image_path

    if page.ocr:
        page_dict["page_size"] = [int(page.ocr.width), int(page.ocr.height)]
        words = [w.text for w in page.ocr.words]
        bboxes = [_normalize_bbox(w.bbox) for w in page.ocr.words]
        page_dict["words"] = words
        page_dict["bboxes"] = bboxes

    return page_dict


def build_tilt_input(req: TiltRequest) -> Dict[str, Any]:
    """Build a single TILT input object compatible with llm.tilt_generate."""
    return {
        "question": req.question,
        "pages": [_page_to_tilt(p) for p in req.pages],
    }


# -------------------------------------------------------------------------
# API
# -------------------------------------------------------------------------
@app.post("/v1/tilt/generate")
def tilt_generate(req: TiltRequest = Body(...)) -> Dict[str, Any]:
    # кастомизация sampling
    sp = DEFAULT_SP
    if req.temperature is not None or req.max_tokens is not None:
        sp = SamplingParams(
            temperature=(
                req.temperature if req.temperature is not None else DEFAULT_SP.temperature
            ),
            max_tokens=(
                req.max_tokens if req.max_tokens is not None else DEFAULT_SP.max_tokens
            ),
            stop=DEFAULT_SP.stop,
        )

    # собираем TILT-input
    tilt_input = build_tilt_input(req)

    # ВАЖНО: используем ИМЕННО .tilt_generate, а не .generate
    outputs = llm.tilt_generate(inputs=[tilt_input], sampling_params=sp)

    # Берём первый текст из первого completion
    text = ""
    if outputs and getattr(outputs[0], "outputs", None):
        try:
            text = outputs[0].outputs[0].text
        except Exception:
            text = ""

    # Оборачиваем в OpenAI-like структуру
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
        "raw": (
            getattr(outputs[0], "outputs", None)[0].__dict__
            if outputs and getattr(outputs[0], "outputs", None)
            else None
        ),
    }


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME}


# -------------------------------------------------------------------------
# Минимальный warmup под tilt_generate
# -------------------------------------------------------------------------
try:
    warmup_input = {"question": "ping", "pages": []}
    llm.tilt_generate(inputs=[warmup_input], sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=8,
        stop=[],
    ))
    log.info("Warmup tilt_generate executed.")
except Exception as e:
    log.warning("Warmup tilt_generate failed but continuing startup: %s", e)
