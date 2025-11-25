from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log = logging.getLogger("tilt_api")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# -----------------------------------------------------------------------------
# Config (ENV)
# -----------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
DTYPE: str = os.getenv("TILT_DTYPE", "float16")                         # float16|bfloat16
TP_SIZE: int = int(os.getenv("TILT_TP", "1"))
MAX_MODEL_LEN: int = int(os.getenv("TILT_MAX_LEN", "16384"))
GPU_UTIL: float = float(os.getenv("VLLM_GPU_UTIL", os.getenv("GPU_UTIL", "0.90")))
HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/cache/hf")
ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "1") in ("1", "true", "True")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Arctic-TILT API", version="1.0")

# -----------------------------------------------------------------------------
# vLLM: используем обычный generate-runner
# -----------------------------------------------------------------------------
log.info(
    "Starting LLM with model=%s, dtype=%s, tp=%s, max_model_len=%s",
    MODEL_NAME, DTYPE, TP_SIZE, MAX_MODEL_LEN
)

# ВАЖНО: task НЕ задаём → используется стандартный runner "generate"
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    dtype=DTYPE,                       # torch_dtype устарел, используем dtype
    tensor_parallel_size=TP_SIZE,
    max_model_len=MAX_MODEL_LEN,
    download_dir=HF_CACHE_DIR,
    gpu_memory_utilization=GPU_UTIL,
    enforce_eager=ENFORCE_EAGER,
)

DEFAULT_SP = SamplingParams(
    temperature=float(os.getenv("TILT_TEMPERATURE", "0.0")),
    max_tokens=int(os.getenv("TILT_MAX_TOKENS", "256")),
    stop=[],
)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class OCRWord(BaseModel):
    text: str = Field(..., description="Recognized token text")
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="[x0,y0,x1,y1] in pixels")

class OCRPage(BaseModel):
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")
    words: List[OCRWord] = Field(default_factory=list, description="Words with absolute bboxes")

class InputPage(BaseModel):
    image_b64: Optional[str] = Field(None, description="Base64-encoded PNG/JPG of the page")
    image_path: Optional[str] = Field(None, description="Filesystem path to a PNG/JPG page image")
    ocr: Optional[OCRPage] = Field(None, description="OCR result with word boxes in pixels")

class TiltRequest(BaseModel):
    question: str = Field(..., description="Doc-VQA / KIE question or instruction")
    pages: List[InputPage] = Field(..., description="List of page images and/or OCR data")
    model: Optional[str] = Field(None, description="Override model name (optional)")
    temperature: Optional[float] = Field(None, description="Overrides default temperature")
    max_tokens: Optional[int] = Field(None, description="Overrides default max tokens")

# -----------------------------------------------------------------------------
# Helpers: собираем текстовый промпт из структуры документа
# -----------------------------------------------------------------------------
def build_prompt_from_request(req: TiltRequest) -> str:
    """Упаковываем структуру (question + OCR) в текстовый промпт."""
    lines: List[str] = []
    lines.append(
        "You are an AI assistant specialized in understanding receipts, "
        "invoices and other business documents based on OCR text with "
        "bounding boxes."
    )
    lines.append(
        "You are given the OCR outputs of a document. Use the text and "
        "their relative positions only as hints; you do not have access "
        "to the original image."
    )

    for page_idx, page in enumerate(req.pages, start=1):
        lines.append(f"\n[Page {page_idx}]")
        if page.ocr:
            lines.append(f"size: width={page.ocr.width}, height={page.ocr.height}")
            for i, w in enumerate(page.ocr.words, start=1):
                x0, y0, x1, y1 = w.bbox
                lines.append(
                    f"{i}. text={w.text!r}, bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
                )
        elif page.image_path:
            lines.append(f"image_path={page.image_path} (no OCR words provided)")
        elif page.image_b64:
            lines.append("image_b64 provided (no OCR words provided)")
        else:
            lines.append("no OCR or image data provided for this page")

    lines.append("\nQuestion:")
    lines.append(req.question)
    lines.append("\nAnswer concisely and precisely based only on the OCR text above.")

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.post("/v1/tilt/generate")
def tilt_generate(req: TiltRequest = Body(...)) -> Dict[str, Any]:
    # Sampling overrides
    sp = DEFAULT_SP
    if req.temperature is not None or req.max_tokens is not None:
        sp = SamplingParams(
            temperature=req.temperature if req.temperature is not None else DEFAULT_SP.temperature,
            max_tokens=req.max_tokens if req.max_tokens is not None else DEFAULT_SP.max_tokens,
            stop=DEFAULT_SP.stop,
        )

    prompt = build_prompt_from_request(req)

    outputs = llm.generate(
        [prompt],
        sampling_params=sp,
    )

    text = ""
    raw = None
    if outputs and getattr(outputs[0], "outputs", None):
        try:
            text = outputs[0].outputs[0].text
            raw = outputs[0].outputs[0].__dict__
        except Exception:
            text = ""

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
        "raw": raw,
    }

@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME}

# -----------------------------------------------------------------------------
# Тихий warmup
# -----------------------------------------------------------------------------
try:
    warmup_prompt = "Warmup: answer with a short word."
    llm.generate(
        [warmup_prompt],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )
    log.info("Warmup generate executed.")
except Exception as e:
    log.warning("Warmup generate failed but continuing startup: %s", e)
