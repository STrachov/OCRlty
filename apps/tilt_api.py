"""
FastAPI server for Arctic-TILT on vLLM.

- Логика максимально близка к examples/tilt_example.py.
- Работает с OCR-only входом (слова + bbox). Если нет реальной картинки,
  создаётся белый dummy-Image нужного размера, чтобы TiltPreprocessor был доволен.
- Один основной endpoint: POST /v1/tilt/generate

Updates (2025-12):
- X-Request-ID middleware (generate/proxy correlation id)
- Structured JSON logging events
- Prometheus /metrics with stage timers (preprocess / infer / total) and error counters
- Use request_id to build vLLM request_id (instead of document.ident-q0)
- Remove noisy prints, avoid logging full payloads / OCR content
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
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




# -----------------------------------------------------------------------------
# Logging (shared structured logger)
# -----------------------------------------------------------------------------
from lib.utils.logging import get_event_logger

obs = get_event_logger("tilt_api", logger_name="tilt_api")

# -------------------------------------------------------------------------
# Prometheus metrics 
# -------------------------------------------------------------------------
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "0") == "1"
if PROMETHEUS_ENABLED:

    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total",
        "HTTP requests total",
        ["service", "path", "method", "status"],
    )
    HTTP_REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration (seconds)",
        ["service", "path", "method"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
    )
    TILT_STAGE_DURATION = Histogram(
        "pipeline_stage_duration_seconds",
        "Pipeline stage duration (seconds)",
        ["stage"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
    )
    TILT_ERRORS_TOTAL = Counter(
        "pipeline_errors_total",
        "Pipeline errors total",
        ["stage", "error_code"],
    )

# -------------------------------------------------------------------------
# Config from ENV
# -------------------------------------------------------------------------

MODEL_NAME: str = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
DTYPE: str = os.getenv("TILT_DTYPE", "bfloat16")
TP_SIZE: int = int(os.getenv("TILT_TP", os.getenv("TILT_TP_SIZE", "1")))
MAX_MODEL_LEN_ENV: Optional[str] = os.getenv("TILT_MAX_MODEL_LEN", None)
GPU_UTIL: float = float(os.getenv("TILT_GPU_UTIL", os.getenv("VLLM_GPU_UTIL", "0.9")))
HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/cache/hf")
ENFORCE_EAGER: bool = True
DEFAULT_TEMPERATURE: float = float(os.getenv("TILT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("TILT_MAX_TOKENS", "256"))

obs.log_event(
    "INFO",
    "gpu.startup_config",
    msg="TILT config loaded",
    tilt={"model": MODEL_NAME, "dtype": DTYPE, "tp_size": TP_SIZE, "gpu_util": GPU_UTIL},
)

# -------------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------------

app = FastAPI(title="Arctic-TILT API", version="1.1")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = rid

    start = time.time()
    status_code = 500
    try:
        response: Response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        dur = time.time() - start
        obs.log_event(
            "INFO",
            "gpu.http_done",
            request_id=rid,
            http={"method": request.method, "path": request.url.path, "status": status_code},
            duration_ms=round(dur * 1000, 2),
        )
        if PROMETHEUS_ENABLED:
            HTTP_REQUESTS_TOTAL.labels("tilt_api", request.url.path, request.method, str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels("tilt_api", request.url.path, request.method).observe(dur)


# -------------------------------------------------------------------------
# vLLM Engine + TiltPreprocessor
# -------------------------------------------------------------------------


def _build_llm_engine() -> Tuple[LLMEngine, TiltPreprocessor]:
    parser = FlexibleArgumentParser(description="Arctic-TILT vLLM engine (behind FastAPI).")
    parser = AsyncEngineArgs.add_cli_args(parser, async_args_only=False)

    parser.set_defaults(
        model=MODEL_NAME,
        task="tilt_generate",
        scheduler_cls="vllm.tilt.scheduler.Scheduler",
        gpu_memory_utilization=GPU_UTIL,
        dtype=DTYPE,
        max_num_seqs=16,
        enforce_eager=ENFORCE_EAGER,
        disable_async_output_proc=True,
    )

    args = parser.parse_args([])
    args.tensor_parallel_size = TP_SIZE
    args.download_dir = HF_CACHE_DIR

    if MAX_MODEL_LEN_ENV:
        try:
            args.max_model_len = int(MAX_MODEL_LEN_ENV)
        except ValueError:
            obs.log_event("WARNING", "gpu.config_invalid", msg=f"Invalid TILT_MAX_MODEL_LEN={MAX_MODEL_LEN_ENV}, ignoring.")

    engine_args = EngineArgs.from_cli_args(args)
    obs.log_event("INFO", "gpu.engine_create", msg=f"Creating LLMEngine task={engine_args.task}")

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
# Pydantic request models
# -------------------------------------------------------------------------


class OCRWord(BaseModel):
    text: str = Field(..., description="Recognized token text")
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="[x0,y0,x1,y1] in pixels")


class OCRPage(BaseModel):
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")
    words: List[OCRWord] = Field(default_factory=list, description="Words with absolute bboxes")


class InputPage(BaseModel):
    ocr: Optional[OCRPage] = Field(None, description="OCR result with word boxes in pixels")
    image_b64: Optional[str] = Field(None, description="Base64-encoded PNG/JPG of the page (optional).")
    image_path: Optional[str] = Field(None, description="Path to page image inside container (optional).")


class TiltRequest(BaseModel):
    question: str = Field(..., description="Doc-VQA / KIE question or instruction")
    pages: List[InputPage] = Field(..., description="List of pages with OCR and/or images")
    model: Optional[str] = Field(None, description="Override model name (optional).")
    temperature: Optional[float] = Field(None, description="Overrides default temperature.")
    max_tokens: Optional[int] = Field(None, description="Overrides default max tokens.")
    request_id: Optional[str] = Field(None, description="Correlation id (optional)")


# -------------------------------------------------------------------------
# Helpers: InputPage -> TILT Page / Document / Question
# -------------------------------------------------------------------------


def _decode_image(page: InputPage) -> Image.Image:
    if page.image_b64:
        try:
            raw = base64.b64decode(page.image_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            obs.log_event("WARNING", "gpu.image_decode_failed", msg=str(exc))

    if page.image_path:
        try:
            return Image.open(page.image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            obs.log_event("WARNING", "gpu.image_open_failed", msg=f"path={page.image_path} err={exc}")

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
        width, height = img.size
        words = []
        bboxes = []

    return Page(words=words, bboxes=bboxes, width=width, height=height, image=img)


def _build_document(req: TiltRequest, request_id: str) -> Document:
    doc_id = request_id

    tilt_pages: List[Page] = []
    total_words = 0
    for idx, p in enumerate(req.pages):
        tilt_page = _input_page_to_tilt_page(p)
        if not tilt_page.words:
            obs.log_event(
                "WARNING",
                "gpu.ocr_page_skipped",
                request_id=request_id,
                msg=f"Skipping page {idx}: no OCR words",
                ocr={"page_idx": idx, "width": getattr(tilt_page, "width", None), "height": getattr(tilt_page, "height", None)},
            )
            continue
        total_words += len(tilt_page.words)
        tilt_pages.append(tilt_page)

    if not tilt_pages:
        raise HTTPException(status_code=400, detail="No valid OCR words found on any page for TILT.")

    obs.log_event("INFO", "gpu.document_built", request_id=request_id, doc={"pages": len(tilt_pages)}, ocr={"words_total": total_words})
    return Document(ident=doc_id, split=None, pages=tilt_pages)


def _build_questions(req: TiltRequest) -> List[Question]:
    return [Question(feature_name="user_question", text=req.question.strip())]


# -------------------------------------------------------------------------
# Core inference
# -------------------------------------------------------------------------


def _run_tilt_inference(req: TiltRequest, request_id: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    if not req.pages:
        raise HTTPException(status_code=400, detail="Request must contain at least one page.")

    t0 = time.time()
    document = _build_document(req, request_id=request_id)
    questions = _build_questions(req)

    # preprocess
    tp0 = time.time()
    try:
        samples = preprocessor.preprocess(document, questions)
    except IndexError as exc:
        if PROMETHEUS_ENABLED:
            TILT_ERRORS_TOTAL.labels("preprocess", "INDEX_ERROR").inc()
        obs.log_event(
            "ERROR",
            "gpu.preprocess_failed",
            request_id=request_id,
            msg="IndexError in TiltPreprocessor.preprocess (likely no tokens produced from OCR/bboxes).",
            error={"type": "IndexError", "message": str(exc)},
        )
        raise HTTPException(status_code=500, detail="TILT preprocessing failed: no page tokens produced from OCR.") from exc
    except Exception as exc:  # noqa: BLE001
        if PROMETHEUS_ENABLED:
            TILT_ERRORS_TOTAL.labels("preprocess", "EXCEPTION").inc()
        obs.log_event(
            "ERROR",
            "gpu.preprocess_failed",
            request_id=request_id,
            msg="TiltPreprocessor.preprocess failed.",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise HTTPException(status_code=500, detail=f"TILT preprocessing failed: {exc}") from exc

    preprocess_s = time.time() - tp0
    if PROMETHEUS_ENABLED:
        TILT_STAGE_DURATION.labels("preprocess").observe(preprocess_s)

    if not samples:
        obs.log_event("WARNING", "gpu.preprocess_empty", request_id=request_id, msg="preprocess returned empty samples")
        return "", None, {"preprocess_s": preprocess_s, "infer_s": 0.0, "total_s": time.time() - t0}

    sample = samples[0]

    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=0)

    vllm_request_id = f"{request_id}-q0"

    infer0 = time.time()
    final_output = None

    with _engine_lock:
        llm_engine.add_request(prompt=sample, request_id=vllm_request_id, params=sampling_params)

        while True:
            request_outputs = llm_engine.step()
            if not request_outputs:
                continue

            for out in request_outputs:
                if out.request_id != vllm_request_id:
                    continue
                if not out.finished:
                    continue
                final_output = out
                break

            if final_output is not None:
                break

    infer_s = time.time() - infer0
    if PROMETHEUS_ENABLED:
        TILT_STAGE_DURATION.labels("infer").observe(infer_s)

    if final_output is None:
        if PROMETHEUS_ENABLED:
            TILT_ERRORS_TOTAL.labels("infer", "NO_OUTPUT").inc()
        obs.log_event("WARNING", "gpu.infer_no_output", request_id=request_id, msg="No RequestOutput produced")
        return "", None, {"preprocess_s": preprocess_s, "infer_s": infer_s, "total_s": time.time() - t0}

    outputs = getattr(final_output, "outputs", None) or []
    if not outputs:
        if PROMETHEUS_ENABLED:
            TILT_ERRORS_TOTAL.labels("infer", "EMPTY_OUTPUTS").inc()
        obs.log_event("WARNING", "gpu.infer_empty_outputs", request_id=request_id, msg="RequestOutput.outputs is empty")
        return "", repr(final_output), {"preprocess_s": preprocess_s, "infer_s": infer_s, "total_s": time.time() - t0}

    first = outputs[0]
    debug_repr = repr(first)

    try:
        text = first.text
    except Exception as exc:  # noqa: BLE001
        if PROMETHEUS_ENABLED:
            TILT_ERRORS_TOTAL.labels("infer", "TEXT_READ_FAILED").inc()
        obs.log_event("WARNING", "gpu.infer_text_read_failed", request_id=request_id, msg=str(exc))
        return "", debug_repr, {"preprocess_s": preprocess_s, "infer_s": infer_s, "total_s": time.time() - t0}

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    total_s = time.time() - t0
    if PROMETHEUS_ENABLED:
        TILT_STAGE_DURATION.labels("total").observe(total_s)

    preview = text[:160]
    obs.log_event(
        "INFO",
        "gpu.infer_done",
        request_id=request_id,
        tilt={
            "model": req.model or MODEL_NAME,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output_chars": len(text),
            "output_preview": preview,
        },
        duration_ms=round(total_s * 1000, 2),
    )

    return text, debug_repr, {"preprocess_s": preprocess_s, "infer_s": infer_s, "total_s": total_s}


# -------------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------------


@app.post("/v1/tilt/generate")
def tilt_generate(
    req: TiltRequest = Body(...),
    request: Request = None,
    x_request_id: Optional[str] = Header(None),
) -> Dict[str, Any]:
    rid = x_request_id or getattr(getattr(request, "state", None), "request_id", None) or req.request_id or uuid.uuid4().hex

    obs.log_event("INFO", "gpu.request_received", request_id=rid, doc={"pages_in": len(req.pages)}, tilt={"model": req.model or MODEL_NAME})

    try:
        text, debug_repr, meta = _run_tilt_inference(req, request_id=rid)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        if PROMETHEUS_ENABLED:
            TILT_ERRORS_TOTAL.labels("total", "UNHANDLED").inc()
        obs.log_event("ERROR", "gpu.unhandled_error", request_id=rid, error={"type": type(exc).__name__, "message": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": f"tiltcmpl-{rid}",
        "object": "chat.completion",
        "model": req.model or MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "raw": {"output_repr": debug_repr, "meta": meta} if debug_repr is not None else {"meta": meta},
    }


@app.get("/v1/health")
def health(request: Request) -> Dict[str, Any]:
    rid = getattr(getattr(request, "state", None), "request_id", None) or uuid.uuid4().hex
    obs.log_event("INFO", "gpu.health", request_id=rid)
    return {"status": "ok", "model": MODEL_NAME, "dtype": DTYPE, "gpu_util": GPU_UTIL, "tp_size": TP_SIZE}


@app.get("/metrics")
def metrics() -> Response:
    if not PROMETHEUS_ENABLED:
        return PlainTextResponse("PROMETHEUS_ENABLED=0\n", status_code=404)
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
