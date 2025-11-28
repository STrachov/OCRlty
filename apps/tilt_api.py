"""
FastAPI server for Arctic-TILT on vLLM.

Key design goals:
- Follow the official examples/tilt_example.py logic for building EngineArgs,
  TiltPreprocessor and for calling LLMEngine.add_request/step.
- Accept OCR-only inputs (no image required). When no image is provided, we use
  a dummy white image with the given page size so TiltPreprocessor is happy.
- Keep the API small and simple: a single /v1/tilt/generate endpoint.
"""

import logging
import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from PIL import Image

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import FlexibleArgumentParser

from vllm.multimodal.tilt_processor import (
    Document,
    Page,
    Question,
    TiltPreprocessor,
)

logger = logging.getLogger("tilt_api")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Configuration via env vars (with safe defaults)
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")

# Snowflake's example uses bfloat16; using float16 leads to numerical issues (NaNs)
# on long contexts with this model. Keep bf16 unless you REALLY need otherwise.
DTYPE = os.getenv("TILT_DTYPE", "bfloat16")

TP_SIZE = int(os.getenv("TILT_TP_SIZE", "1"))

GPU_UTIL = float(os.getenv("TILT_GPU_UTIL", "0.9"))

# Max model length. The model advertises 125k, but that's very memory heavy.
# You can override via env if needed.
MAX_MODEL_LEN = int(os.getenv("TILT_MAX_MODEL_LEN", "125000"))

DOWNLOAD_DIR = os.getenv("HF_HOME", "/workspace/cache/hf")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class OCRWord(BaseModel):
    text: str
    bbox: List[float] = Field(..., min_items=4, max_items=4)


class OCRContent(BaseModel):
    width: int
    height: int
    words: List[OCRWord]


class InputPage(BaseModel):
    # For now we only support OCR-only; later you can extend with image_url/base64.
    ocr: OCRContent


class TiltRequest(BaseModel):
    question: str
    pages: List[InputPage]
    temperature: float = 0.0
    max_tokens: int = 128


class TiltChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str


class TiltUsage(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class TiltResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[TiltChoice]
    usage: TiltUsage
    raw: Optional[dict] = None


# ---------------------------------------------------------------------------
# vLLM / TILT initialization
# ---------------------------------------------------------------------------

_parser = FlexibleArgumentParser()
AsyncEngineArgs.add_cli_args(_parser, description="Arctic-TILT vLLM engine args")

# Defaults aligned with examples/tilt_example.py
_parser.set_defaults(
    model=MODEL_NAME,
    task="tilt_generate",
    scheduler_cls="vllm.tilt.scheduler.Scheduler",
    gpu_memory_utilization=GPU_UTIL,
    dtype=DTYPE,
    max_model_len=MAX_MODEL_LEN,
    tensor_parallel_size=TP_SIZE,
    max_num_seqs=16,
    enforce_eager=True,
    # Disable the async output processor (not supported by V1 engine yet, so
    # vLLM will automatically fall back to the V0 engine – same as example).
    disable_async_output_proc=True,
)

_args = _parser.parse_args([])

engine_args = EngineArgs.from_cli_args(_args)

logger.info(
    "TILT config: model=%s, dtype=%s, tp=%d, gpu_util=%.2f, max_model_len=%d",
    engine_args.model,
    engine_args.dtype,
    engine_args.tensor_parallel_size,
    engine_args.gpu_memory_utilization,
    engine_args.max_model_len,
)

# Single global engine & preprocessor – created once at import time.
llm_engine: LLMEngine = LLMEngine.from_engine_args(engine_args)

# HuggingFace tokenizer wrapper.
_tokenizer = llm_engine.get_tokenizer()
_model_config = llm_engine.model_config

# Build TiltPreprocessor using HF config + tokenizer backend.
preprocessor = TiltPreprocessor.from_config(
    model_config=_model_config.hf_config,
    tokenizer=_tokenizer.backend_tokenizer,
)

# Reasonable default sampling; per-request overrides allowed.
DEFAULT_SP = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    logprobs=0,
)


# ---------------------------------------------------------------------------
# Helper: convert HTTP payload into TILT Document / Question objects
# ---------------------------------------------------------------------------

def _input_page_to_tilt_page(page: InputPage) -> Page:
    """Convert our InputPage into TILT's Page.

    If no image is available (our case), we create a dummy white image with the
    same resolution. TILT uses the image + the OCR tokens jointly, but the
    white background still allows it to run.
    """
    ocr = page.ocr
    words = [w.text for w in ocr.words]
    bboxes = [w.bbox for w in ocr.words]

    # Dummy white image; later you can replace with real page image if you have it.
    img = Image.new("RGB", (ocr.width, ocr.height), color=(255, 255, 255))

    return Page(
        words=words,
        bboxes=bboxes,
        width=ocr.width,
        height=ocr.height,
        image=img,
    )


def _build_document(req: TiltRequest) -> Document:
    pages = [_input_page_to_tilt_page(p) for p in req.pages]
    # We don't use dataset splits here, so split=None is fine.
    return Document(ident="user-doc", split=None, pages=pages)


def _build_questions(req: TiltRequest) -> List[Question]:
    # Arctic-TILT supports multiple questions per document; we use just one.
    q_text = req.question.strip()
    return [Question(key="q0", text=q_text)]


def _run_tilt_inference(req: TiltRequest) -> tuple[str, dict]:
    """Single-request inference loop following examples/tilt_example.py.

    It:
      1. builds Document + Question
      2. preprocesses into TiltSample(s)
      3. submits a single sample via llm_engine.add_request
      4. loops over llm_engine.step() until that request finishes
    """
    document = _build_document(req)
    questions = _build_questions(req)

    samples = preprocessor.preprocess(document, questions)
    if not samples:
        raise RuntimeError("TILT preprocessor returned 0 samples – check OCR input.")

    # We handle only one question, so we take the first sample.
    sample = samples[0]

    sp = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        logprobs=0,  # required by example_tilt.py / TILT task
    )

    request_id = f"tilt-{time.time_ns()}"
    llm_engine.add_request(
        prompt=sample,
        request_id=request_id,
        params=sp,
    )

    # Step loop – identical idea to examples/tilt_example.py: we iterate until
    # the request with our ID reports finished=True.
    while True:
        request_outputs = llm_engine.step()
        for out in request_outputs:
            if out.request_id != request_id:
                continue
            if not out.finished:
                # Not finished yet; continue stepping.
                continue

            # We expect a single output for this request.
            assert out.outputs, "No outputs from TILT request"
            text = out.outputs[0].text
            if text is None:
                text = ""

            # Return text and raw output for debugging.
            return text, {"output_repr": repr(out)}

        # No answer for us in this batch – sleep a tiny bit.
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Arctic-TILT API", version="0.1.0")


@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/tilt/generate", response_model=TiltResponse)
async def tilt_generate(req: TiltRequest):
    try:
        answer_text, raw = _run_tilt_inference(req)
    except Exception as exc:
        logger.exception("Error during TILT inference: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Build OpenAI-compatible-ish response.
    choice = TiltChoice(
        index=0,
        message={"role": "assistant", "content": answer_text},
        finish_reason="stop",
    )

    usage = TiltUsage(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    # id is not strictly important; we use a dummy.
    resp = TiltResponse(
        id="tiltcmpl-1",
        model=MODEL_NAME,
        choices=[choice],
        usage=usage,
        raw=raw,
    )
    return resp
