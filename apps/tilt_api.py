# apps/tilt_api.py
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import itertools
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from PIL import Image

from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal.tilt_processor import (
    Document,
    Page,
    Question,
    TiltPreprocessor,
)
from vllm.outputs import RequestOutput


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
log = logging.getLogger("tilt_api")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))


# -------------------------------------------------------------------------
# Config (ENV)
# -------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")

# В официальном примере по умолчанию bfloat16, но оставим возможность
# переопределить через ENV.
DTYPE: str = os.getenv("TILT_DTYPE", "bfloat16")  # "float16" или "bfloat16"

TP_SIZE: int = int(os.getenv("TILT_TP", "1"))

# В примере для TILT Long используют очень длинный контекст (125000).
MAX_MODEL_LEN: int = int(os.getenv("TILT_MAX_LEN", "125000"))

GPU_UTIL: float = float(os.getenv("VLLM_GPU_UTIL",
                                  os.getenv("GPU_UTIL", "0.90")))
HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/cache/hf")
ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "1") in (
    "1",
    "true",
    "True",
)

# -------------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------------
app = FastAPI(title="Arctic-TILT API", version="1.0")

# -------------------------------------------------------------------------
# vLLM Engine + TiltPreprocessor
# -------------------------------------------------------------------------

# EngineArgs максимально близко к example_tilt.py
engine_args = EngineArgs(
    model=MODEL_NAME,
    task="tilt_generate",
    dtype=DTYPE,
    tensor_parallel_size=TP_SIZE,
    download_dir=HF_CACHE_DIR,
    gpu_memory_utilization=GPU_UTIL,
    enforce_eager=ENFORCE_EAGER,
    max_model_len=MAX_MODEL_LEN,
    # Критично: TILT-шедулер
    scheduler_cls="vllm.tilt.scheduler.Scheduler",
    # Как в примере:
    disable_async_output_proc=True,  # Not implemented in TILT scheduler
    disable_log_requests=True,
)

log.info(
    "TILT config: model=%s, dtype=%s, tp=%s, gpu_util=%.2f, max_model_len=%s",
    MODEL_NAME,
    DTYPE,
    TP_SIZE,
    GPU_UTIL,
    MAX_MODEL_LEN,
)

log.info("Creating LLMEngine with task=tilt_generate")
llm_engine = LLMEngine.from_engine_args(engine_args)

# Инициализируем препроцессор TILT (строго как в примере)
tokenizer = llm_engine.get_tokenizer()
preprocessor = TiltPreprocessor.from_config(
    model_config=llm_engine.model_config.hf_config,
    tokenizer=tokenizer.backend_tokenizer,
)

# Стриминговый LLMEngine не тред-сейф → оборачиваем в lock
_llm_lock = threading.Lock()
_request_counter = itertools.count()

# Дефолтные SamplingParams
DEFAULT_SP = SamplingParams(
    temperature=float(os.getenv("TILT_TEMPERATURE", "0.0")),
    max_tokens=int(os.getenv("TILT_MAX_TOKENS", "64")),
    logprobs=0,
)


# -------------------------------------------------------------------------
# Pydantic Schemas (входной контракт API)
# -------------------------------------------------------------------------
class OCRWord(BaseModel):
    text: str = Field(..., description="Recognized token text")
    bbox: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="[x0,y0,x1,y1] in pixels (absolute)",
    )


class OCRPage(BaseModel):
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")
    words: List[OCRWord] = Field(
        default_factory=list,
        description="Words with absolute bboxes in pixels",
    )


class InputPage(BaseModel):
    # Можно передавать:
    #  - только OCR (width/height/words/bboxes)
    #  - OCR + картинку (image_b64 или image_path)
    image_b64: Optional[str] = Field(
        None, description="Base64-encoded PNG/JPG of the page (optional)"
    )
    image_path: Optional[str] = Field(
        None,
        description="Filesystem path to page image mounted into container (optional)",
    )
    ocr: Optional[OCRPage] = Field(
        None, description="OCR result with word boxes in pixels"
    )


class TiltRequest(BaseModel):
    question: str = Field(
        ..., description="Doc-VQA / KIE question or instruction"
    )
    pages: List[InputPage] = Field(
        ..., description="List of page images and/or OCR data"
    )

    # Опциональные поля
    model: Optional[str] = Field(
        None, description="Override model name (optional, обычно не нужно)"
    )
    temperature: Optional[float] = Field(
        None, description="Override default temperature"
    )
    max_tokens: Optional[int] = Field(
        None, description="Override default max_tokens"
    )
    doc_id: Optional[str] = Field(
        None, description="Optional document identifier for debugging/logging"
    )


# -------------------------------------------------------------------------
# Helpers: конвертация JSON → Document/Page/Question
# -------------------------------------------------------------------------
def _decode_image_from_page(page: InputPage) -> Image.Image:
    """
    Получить PIL.Image для страницы:
      - если есть image_b64 → декодируем;
      - если есть image_path → открываем файл;
      - если ничего нет → создаём белый dummy image по размеру OCR (или 768x1086).
    """
    # 1) image_b64
    if page.image_b64:
        try:
            raw = base64.b64decode(page.image_b64, validate=False)
            return Image.open(io.BytesIO(raw)).convert("L")
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to decode image_b64: %s", e)

    # 2) image_path
    if page.image_path:
        try:
            img = Image.open(page.image_path)
            return img.convert("L")
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to open image_path %s: %s", page.image_path, e)

    # 3) dummy по размеру OCR или дефолт
    width = 768
    height = 1086
    if page.ocr:
        width = max(int(page.ocr.width), 1)
        height = max(int(page.ocr.height), 1)

    return Image.new(mode="L", size=(width, height), color=255)


def _page_to_tilt_page(page: InputPage) -> Page:
    """
    Конвертирует InputPage в vLLM Page, как это делает Loader в tilt_example.py.
    """
    if page.ocr:
        width = int(page.ocr.width)
        height = int(page.ocr.height)
        words = [w.text for w in page.ocr.words]
        # В примере bboxes берутся как есть из tokens_layer["positions"].
        bboxes = [list(map(float, w.bbox)) for w in page.ocr.words]
    else:
        # Без OCR у нас нет токенов, но картинку/заглушку всё равно передадим.
        img = _decode_image_from_page(page)
        width, height = img.size
        words = []
        bboxes = []

    img = _decode_image_from_page(page)

    return Page(
        words=words,
        bboxes=bboxes,
        width=width,
        height=height,
        image=img,
    )


def build_document_and_questions(
    req: TiltRequest,
) -> Tuple[Document, List[Question]]:
    """
    Строим Document + список Questions по входящему TiltRequest.
    """
    pages: List[Page] = []
    for p in req.pages:
        pages.append(_page_to_tilt_page(p))

    doc_ident = req.doc_id or "user-doc-1"
    document = Document(ident=doc_ident, split=None, pages=pages)

    # Для TILT question.feature_name использовали как "ключ поля",
    # а text — как текст вопроса. Нам важен только текст.
    questions: List[Question] = [
        Question(feature_name="user_question", text=req.question)
    ]

    return document, questions


# -------------------------------------------------------------------------
# Небольшой helper: выполнить один sample через LLMEngine
# -------------------------------------------------------------------------
def _run_single_sample(sample: Any, sp: SamplingParams) -> RequestOutput:
    """
    Запускает один sample через LLMEngine (add_request + step),
    блокируя общий движок через _llm_lock.
    """
    request_id = str(next(_request_counter))

    with _llm_lock:
        llm_engine.add_request(
            prompt=sample,
            request_id=request_id,
            params=sp,
        )

        final_output: Optional[RequestOutput] = None

        while True:
            request_outputs = llm_engine.step()
            for out in request_outputs:
                if out.request_id == request_id and out.finished:
                    final_output = out
                    break
            if final_output is not None:
                break

    if final_output is None:
        raise RuntimeError("LLMEngine.step() ended without a finished output")

    return final_output


# -------------------------------------------------------------------------
# API
# -------------------------------------------------------------------------
@app.post("/v1/tilt/generate")
def tilt_generate(req: TiltRequest = Body(...)) -> Dict[str, Any]:
    # 1) Сформировать SamplingParams с учётом оверрайдов
    sp = DEFAULT_SP
    if req.temperature is not None or req.max_tokens is not None:
        sp = SamplingParams(
            temperature=(
                req.temperature if req.temperature is not None else DEFAULT_SP.temperature
            ),
            max_tokens=(
                req.max_tokens if req.max_tokens is not None else DEFAULT_SP.max_tokens
            ),
            logprobs=DEFAULT_SP.logprobs,
        )

    # 2) Собрать Document + Questions и прогнать через TiltPreprocessor
    document, questions = build_document_and_questions(req)
    log.debug("Built Document ident=%s with %d pages", document.ident, len(document.pages))

    samples = preprocessor.preprocess(document, questions)
    if not samples:
        log.warning("TiltPreprocessor returned no samples for document %s", document.ident)
        text = ""
        final_output: Optional[RequestOutput] = None
    else:
        sample = samples[0]
        # 3) Прогоняем sample через LLMEngine (как в example_tilt, только синхронно)
        final_output = _run_single_sample(sample, sp)

        # 4) Достаём текст так же, как _transform_output в примере
        text = ""
        try:
            if final_output.outputs:
                text = final_output.outputs[0].text
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to extract text from RequestOutput: %s", e)
            text = ""

    # 5) Оборачиваем в OpenAI-подобный ответ
    raw_dict: Optional[Dict[str, Any]] = None
    if final_output is not None and final_output.outputs:
        try:
            raw_dict = final_output.outputs[0].__dict__
        except Exception:  # noqa: BLE001
            raw_dict = None

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
        "raw": raw_dict,
    }


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "max_model_len": MAX_MODEL_LEN,
    }


# -------------------------------------------------------------------------
# Опциональный warmup (микро-запрос через полный пайплайн)
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
    dummy_doc, dummy_questions = build_document_and_questions(dummy_req)
    dummy_samples = preprocessor.preprocess(dummy_doc, dummy_questions)
    if dummy_samples:
        _ = _run_single_sample(
            dummy_samples[0],
            SamplingParams(temperature=0.0, max_tokens=8, logprobs=0),
        )
        log.info("Warmup for TILT completed successfully.")
    else:
        log.warning("Warmup: preprocessor produced no samples, skipped.")
except Exception as e:  # noqa: BLE001
    log.warning("Warmup failed (non-fatal): %s", e)
