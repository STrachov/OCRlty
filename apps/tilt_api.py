# /apps/tilt_api.py
from __future__ import annotations
import os, json, asyncio, logging
from typing import Any, Dict

from fastapi import FastAPI, Body
from pydantic import BaseModel
from vllm import LLM, SamplingParams

log = logging.getLogger("tilt_api")
logging.basicConfig(level=logging.INFO)

# --------- конфиг из ENV
MODEL_NAME      = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
DTYPE           = os.getenv("TILT_DTYPE", "float16")           # float16|bfloat16
TP_SIZE         = int(os.getenv("TILT_TP", "1"))
MAX_MODEL_LEN   = int(os.getenv("TILT_MAX_MODEL_LEN", "16384"))
HF_HOME         = os.getenv("HF_HOME", "/workspace/cache/hf")
os.environ.setdefault("HF_HOME", HF_HOME)

# Включаем XFormers по умолчанию
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_FLASH_ATTENTION", "0")

app = FastAPI(title="Arctic-TILT via vLLM", version="0.8.3")
llm: LLM | None = None

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95

@app.on_event("startup")
async def _startup():
    global llm
    log.info(
        "Starting TILT LLM with model=%s, dtype=%s, tp=%d, max_model_len=%d",
        MODEL_NAME, DTYPE, TP_SIZE, MAX_MODEL_LEN,
    )
    # dtype
    dtype = {"float16":"float16","bf16":"bfloat16","bfloat16":"bfloat16"}.get(DTYPE, "float16")

    # Инициализация LLM с task=tilt_generate (V0 engine внутри 0.8.3)
    llm = LLM(
        model=MODEL_NAME,
        task="tilt_generate",
        dtype=dtype,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        download_dir=HF_HOME,
        enforce_eager=True,
    )

    # Прогрев одной короткой итерацией
    sp = SamplingParams(max_tokens=1, temperature=0.1, top_p=0.95)
    test_llm = llm.generate(["hi"], sampling_params=sp)
    log.info(f"LLM initialized. test result: {str(test_llm)}")

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    assert llm is not None
    sp = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)
    prompt = llm.get_tokenizer().apply_chat_template(
        req.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm.generate([prompt], sampling_params=sp)
    # Простейший рендер в OpenAI‑похожий ответ
    text = outputs[0].outputs[0].text if outputs else ""
    return {
        "id": "chatcmpl-tilt",
        "object": "chat.completion",
        "model": req.model or MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }

@app.get("/v1/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}