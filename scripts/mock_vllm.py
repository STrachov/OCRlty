# scripts/mock_vllm.py
from __future__ import annotations

import os
import re
import json
import time
import hashlib
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Config (via ENV) --------------------
MOCK_MODEL_ID = os.getenv("MOCK_MODEL_ID", "Snowflake/snowflake-arctic-tilt-v1.3")
MOCK_DELAY_MS = int(os.getenv("MOCK_VLLM_DELAY_MS", "0"))           # задержка ответа, мс
RETURN_CODEBLOCK = os.getenv("MOCK_VLLM_CODEBLOCK", "0") == "1"     # оборачивать ответ в ```json
FIXED_MERCHANT = os.getenv("MOCK_VLLM_MERCHANT", "MOCK_MERCHANT")
CURRENCY = os.getenv("MOCK_VLLM_CURRENCY", "USD")
TAX_RATE = float(os.getenv("MOCK_VLLM_TAX", "0.08"))                # 8% по умолчанию

# -------------------- Schemas --------------------
# class Message(BaseModel):
#     role: str = Field(..., description="user|assistant|system")
#     content: str
# from typing import Union

class Message(BaseModel):
    role: str = Field(..., description="user|assistant|system")
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: Optional[str] = MOCK_MODEL_ID
    messages: List[Message]
    temperature: Optional[float] = 0.0
    # оставляем поле для совместимости; игнорируется мок-сервером
    stream: Optional[bool] = False

# -------------------- App --------------------
app = FastAPI(title="Mock vLLM (OpenAI-compatible)")

# Разрешим CORS на всякий случай
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def root():
    return {"ok": True, "service": "mock-vllm", "model": MOCK_MODEL_ID}

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MOCK_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock",
            }
        ],
    }

# -------------------- Helpers --------------------
_len_rx = re.compile(r"len\s*=\s*(\d+)\s*bytes", re.IGNORECASE)

def _content_to_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)

def _synthesize_fields(messages: List[Message]) -> Dict[str, Any]:
    base_len = None
    for m in messages:
        text = _content_to_text(m.content)
        mt = _len_rx.search(text)
        if mt:
            base_len = int(mt.group(1))
            break
    if base_len is None:
        concat = "|".join(f"{m.role}:{_content_to_text(m.content)}" for m in messages)
        base_len = int(hashlib.md5(concat.encode("utf-8")).hexdigest()[:6], 16)

    # Синтетические суммы (детерминированные)
    subtotal = round((base_len % 10000) / 100.0 + 1.0, 2)  # от 1.00 до 101.00
    tax_amount = round(subtotal * TAX_RATE, 2)
    total = round(subtotal + tax_amount, 2)

    # Простая дата — фикс, можно сделать «сегодня»
    result = {
        "merchant": FIXED_MERCHANT,
        "date": "2024-01-01",
        "currency": CURRENCY,
        "subtotal": float(subtotal),
        "tax_amount": float(tax_amount),
        "total": float(total),
    }
    return result

def _wrap_content(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    if RETURN_CODEBLOCK:
        return f"```json\n{text}\n```"
    return text

def _usage_stub(messages: List[Message], content: str) -> Dict[str, int]:
    prompt_tokens = sum(len(_content_to_text(m.content).split()) for m in messages)
    completion_tokens = len(content.split())

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

# -------------------- Endpoints --------------------
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # Имитируем задержку
    if MOCK_DELAY_MS > 0:
        time.sleep(MOCK_DELAY_MS / 1000.0)

    model_id = req.model or MOCK_MODEL_ID
    payload = _synthesize_fields(req.messages)
    content = _wrap_content(payload)

    now = int(time.time())
    return {
        "id": f"chatcmpl-mock-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_stub(req.messages, content),
    }

# -------------------- CLI --------------------
if __name__ == "__main__":
    # Локальный запуск: python scripts/mock_vllm.py
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("scripts.mock_vllm:app", host="0.0.0.0", port=port, reload=True)
