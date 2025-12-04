from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from lib.pipelines.tilt_client import ArcticTiltClient
from lib.post.rules import postprocess_rules  # теперь обязательная зависимость

log = logging.getLogger("api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# === ENV ===
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8001/v1").rstrip("/")
TILT_MODEL = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
TILT_TIMEOUT_S = float(os.getenv("TILT_TIMEOUT_S", "10.0"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")
MOCK_VLLM = os.getenv("MOCK_VLLM", "0") == "1"

ENABLE_CORS = os.getenv("ENABLE_CORS", "1") == "1"
CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_CONTENT_TYPES = {
    ct.strip()
    for ct in os.getenv(
        "ALLOWED_CONTENT_TYPES",
        "application/pdf,image/jpeg,image/png",
    ).split(",")
}

# Позволяет временно отключить правила без изменений кода
RULES_ENABLED = os.getenv("RULES_ENABLED", "1") == "1"

tilt: ArcticTiltClient | None = None  # singleton


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/Shutdown через lifespan."""
    global tilt
    log.info(
        "Starting API; VLLM_BASE_URL=%s, MODEL=%s, MOCK_VLLM=%s, RULES_ENABLED=%s",
        VLLM_BASE_URL,
        TILT_MODEL,
        MOCK_VLLM,
        RULES_ENABLED,
    )
    tilt = ArcticTiltClient(
        base_url=VLLM_BASE_URL,
        model=TILT_MODEL,
        timeout=TILT_TIMEOUT_S,
        api_key=VLLM_API_KEY,
    )
    # Лёгкий ping tilt_api /v1/health (не критично для старта)
    try:
        with httpx.Client(timeout=5.0) as cli:
            r = cli.get(f"{VLLM_BASE_URL}/health")
            r.raise_for_status()
            health = r.json()
            log.info(
                "tilt_api ready; model=%s, dtype=%s, tp=%s",
                health.get("model"),
                health.get("dtype"),
                health.get("tp_size"),
            )
    except Exception as e:  # noqa: BLE001
        log.warning("tilt_api /health ping failed: %s", e)

    yield

    try:
        if tilt is not None:
            tilt.close()
    except Exception as e:  # noqa: BLE001
        log.warning("Error while closing ArcticTiltClient: %s", e)


app = FastAPI(
    title="OCRlty Arctic-TILT API (GPU)",
    version=os.getenv("API_VERSION", "0.1.0"),
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/v1/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """Проверка живости API + proxied health от tilt_api."""
    tilt_ok = False
    tilt_info: Dict[str, Any] | None = None
    try:
        with httpx.Client(timeout=2.0) as cli:
            r = cli.get(f"{VLLM_BASE_URL}/health")
            r.raise_for_status()
            tilt_info = r.json()
            tilt_ok = True
    except Exception as e:  # noqa: BLE001
        tilt_ok = False
        tilt_info = {"error": str(e)}

    return {
        "status": "ok",
        "tilt": {
            "base_url": VLLM_BASE_URL,
            "model": TILT_MODEL,
            "reachable": tilt_ok,
            "mock": MOCK_VLLM,
            "info": tilt_info,
        },
        "versions": {
            "api": app.version,
            "ruleset_version": os.getenv("RULESET_VERSION", "rules-0.1.0"),
            "model_version": TILT_MODEL,
        },
        "rules_enabled": RULES_ENABLED,
    }


@app.post("/v1/extract", tags=["inference"])
async def extract(file: UploadFile = File(...), question: Optional[str] = None) -> Dict[str, Any]:
    if tilt is None:
        raise HTTPException(status_code=503, detail="Model client not initialized")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes > {max_bytes} bytes",
        )

    request_id = uuid.uuid4().hex
    log.info(
        "Request %s: filename=%s, content_type=%s, size=%d bytes",
        request_id,
        file.filename,
        file.content_type,
        len(content),
    )

    try:
        # важно передать content_type, чтобы корректно определить PDF vs image
        fields = tilt.infer(content, content_type=file.content_type or None, question=question)
    except Exception as e:  # noqa: BLE001
        log.exception("TILT inference failed for request %s: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"TILT inference failed: {e}") from e

    if RULES_ENABLED:
        try:
            fields = postprocess_rules(fields)
        except Exception as e:  # noqa: BLE001
            # если правила отвалились — вернём сырые поля, но не уроним запрос
            log.warning("postprocess_rules failed for request %s: %s", request_id, e)

    return {
        "data": fields,
        "meta": {
            "request_id": request_id,
            "model_version": TILT_MODEL,
            "ruleset_version": os.getenv("RULESET_VERSION", "rules-0.1.0"),
            "source_file": file.filename,
            "mock_vllm": MOCK_VLLM,
        },
    }
