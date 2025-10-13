from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict
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
    ct.strip() for ct in os.getenv(
        "ALLOWED_CONTENT_TYPES", "application/pdf,image/jpeg,image/png"
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
        VLLM_BASE_URL, TILT_MODEL, MOCK_VLLM, RULES_ENABLED,
    )
    tilt = ArcticTiltClient(
        base_url=VLLM_BASE_URL,
        model=TILT_MODEL,
        timeout=TILT_TIMEOUT_S,
        api_key=VLLM_API_KEY,
    )
    # Лёгкий ping /v1/models (не критично)
    try:
        with httpx.Client(timeout=5.0) as cli:
            r = cli.get(f"{VLLM_BASE_URL}/models", headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
            r.raise_for_status()
            models = r.json().get("data", [])
            ids = [m.get("id") for m in models if isinstance(m, dict)]
            if TILT_MODEL not in ids:
                log.warning("Model '%s' not in /models list: %s", TILT_MODEL, ids)
            else:
                log.info("vLLM ready; model found: %s", TILT_MODEL)
    except Exception as e:
        log.warning("vLLM /models ping failed: %s", e)

    yield

    try:
        if tilt and hasattr(tilt, "close"):
            tilt.close()  # type: ignore[attr-defined]
    except Exception as e:
        log.warning("Tilt client close failed: %s", e)


app = FastAPI(
    title="Arctic-TILT Inference API",
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


@app.get("/v1/health", tags=["system"])
def health() -> Dict[str, Any]:
    vllm_ok = False
    try:
        with httpx.Client(timeout=2.0) as cli:
            r = cli.get(f"{VLLM_BASE_URL}/models", headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
            r.raise_for_status()
            vllm_ok = True
    except Exception:
        vllm_ok = False

    return {
        "status": "ok",
        "vllm": {"base_url": VLLM_BASE_URL, "model": TILT_MODEL, "reachable": vllm_ok, "mock": MOCK_VLLM},
        "versions": {
            "api": app.version,
            "ruleset_version": os.getenv("RULESET_VERSION", "rules-0.1.0"),
            "model_version": TILT_MODEL,
        },
        "rules_enabled": RULES_ENABLED,
    }


@app.post("/v1/extract", tags=["inference"])
async def extract(file: UploadFile = File(...)) -> Dict[str, Any]:
    if tilt is None:
        raise HTTPException(status_code=503, detail="Model client not initialized")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    try:
        content = await file.read()
        if not content:
            raise ValueError("empty file")
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    request_id = str(uuid.uuid4())

    try:
        fields = tilt.infer(content)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TILT inference error: {e}")

    if RULES_ENABLED:
        try:
            fields = postprocess_rules(fields)
        except Exception as e:
            # если правила отвалились — вернём сырые поля, но не урони́м запрос
            log.warning("postprocess_rules failed: %s", e)

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
