from __future__ import annotations

import json
import os
import time
import uuid
import secrets
import logging
from typing import Any, Dict, Optional, Set, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from lib.pipelines.tilt_client import ArcticTiltClient
from lib.post.rules import postprocess_rules  # теперь обязательная зависимость


# -----------------------------------------------------------------------------
# Logging (shared structured logger)
# -----------------------------------------------------------------------------
from lib.utils.logging import get_event_logger

obs = get_event_logger("api")

# -----------------------------------------------------------------------------
# Prometheus metrics
# -----------------------------------------------------------------------------
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "0") == "1"
if PROMETHEUS_ENABLED:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["service", "path", "method", "status"],
    )
    HTTP_REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["service", "path", "method"],
    )

    PIPELINE_STAGE_DURATION = Histogram(
        "pipeline_stage_duration_seconds",
        "Pipeline stage duration in seconds",
        ["stage"],
    )

    PIPELINE_ERRORS_TOTAL = Counter(
        "pipeline_errors_total",
        "Pipeline errors by code",
        ["stage", "error_code"],
    )


# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8001/v1").rstrip("/")
TILT_MODEL = os.getenv("TILT_MODEL", "Snowflake/snowflake-arctic-tilt-v1.3")
TILT_TIMEOUT_S = float(os.getenv("TILT_TIMEOUT_S", "10.0"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")
MOCK_VLLM = os.getenv("MOCK_VLLM", "0") == "1"

ENABLE_CORS = os.getenv("ENABLE_CORS", "1") == "1"
CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "1") == "1"

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "1") == "1"
API_KEYS_JSON_DEFAULT = '{"k_infer_123":{"id":"infer-1","scopes":["infer"]},"k_debug_456":{"id":"debug-1","scopes":["infer","debug"]}}'
API_KEYS_JSON = os.getenv("API_KEYS_JSON", API_KEYS_JSON_DEFAULT).strip()

def _load_api_keys() -> Dict[str, Dict[str, Any]]:
    """Load API keys from env.

    Expected format (JSON object):
      {
        "<api_key>": {"id": "infer-1", "scopes": ["infer"]},
        "<api_key>": {"id": "debug-1", "scopes": ["infer", "debug"]}
      }
    """
    if not API_KEYS_JSON:
        return {}
    try:
        data = json.loads(API_KEYS_JSON)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("api").error("Failed to parse API_KEYS_JSON: %s", e)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (data or {}).items():
        if not isinstance(k, str) or not k:
            continue
        scopes = v.get("scopes", []) if isinstance(v, dict) else []
        if not isinstance(scopes, list):
            scopes = []
        out[k] = {
            "id": (v.get("id") if isinstance(v, dict) else None) or "unknown",
            "scopes": set(str(s) for s in scopes),
        }
    return out

_API_KEYS: Dict[str, Dict[str, Any]] = _load_api_keys()
_API_KEY_LIST: List[str] = list(_API_KEYS.keys())

def _require_api_key(request: Request, required_scope: str) -> Dict[str, Any]:
    """Authenticate and authorize request by X-API-Key + scope.

    - Always expects header: X-API-Key
    - required_scope: "infer" or "debug"
    """
    if AUTH_ENABLED:
        if not _API_KEY_LIST:
            raise HTTPException(status_code=503, detail="Auth not configured (API_KEYS_JSON is empty)")
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")

        rec: Optional[Dict[str, Any]] = None
        # Constant-time match against configured keys (expected small N).
        for k in _API_KEY_LIST:
            if secrets.compare_digest(api_key, k):
                rec = _API_KEYS[k]
                break
        if rec is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        scopes: Set[str] = rec.get("scopes", set())
        if required_scope not in scopes:
            raise HTTPException(status_code=403, detail="Forbidden")

        return rec

    # AUTH disabled (dev only)
    return {"id": "auth_disabled", "scopes": set(["infer", "debug"])}
  # optional; if set, require X-Debug-Token header for eval-mode traces
ALLOWED_CONTENT_TYPES = {
    ct.strip()
    for ct in os.getenv(
        "ALLOWED_CONTENT_TYPES",
        "application/pdf,image/jpeg,image/png",
    ).split(",")
}

# Позволяет временно отключить правила без изменений кода
RULES_ENABLED = os.getenv("RULES_ENABLED", "0") == "1"

tilt: ArcticTiltClient | None = None  # singleton


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/Shutdown через lifespan."""
    global tilt

    obs.log_event(
        "info",
        "api.starting",
        request_id=None,
        config={
            "vllm_base_url": VLLM_BASE_URL,
            "model": TILT_MODEL,
            "mock_vllm": MOCK_VLLM,
            "rules_enabled": RULES_ENABLED,
        },
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
            obs.log_event(
                "info",
                "tilt_api.health_ok",
                model=health.get("model"),
                dtype=health.get("dtype"),
                tp_size=health.get("tp_size"),
            )
    except Exception as e:  # noqa: BLE001
        obs.log_event("warning", "tilt_api.health_failed", error={"type": type(e).__name__, "message": str(e)})

    yield

    try:
        if tilt is not None:
            tilt.close()
    except Exception as e:  # noqa: BLE001
        obs.log_event("warning", "api.shutdown_close_client_failed", error={"type": type(e).__name__, "message": str(e)})


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


# -----------------------------------------------------------------------------
# Middleware: Request-ID + HTTP metrics + structured access log
# -----------------------------------------------------------------------------
@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    path = request.url.path
    method = request.method

    t0 = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = int(getattr(response, "status_code", 200))
    except HTTPException as e:
        status = int(e.status_code)
        obs.log_event(
            "warning",
            "api.http_exception",
            request_id=request_id,
            http={"method": method, "path": path, "status": status},
            error={"type": "HTTPException", "message": str(e.detail)},
        )
        response = JSONResponse(status_code=status, content={"detail": e.detail})
    except Exception as e:  # noqa: BLE001
        status = 500
        obs.log_event(
            "error",
            "api.unhandled_exception",
            request_id=request_id,
            http={"method": method, "path": path, "status": status},
            error={"type": type(e).__name__, "message": str(e)},
        )
        response = JSONResponse(status_code=500, content={"detail": "Internal Server Error"}) #TODO: replace the string with e?
    finally:
        dt = time.perf_counter() - t0
        if PROMETHEUS_ENABLED:
            HTTP_REQUESTS_TOTAL.labels("api", path, method, str(status)).inc()
            HTTP_REQUEST_DURATION.labels("api", path, method).observe(dt)

        # structured access log (only for API calls, keep it concise)
        obs.log_event(
            "info",
            "api.request_done",
            request_id=request_id,
            http={"method": method, "path": path, "status": status},
            duration_ms=round(dt * 1000.0, 3),
        )

    # Add correlation header (always)
    response.headers["X-Request-ID"] = request_id
    return response


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    if not PROMETHEUS_ENABLED:
        raise HTTPException(status_code=404, detail="Prometheus metrics disabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
async def extract(
    request: Request,
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    field_name: Optional[str] = Form(None),
    max_candidates: Optional[int] = Form(None),
    max_neighbours: Optional[int] = Form(None),
    sample_id: Optional[str] = Form(None),
    gt_raw: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Unified inference endpoint.

    - Normal inference (no trace): call without `sample_id`.
    - Eval/debug trace mode: provide `sample_id` (non-empty). In this mode the response includes `trace`
      with OCR words (incl. score), candidates+anchors, raw model output text, and timings.

    NOTE: Eval/debug trace mode is gated by ENABLE_DEBUG_ENDPOINTS.
    """
    if tilt is None:
        if PROMETHEUS_ENABLED:
            PIPELINE_ERRORS_TOTAL.labels("api", "MODEL_CLIENT_NOT_INITIALIZED").inc()
        raise HTTPException(status_code=503, detail="Model client not initialized")

    request_id: str = getattr(request.state, "request_id", uuid.uuid4().hex)

    eval_mode = bool(sample_id and str(sample_id).strip())
    required_scope = "debug" if eval_mode else "infer"
    auth = _require_api_key(request, required_scope=required_scope)
    key_id = auth.get("id", "unknown")
    if eval_mode:
        # Gate eval-mode to prevent accidental PII leakage in production.
        if not ENABLE_DEBUG_ENDPOINTS:
            raise HTTPException(status_code=404, detail="Not found")

    # If question is not provided, require env default to be set
    if not question and not os.getenv("TILT_KIE_PROMPT"):
        if PROMETHEUS_ENABLED:
            PIPELINE_ERRORS_TOTAL.labels("api", "MISSING_QUESTION").inc()
        raise HTTPException(status_code=400, detail="Missing 'question' (and TILT_KIE_PROMPT is not set)")

    # read file bytes early to compute size
    content = await file.read()
    size_bytes = len(content)

    obs.log_event(
        "info",
        "api.extract_received",
        request_id=request_id,
        mode="eval" if eval_mode else "inference",
        auth={"key_id": key_id},
        doc={
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
        },
        field={"name": field_name},
        question_len_chars=len((question or os.getenv("TILT_KIE_PROMPT") or "")),
        eval={"sample_id": sample_id} if eval_mode else None,
    )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        if PROMETHEUS_ENABLED:
            PIPELINE_ERRORS_TOTAL.labels("api", "UNSUPPORTED_CONTENT_TYPE").inc()
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    if not content:
        if PROMETHEUS_ENABLED:
            PIPELINE_ERRORS_TOTAL.labels("api", "EMPTY_UPLOAD").inc()
        raise HTTPException(status_code=400, detail="Empty file")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        if PROMETHEUS_ENABLED:
            PIPELINE_ERRORS_TOTAL.labels("api", "UPLOAD_TOO_LARGE").inc()
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_bytes} bytes > {max_bytes} bytes",
        )

    # Inference (TILT client)
    t0 = time.perf_counter()
    trace: Dict[str, Any] = {}
    raw_text: str = ""
    try:
        tilt_response = tilt.infer(
            content,
            content_type=file.content_type or None,
            question=question,
            field_name=field_name,
            request_id=request_id,
            max_candidates=max_candidates,
            max_neighbours=max_neighbours,
            trace=eval_mode,
        )
        raw_text = tilt_response["response"]
        if eval_mode:
            trace = tilt_response.get("trace") or {}
            trace["eval"] = {"sample_id": sample_id, "gt_raw": gt_raw}
    except Exception as e:  # noqa: BLE001
        if PROMETHEUS_ENABLED:
            PIPELINE_ERRORS_TOTAL.labels("tilt_client", "INFER_FAILED").inc()
        obs.log_event(
            "error",
            "api.tilt_infer_failed",
            request_id=request_id,
            mode="eval" if eval_mode else "inference",
            auth={"key_id": key_id},
            error={"type": type(e).__name__, "message": str(e)},
        )
        raise HTTPException(status_code=500, detail=f"TILT inference failed: {e}") from e
    finally:
        dt = time.perf_counter() - t0
        if PROMETHEUS_ENABLED:
            PIPELINE_STAGE_DURATION.labels("tilt_infer").observe(dt)

    # Postprocess rules (final data only)
    fields = raw_text
    rules_applied = False
    t_rules_ms = 0.0
    if RULES_ENABLED:
        t1 = time.perf_counter()
        try:
            fields = postprocess_rules(fields)
            rules_applied = True
        except Exception as e:  # noqa: BLE001
            if PROMETHEUS_ENABLED:
                PIPELINE_ERRORS_TOTAL.labels("postprocess", "RULES_FAILED").inc()
            obs.log_event(
                "warning",
                "api.postprocess_rules_failed",
                request_id=request_id,
                mode="eval" if eval_mode else "inference",
                auth={"key_id": key_id},
                error={"type": type(e).__name__, "message": str(e)},
            )
        finally:
            t_rules_ms = (time.perf_counter() - t1) * 1000.0
            if PROMETHEUS_ENABLED:
                PIPELINE_STAGE_DURATION.labels("postprocess_rules").observe(t_rules_ms / 1000.0)

    total_ms = (time.perf_counter() - t0) * 1000.0
    if eval_mode:
        timings = (trace.get("timings_ms") or {})
        timings.update({"postprocess_rules": round(t_rules_ms, 2), "total": round(total_ms, 2)})
        trace["timings_ms"] = timings

    obs.log_event(
        "info",
        "api.extract_done",
        request_id=request_id,
        mode="eval" if eval_mode else "inference",
        rules_applied=rules_applied,
    )

    response: Dict[str, Any] = {
        "data": fields,
        "meta": {
            "request_id": request_id,
            "client_key_id": key_id,
            "status": "ok",
            "model_version": TILT_MODEL,
            "ruleset_version": os.getenv("RULESET_VERSION", "rules-0.1.0"),
            "rules_enabled": RULES_ENABLED,
            "rules_applied": rules_applied,
            "source_file": file.filename,
        },
    }
    if eval_mode:
        response["trace"] = trace
    return response
