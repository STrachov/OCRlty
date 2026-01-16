#!/usr/bin/env python3
"""
OCRlty eval-runner

Runs an evaluation loop against the OCRlty API (/v1/extract) in "eval mode" by sending `sample_id`
(and optional `gt_raw`) so the server returns `trace`.

This script writes ONE artifact JSON file that contains:
- run metadata (prompt, field_name, endpoint, params)
- per-sample request context (sample_id, image_path, gt_raw)
- the full API response (data/meta/trace) or an error record

Recommended input format: a manifest (JSONL or CSV) with columns/keys:
  sample_id, image_path, gt_raw

Examples:

1) JSONL manifest (one object per line):
{"sample_id":"cord_train_0000","image_path":".../cord_0000.jpg","gt_raw":"1591600"}
{"sample_id":"cord_train_0001","image_path":".../cord_0001.jpg","gt_raw":"33.00"}

2) CSV manifest (header required):
sample_id,image_path,gt_raw
cord_train_0000,/path/cord_0000.jpg,1591600

Usage:
  python eval_runner.py --field-name total_price --prompt-file prompt.txt \\
    --manifest manifest.jsonl --api-key "$OCRLTY_API_KEY" --url http://HOST:8000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Prefer httpx if available (better timeouts); fallback to requests.
try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except Exception:
    httpx = None  # type: ignore
    _HAS_HTTPX = False

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".pdf":
        return "application/pdf"
    # Your API validates content types; extend if needed.
    return "application/octet-stream"


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    raise SystemExit("ERROR: Provide --prompt or --prompt-file")


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: List[Dict[str, Any]] = []
    suf = manifest_path.suffix.lower()

    if suf in (".jsonl", ".jl"):
        for lineno, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {manifest_path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {lineno} must be an object/dict")
            samples.append(obj)
        return samples

    if suf == ".json":
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
            return obj["samples"]
        raise ValueError("JSON manifest must be a list or an object with a 'samples' list")

    if suf == ".csv":
        import csv
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV manifest must have a header row")
            for row in reader:
                samples.append(dict(row))
        return samples

    raise ValueError("Manifest must be .jsonl/.json/.csv")


def _normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Accept a few common key variants.
    sid = sample.get("sample_id") or sample.get("id") or sample.get("name")
    img = sample.get("image_path") or sample.get("path") or sample.get("file") or sample.get("image")
    gt = sample.get("gt_raw") or sample.get("gt") or sample.get("label") or sample.get("gt_value_raw")

    if img is None:
        raise ValueError(f"Sample is missing image_path: {sample}")

    img_path = Path(str(img))
    if sid is None or str(sid).strip() == "":
        sid = img_path.stem

    out = {
        "sample_id": str(sid),
        "image_path": str(img_path),
        "gt_raw": None if gt is None else str(gt),
    }
    # Preserve any other fields (optional)
    for k, v in sample.items():
        if k in out:
            continue
        out[k] = v
    return out


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass
class RequestConfig:
    url: str
    api_key: str
    timeout_s: float
    retries: int
    retry_backoff_s: float
    max_candidates: Optional[int]
    max_neighbours: Optional[int]
    candidates_mode: str


def _post_extract_httpx(
    cfg: RequestConfig,
    file_path: Path,
    content_type: str,
    question: str,
    field_name: str,
    sample_id: str,
    gt_raw: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    assert httpx is not None
    headers = {"X-API-Key": cfg.api_key}
    data: Dict[str, str] = {
        "question": question,
        "field_name": field_name,
        "sample_id": sample_id,  # triggers eval_mode on server
    }
    if gt_raw is not None:
        data["gt_raw"] = gt_raw
    if cfg.max_candidates is not None:
        data["max_candidates"] = str(cfg.max_candidates)
    if cfg.max_neighbours is not None:
        data["max_neighbours"] = str(cfg.max_neighbours)
    if cfg.candidates_mode and cfg.candidates_mode != "off":
        data["candidates_mode"] = cfg.candidates_mode

    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, content_type)}
        with httpx.Client(timeout=cfg.timeout_s) as client:
            r = client.post(cfg.url.rstrip("/") + "/v1/extract", headers=headers, data=data, files=files)
            return r.status_code, r.json()


def _post_extract_requests(
    cfg: RequestConfig,
    file_path: Path,
    content_type: str,
    question: str,
    field_name: str,
    sample_id: str,
    gt_raw: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    assert requests is not None
    headers = {"X-API-Key": cfg.api_key}
    data: Dict[str, str] = {
        "question": question,
        "field_name": field_name,
        "sample_id": sample_id,
    }
    if gt_raw is not None:
        data["gt_raw"] = gt_raw
    if cfg.max_candidates is not None:
        data["max_candidates"] = str(cfg.max_candidates)
    if cfg.max_neighbours is not None:
        data["max_neighbours"] = str(cfg.max_neighbours)

    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, content_type)}
        r = requests.post(
            cfg.url.rstrip("/") + "/v1/extract",
            headers=headers,
            data=data,
            files=files,
            timeout=cfg.timeout_s,
        )
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_non_json_body": r.text}


def _post_extract(
    cfg: RequestConfig,
    file_path: Path,
    content_type: str,
    question: str,
    field_name: str,
    sample_id: str,
    gt_raw: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.retries + 1):
        try:
            if _HAS_HTTPX:
                return _post_extract_httpx(cfg, file_path, content_type, question, field_name, sample_id, gt_raw)
            if _HAS_REQUESTS:
                return _post_extract_requests(cfg, file_path, content_type, question, field_name, sample_id, gt_raw)
            raise RuntimeError("Neither httpx nor requests is installed")
        except Exception as e:
            last_exc = e
            sleep_s = min(cfg.retry_backoff_s * (2 ** attempt), 30.0)
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after retries: {last_exc}") from last_exc


def main() -> None:
    p = argparse.ArgumentParser(description="OCRlty eval-runner (calls /v1/extract in eval_mode).")
    p.add_argument(
        "--url",
        default=os.getenv("OCRLTY_URL", "http://127.0.0.1:8000"),
        help="Base URL of the API, without /v1/extract (default: %(default)s)",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OCRLTY_API_KEY"),
        help="API key for X-API-Key header (or set OCRLTY_API_KEY)",
    )
    p.add_argument("--field-name", required=True, help="Field name (e.g., total_price)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="Prompt text")
    g.add_argument("--prompt-file", help="Path to a UTF-8 text file with the prompt")
    p.add_argument("--manifest", required=True, help="Path to manifest (.jsonl/.json/.csv) with samples")
    p.add_argument("--out", default=None, help="Output artifact JSON path (default: auto in ./eval_runs/)")
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OCRLTY_TIMEOUT_S", "180")),
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("OCRLTY_RETRIES", "2")),
        help="Retries for transient errors/exceptions (default: %(default)s)",
    )
    p.add_argument(
        "--retry-backoff",
        type=float,
        default=float(os.getenv("OCRLTY_RETRY_BACKOFF_S", "1")),
        help="Backoff base seconds for retries (default: %(default)s)",
    )
    p.add_argument("--max-candidates", type=int, default=None, help="Override max_candidates (optional)")
    p.add_argument("--max-neighbours", type=int, default=None, help="Override max_neighbours (optional)")
    p.add_argument("--candidates-mode", choices=["off", "on"], default=os.getenv("OCRLTY_CANDIDATES_MODE", "off"), help="Candidates mode passed to API (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples (for quick tests)")
    p.add_argument("--no-sha256", action="store_true", help="Skip file sha256 computation (faster)")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: --api-key is required (or set OCRLTY_API_KEY)")

    question = _read_prompt(args)
    manifest_path = Path(args.manifest)
    raw_samples = _load_manifest(manifest_path)
    samples = [_normalize_sample(s) for s in raw_samples]
    if args.limit is not None:
        samples = samples[: args.limit]

    started_at = _utc_now_iso()
    run_id = f"{started_at.replace(':','').replace('-','')}_{args.field_name}"
    out_path = Path(args.out) if args.out else Path("eval_runs") / f"run_{run_id}.json"

    cfg = RequestConfig(
        url=args.url,
        api_key=args.api_key,
        timeout_s=args.timeout,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff,
        max_candidates=args.max_candidates,
        max_neighbours=args.max_neighbours,
        candidates_mode=args.candidates_mode,
    )

    artifact: Dict[str, Any] = {
        "schema_version": "ocrlty.evalrun.v1",
        "run": {
            "run_id": run_id,
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "elapsed_s": None,
            "url": args.url,
            "endpoint": "/v1/extract",
            "field_name": args.field_name,
            "question_base": question,
            "max_candidates": args.max_candidates,
            "max_neighbours": args.max_neighbours,
            "candidates_mode": args.candidates_mode,
            "manifest": str(manifest_path),
            "client": {"http": "httpx" if _HAS_HTTPX else ("requests" if _HAS_REQUESTS else "none")},
        },
        "samples": [],
        "summary": {
            "total": len(samples),
            "ok": 0,
            "error": 0,
            "http_status_counts": {},
        },
    }

    print(f"[eval-runner] url={args.url} field={args.field_name} samples={len(samples)} out={out_path}")
    t_run = time.perf_counter()

    for idx, s in enumerate(samples, start=1):
        sample_id = s["sample_id"]
        img_path = Path(s["image_path"])
        gt_raw = s.get("gt_raw")

        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": str(img_path),
            "gt_raw": gt_raw,
            "status": "error",
            "http_status": None,
            "sha256": None,
            "response": None,
            "error": None,
        }

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"File not found: {img_path}")

            content_type = _guess_content_type(img_path)
            if content_type == "application/octet-stream":
                raise ValueError(f"Unsupported file extension for content-type guessing: {img_path.suffix}")

            if not args.no_sha256:
                record["sha256"] = _sha256_file(img_path)

            status_code, resp = _post_extract(
                cfg=cfg,
                file_path=img_path,
                content_type=content_type,
                question=question,
                field_name=args.field_name,
                sample_id=sample_id,
                gt_raw=gt_raw,
            )

            record["http_status"] = status_code
            record["response"] = resp

            if 200 <= status_code < 300:
                record["status"] = "ok"
                artifact["summary"]["ok"] += 1
            else:
                artifact["summary"]["error"] += 1
                msg = None
                if isinstance(resp, dict):
                    msg = resp.get("detail") or resp.get("message")
                record["error"] = {"type": "HTTPError", "message": msg or "Non-2xx response"}

        except Exception as e:
            artifact["summary"]["error"] += 1
            record["error"] = {"type": type(e).__name__, "message": str(e)}

        sc = record["http_status"]
        key = str(sc) if sc is not None else "exception"
        artifact["summary"]["http_status_counts"][key] = artifact["summary"]["http_status_counts"].get(key, 0) + 1

        artifact["samples"].append(record)
        _atomic_write_json(out_path, artifact)
        print(f"[{idx:>3}/{len(samples)}] {sample_id} -> {record['status']} (http={record['http_status']})")

    elapsed_s = time.perf_counter() - t_run
    artifact["run"]["finished_at_utc"] = _utc_now_iso()
    artifact["run"]["elapsed_s"] = round(elapsed_s, 3)
    _atomic_write_json(out_path, artifact)

    print(f"[eval-runner] done. ok={artifact['summary']['ok']} error={artifact['summary']['error']} elapsed={elapsed_s:.1f}s")
    print(f"[eval-runner] artifact: {out_path.resolve()}")


if __name__ == "__main__":
    main()
