#!/usr/bin/env python3
"""
OCRlty eval-runner (CORD subset friendly)

Calls OCRlty API (/v1/extract). If you pass `sample_id` in form-data, the server switches to eval_mode
and returns `trace`. This runner always sends sample_id, so it always gets trace (intended for eval).

Dataset support:
- CORD subset ground truth file like: data/cord_subset/cord_gt.json
  Format: { "<sample_id>": { "id": "...", "image_file": "...", "<field_name>": ... , ... }, ... }

Output:
- One JSON artifact file with run metadata and per-sample full API responses.

Minimal usage (assuming repo root contains data/cord_subset/cord_gt.json):
  python eval_runner_updated.py --url http://HOST:8000 --api-key "$OCRLTY_API_KEY" \
    --field-name total_price --prompt-file prompt.txt

Override gt path:
  python eval_runner_updated.py ... --cord-gt data/cord_subset/cord_gt.json
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
    return "application/octet-stream"


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    raise SystemExit("ERROR: Provide --prompt or --prompt-file")


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


def _gt_to_string(v: Any) -> Optional[str]:
    """Convert gt value to a stable string for trace.eval.gt_raw."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    # For list/dict (e.g. menu), keep as JSON string so you can re-evaluate later.
    return json.dumps(v, ensure_ascii=False)


def _resolve_image_path(image_file: str, dataset_root: Optional[str]) -> Path:
    """
    image_file in cord_gt.json is often like: "data/cord_subset/images/cord_0003.jpg"
    If you run from repo root, Path(image_file) exists.
    If you run from another directory, you can set --dataset-root to help.
    """
    p = Path(image_file)
    if p.is_absolute():
        return p

    # Try as-is relative to CWD
    if p.exists():
        return p

    # If dataset_root provided, try relative to it
    if dataset_root:
        cand = Path(dataset_root) / p
        if cand.exists():
            return cand

    # Common case: strip leading "data/"
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "data":
        p2 = Path(*parts[1:])
        if p2.exists():
            return p2
        if dataset_root:
            cand2 = Path(dataset_root) / p2
            if cand2.exists():
                return cand2

    return p  # caller will raise FileNotFoundError with a clear message


def _load_cord_gt(cord_gt_path: Path) -> Dict[str, Any]:
    if not cord_gt_path.exists():
        raise FileNotFoundError(f"CORD GT file not found: {cord_gt_path}")
    obj = json.loads(cord_gt_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("cord_gt.json must be a JSON object: {sample_id: record, ...}")
    return obj


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    gt_raw: Optional[str]


def _build_samples_from_cord_gt(
    cord_gt: Dict[str, Any],
    field_name: str,
    dataset_root: Optional[str],
    limit: Optional[int],
) -> List[Sample]:
    out: List[Sample] = []
    for k in sorted(cord_gt.keys()):
        rec = cord_gt[k]
        if not isinstance(rec, dict):
            continue
        sid = str(rec.get("id") or k)
        img = rec.get("image_file")
        if not img:
            continue
        img_path = _resolve_image_path(str(img), dataset_root=dataset_root)
        gt = _gt_to_string(rec.get(field_name))
        out.append(Sample(sample_id=sid, image_path=img_path, gt_raw=gt))
        if limit is not None and len(out) >= limit:
            break
    return out


@dataclass
class RequestConfig:
    url: str
    api_key: str
    timeout_s: float
    retries: int
    retry_backoff_s: float
    max_candidates: Optional[int]
    max_neighbours: Optional[int]


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
            time.sleep(min(cfg.retry_backoff_s * (2 ** attempt), 30.0))
    raise RuntimeError(f"Request failed after retries: {last_exc}") from last_exc


def main() -> None:
    p = argparse.ArgumentParser(description="OCRlty eval-runner (CORD subset).")
    p.add_argument("--url", default=os.getenv("OCRLTY_URL", "http://127.0.0.1:8000"))
    p.add_argument("--api-key", default=os.getenv("OCRLTY_API_KEY"))
    p.add_argument("--field-name", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")

    # Dataset options (CORD GT by default)
    p.add_argument("--cord-gt", default=os.getenv("CORD_GT_PATH", "data/cord_subset/cord_gt.json"))
    p.add_argument("--dataset-root", default=os.getenv("DATASET_ROOT", None))

    p.add_argument("--out", default=None)
    p.add_argument("--timeout", type=float, default=float(os.getenv("OCRLTY_TIMEOUT_S", "180")))
    p.add_argument("--retries", type=int, default=int(os.getenv("OCRLTY_RETRIES", "2")))
    p.add_argument("--retry-backoff", type=float, default=float(os.getenv("OCRLTY_RETRY_BACKOFF_S", "1")))
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--max-neighbours", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-sha256", action="store_true")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: --api-key is required (or set OCRLTY_API_KEY)")

    question = _read_prompt(args)

    cord_gt_path = Path(args.cord_gt)
    cord_gt = _load_cord_gt(cord_gt_path)
    samples = _build_samples_from_cord_gt(cord_gt, args.field_name, args.dataset_root, args.limit)

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
            "cord_gt_path": str(cord_gt_path),
            "dataset_root": args.dataset_root,
            "client": {"http": "httpx" if _HAS_HTTPX else ("requests" if _HAS_REQUESTS else "none")},
        },
        "samples": [],
        "summary": {
            "total": len(samples), 
            "ok": 0, 
            "error": 0, 
            "http_status_counts": {}
        },
    }

    print(f"[eval-runner] url={args.url} field={args.field_name} samples={len(samples)} out={out_path}")
    t_run = time.perf_counter()

    for idx, s in enumerate(samples, start=1):
        record: Dict[str, Any] = {
            "sample_id": s.sample_id,
            "image_path": str(s.image_path),
            "gt_raw": s.gt_raw,
            "status": "error",
            "http_status": None,
            "sha256": None,
            "response": None,
            "error": None,
        }

        try:
            if not s.image_path.exists():
                raise FileNotFoundError(f"File not found: {s.image_path}")

            content_type = _guess_content_type(s.image_path)
            if content_type == "application/octet-stream":
                raise ValueError(f"Unsupported file extension: {s.image_path.suffix}")

            if not args.no_sha256:
                record["sha256"] = _sha256_file(s.image_path)

            status_code, resp = _post_extract(
                cfg=cfg,
                file_path=s.image_path,
                content_type=content_type,
                question=question,
                field_name=args.field_name,
                sample_id=s.sample_id,
                gt_raw=s.gt_raw,
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

        key = str(record["http_status"]) if record["http_status"] is not None else "exception"
        artifact["summary"]["http_status_counts"][key] = artifact["summary"]["http_status_counts"].get(key, 0) + 1

        artifact["samples"].append(record)
        _atomic_write_json(out_path, artifact)
        print(f"[{idx:>3}/{len(samples)}] {s.sample_id} -> {record['status']} (http={record['http_status']})")

    elapsed_s = time.perf_counter() - t_run
    artifact["run"]["finished_at_utc"] = _utc_now_iso()
    artifact["run"]["elapsed_s"] = round(elapsed_s, 3)
    _atomic_write_json(out_path, artifact)

    print(f"[eval-runner] done. ok={artifact['summary']['ok']} error={artifact['summary']['error']} elapsed={elapsed_s:.1f}s")
    print(f"[eval-runner] artifact: {out_path.resolve()}")


if __name__ == "__main__":
    main()
