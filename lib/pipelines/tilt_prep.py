# lib/pipelines/tilt_prep.py
from __future__ import annotations
from typing import Any, Dict, List
from PIL import Image
import httpx

def build_ocr_document(img_path: str, pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует результат PaddleOCR 3.x (predict) в OCR Document для Arctic-TILT.
    Ожидает ключи: 'rec_texts', 'rec_polys'. Поля в пикселях.
    """
    with Image.open(img_path) as im:
        w, h = im.size

    texts = pred.get("rec_texts", []) or []
    polys = pred.get("rec_polys", []) or []
    spans: List[Dict[str, Any]] = []

    for i, txt in enumerate(texts):
        poly = polys[i] if i < len(polys) else None
        if not txt or not poly:
            continue
        # poly: [[x,y], [x,y], [x,y], [x,y]] → AABB
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        spans.append({"bbox": bbox, "text": str(txt)})

    return {"pages": [{"width": float(w), "height": float(h), "spans": spans}]}

def make_messages_ocr_doc(ocr_doc: Dict[str, Any], system_prompt: str | None = None) -> List[Dict[str, Any]]:
    sys_content = [{"type": "text", "text": system_prompt}] if system_prompt else []
    msgs: List[Dict[str, Any]] = []
    if sys_content:
        msgs.append({"role": "system", "content": sys_content})
    msgs.append({"role": "user", "content": [{"type": "input_ocr_document", "document": ocr_doc}]})
    return msgs

def call_tilt_openai_chat(vllm_base_url: str, model: str, messages: List[Dict[str, Any]], api_key: str = "", timeout: float = 30.0) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout) as cli:
        r = cli.post(f"{vllm_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
