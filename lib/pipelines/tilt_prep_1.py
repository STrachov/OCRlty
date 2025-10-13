# lib/pipelines/tilt_prep.py
from __future__ import annotations
import json
import httpx
from PIL import Image
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


def _to_lines_from_paddle(result: Any) -> List[dict]:
    """Нормализуем ответ PaddleOCR 3.x список {'text','bbox'}."""
    lines: List[dict] = []
    # 3.x: dict с rec_texts/rec_polys
    #dict_keys = ['input_path', 'page_index', 'doc_preprocessor_res', 'dt_polys', 'model_settings', 'text_det_params', 'text_type', 'text_rec_score_thresh', 'return_word_box', 'rec_texts', 'rec_scores', 'rec_polys', 'vis_fonts', 'textline_orientation_angles', 'rec_boxes']
    texts = result.get("rec_texts") or []
    boxes = result.get("rec_polys") or result.get("dt_polys") or [None]*len(texts)
    for i, t in enumerate(texts):
        b = boxes[i] if i < len(boxes) else None
        if b is not None and len(b) >= 4:  # poly -> AABB
            xs = [p[0] for p in b]; ys = [p[1] for p in b]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            bbox = None
        lines.append({"text": str(t), "bbox": bbox})
    return lines

def build_ocr_page(img_path: str, paddle_result: Any) -> Dict[str, Any]:
    with Image.open(img_path) as im:
        w, h = im.size
    spans = []
    lines = _to_lines_from_paddle(paddle_result)
    print('++++lines: ', lines)
    for ln in lines:
        if not ln.get("bbox") or not ln.get("text"):
            continue
        x1, y1, x2, y2 = ln["bbox"]
        spans.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "text": ln["text"]})
    return {"width": float(w), "height": float(h), "spans": spans}

def build_ocr_document(
    img_path_list: List[str], paddle_result_arr: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    
    if len(img_path_list) != len(paddle_result_arr):
        logger.warning("img_path_list and paddle_result_arr must be the same length")
    
    return [
        build_ocr_page(img_path, paddle_page) 
        for img_path, paddle_page in zip(img_path_list, paddle_result_arr)
    ]


def make_messages(ocr_doc: Dict[str, Any], system_prompt: str | None = None) -> List[Dict[str, Any]]:
    sysc = [{"type": "text", "text": system_prompt}] if system_prompt else []
    return [
        {"role": "system", "content": sysc} if sysc else {"role": "system", "content": []},
        {"role": "user", "content": [{"type": "input_ocr_document", "document": ocr_doc}]},
    ]

def call_tilt(vllm_base_url: str, model: str, messages: List[Dict[str, Any]], api_key: str = "") -> str:
    #messages=json.dumps(messages)
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=30.0) as cli:
        r = cli.post(f"{vllm_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        if r.status_code >= 400:
            print("mock-vllm error:", r.status_code, r.text)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
