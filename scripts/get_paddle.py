#!/usr/bin/env python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont

# PaddleX 3.x pipeline
try:
    from paddlex import create_pipeline  # type: ignore
except Exception:
    create_pipeline = None  # type: ignore

# numpy опционально (как в tilt_client)
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


# =========================
# БАЗОВЫЕ ПУТИ
# =========================
# scripts/get_paddle.py -> project_root = parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================
# НАСТРОЙКИ
# =========================

EVAL_PATH = PROJECT_ROOT / "data" / "cord_subset" / "coord_eval_totals.json"
OUT_DIR = PROJECT_ROOT / "out" / "paddle_debug"
ENRICHED_JSON_NAME = "eval_with_paddle.json"

MIN_ERROR = 0.0
MAX_SAMPLES = 0  # 0 = без лимита

# OCR
MIN_CONFIDENCE = 0.0  # выставь 0.3 если хочешь как в tilt_client по умолчанию

# Визуализация
SAVE_OVERLAY = True
SAVE_WHITE_OVERLAY = False
SAVE_TILT_LIKE = True

# Если хочешь прогонять только проблемные кейсы:
ONLY_MISMATCHES = True

# =========================


def load_eval(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of records in {path}, got {type(data)}")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def select_mismatches(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    for rec in records:
        gt = rec.get("gt_value")
        if gt is None:
            continue

        status = rec.get("status")
        pred = rec.get("pred_value")
        abs_err = rec.get("abs_error")

        if status != "ok":
            rec["_reason"] = str(status)
            mismatches.append(rec)
            continue

        if pred is None:
            rec["_reason"] = "no_number"
            mismatches.append(rec)
            continue

        if isinstance(abs_err, (int, float)):
            if abs_err > MIN_ERROR:
                rec["_reason"] = "abs_error"
                mismatches.append(rec)
        else:
            rec["_reason"] = "no_abs_error"
            mismatches.append(rec)

    def sort_key(r: Dict[str, Any]) -> float:
        err = r.get("abs_error")
        if isinstance(err, (int, float)):
            return float(err)
        return -1.0

    mismatches.sort(key=sort_key, reverse=True)

    if MAX_SAMPLES > 0:
        mismatches = mismatches[:MAX_SAMPLES]

    return mismatches


def safe_read_image(img_path: Path) -> Optional[Any]:
    if not img_path.exists():
        print(f"Файл не найден: {img_path}")
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Не удалось прочитать изображение: {img_path}")
    return img


def resolve_image_path(raw: Optional[str]) -> Optional[Path]:
    """
    Нормализуем image_path из JSON так, чтобы:
    - убрать /workspace/src
    - убрать ведущие / или \ (иначе на Windows станет абсолютным D:\data\...)
    - собрать относительно PROJECT_ROOT
    """
    if not raw:
        return None

    raw = raw.replace("/workspace/src/", "").replace("/workspace/src", "")
    raw = raw.lstrip("/\\")
    p = Path(raw)

    # если внезапно уже абсолютный Windows-путь - используем его
    if p.is_absolute():
        return p.resolve()

    return (PROJECT_ROOT / p).resolve()


# -------------------------
# PaddleX OCR как в tilt_client
# -------------------------

def _first_nonempty(res: Dict[str, Any], keys: Tuple[str, ...]):
    """Аккуратно берём первое непустое значение без bool(np.array)."""
    for key in keys:
        if key not in res:
            continue
        val = res[key]
        if val is None:
            continue
        try:
            if np is not None and isinstance(val, np.ndarray):
                if val.size == 0:
                    continue
            elif isinstance(val, (list, tuple, str)):
                if len(val) == 0:
                    continue
        except Exception:
            pass
        return val
    return None


def run_paddlex_ocr_on_image(
    ocr_pipeline,
    img_bgr,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Повторяет идею tilt_client._run_ocr:
    - преобразуем в PIL
    - сохраняем временный PNG
    - ocr_pipeline.predict(path)
    - достаем boxes/texts/scores
    Возвращаем данные в виде:
      {
        "texts": [...],
        "scores": [...],
        "boxes": [ [[x,y]*4], ... ],  # полигоны для overlay
        "words": [ {"text":..., "bbox":[x1,y1,x2,y2], "score":...}, ... ]  # как у tilt_client
      }
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    w, h = pil.size

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pil.save(tmp_path, format="PNG")
        raw_out = list(ocr_pipeline.predict(tmp_path))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    texts: List[str] = []
    scores: List[float] = []
    polys: List[List[List[float]]] = []
    words: List[Dict[str, Any]] = []

    if not raw_out:
        return {"texts": texts, "scores": scores, "boxes": polys, "words": words, "width": w, "height": h}

    page0 = raw_out[0]

    # В разных версиях PaddleX:
    #   - page0.res — объект
    #   - {"res": .} — dict
    #   - объект результата без поля res
    if hasattr(page0, "res"):
        res = page0.res
    elif isinstance(page0, dict) and "res" in page0:
        res = page0["res"]
    else:
        res = page0

    if hasattr(res, "__dict__") and not isinstance(res, dict):
        res = res.__dict__

    if not isinstance(res, dict):
        return {"texts": texts, "scores": scores, "boxes": polys, "words": words, "width": w, "height": h}

    raw_boxes = _first_nonempty(res, ("rec_boxes", "dt_polys", "det_boxes", "boxes"))
    raw_texts = _first_nonempty(res, ("rec_texts", "rec_text", "texts"))
    raw_scores = _first_nonempty(res, ("rec_scores", "rec_score", "scores"))

    if raw_boxes is None or raw_texts is None:
        return {"texts": texts, "scores": scores, "boxes": polys, "words": words, "width": w, "height": h}

    # приводим боксы к numpy для удобства
    if np is not None:
        try:
            boxes_arr = np.array(raw_boxes)
        except Exception:
            boxes_arr = raw_boxes
    else:
        boxes_arr = raw_boxes

    # нормализатор: box -> bbox [x1,y1,x2,y2] и poly4
    def normalize_box(box) -> Optional[Tuple[List[float], List[List[float]]]]:
        try:
            # numpy-путь
            if np is not None and isinstance(box, np.ndarray):
                pts = box.reshape(-1, 2)
                x1 = float(pts[:, 0].min())
                y1 = float(pts[:, 1].min())
                x2 = float(pts[:, 0].max())
                y2 = float(pts[:, 1].max())
                bbox = [x1, y1, x2, y2]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                return bbox, poly

            # если это [x1,y1,x2,y2]
            if (
                isinstance(box, (list, tuple))
                and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)
            ):
                x1, y1, x2, y2 = map(float, box)
                bbox = [x1, y1, x2, y2]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                return bbox, poly

            # иначе ожидаем список точек
            pts = list(box)
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bbox = [x1, y1, x2, y2]

            # если есть ровно 4 точки - используем их как poly
            poly_pts: List[List[float]] = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    poly_pts.append([float(p[0]), float(p[1])])

            if len(poly_pts) == 4:
                poly = poly_pts
            else:
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            return bbox, poly
        except Exception:
            return None

    for idx, (box, text) in enumerate(zip(boxes_arr, raw_texts)):
        if not text:
            continue

        score = 1.0
        if raw_scores is not None and idx < len(raw_scores):
            try:
                score = float(raw_scores[idx])
            except Exception:
                score = 1.0

        if score < min_confidence:
            continue

        norm = normalize_box(box)
        if norm is None:
            continue
        bbox, poly = norm

        words.append({"text": str(text), "bbox": bbox, "score": float(score)})
        texts.append(str(text))
        scores.append(float(score))
        polys.append(poly)

    return {
        "texts": texts,
        "scores": scores,
        "boxes": polys,
        "words": words,
        "width": int(w),
        "height": int(h),
    }


# -------------------------
# Визуализация
# -------------------------

def draw_overlay_pil(img_bgr, paddle_data: Dict[str, Any]) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    boxes = paddle_data.get("boxes") or []
    texts = paddle_data.get("texts") or []
    scores = paddle_data.get("scores") or []

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue

        xy = [(box[j][0], box[j][1]) for j in range(4)]
        draw.line(xy + [xy[0]], width=2)

        label = ""
        if i < len(texts):
            label += texts[i]
        if i < len(scores):
            label += f" ({scores[i]:.2f})"

        if label:
            x0, y0 = xy[0]
            draw.text((x0 + 2, y0 + 2), label, font=font)

    return pil


def draw_overlay_on_white(img_bgr, paddle_data: Dict[str, Any]) -> Image.Image:
    h, w = img_bgr.shape[:2]
    pil = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(pil)

    boxes = paddle_data.get("boxes") or []
    texts = paddle_data.get("texts") or []
    scores = paddle_data.get("scores") or []

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue

        xy = [(box[j][0], box[j][1]) for j in range(4)]
        draw.line(xy + [xy[0]], width=2)

        label = ""
        if i < len(texts):
            label += texts[i]
        if i < len(scores):
            label += f" ({scores[i]:.2f})"

        if label:
            x0, y0 = xy[0]
            draw.text((x0 + 2, y0 + 2), label, font=font)

    return pil


def make_tilt_like_image(img_bgr, paddle_data: Dict[str, Any]) -> Image.Image:
    """
    Грубая визуальная гипотеза: берём область, покрывающую все детекты.
    Если боксов нет — возвращаем оригинал.
    """
    boxes = paddle_data.get("boxes") or []
    h, w = img_bgr.shape[:2]

    if not boxes:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    xs: List[float] = []
    ys: List[float] = []
    for box in boxes:
        for x, y in box:
            xs.append(float(x))
            ys.append(float(y))

    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(w, int(max(xs)))
    y2 = min(h, int(max(ys)))

    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


# -------------------------
# Main
# -------------------------

def main() -> None:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"EVAL_PATH: {EVAL_PATH}")

    if create_pipeline is None:
        raise RuntimeError(
            "paddlex is not installed. Install PaddleX 3.x to reproduce tilt_client OCR."
        )

    records = load_eval(EVAL_PATH)

    mismatches = select_mismatches(records)
    mismatch_ids = {id(rec): True for rec in mismatches}

    mismatch_id_values = set()
    for rec in mismatches:
        if rec.get("id") is not None:
            mismatch_id_values.add(str(rec.get("id")))

    # Инициализируем PaddleX OCR pipeline
    ocr = create_pipeline(pipeline="OCR")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_out = OUT_DIR / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for rec in records:
        if ONLY_MISMATCHES:
            if mismatch_id_values:
                if str(rec.get("id")) not in mismatch_id_values:
                    continue
            else:
                if id(rec) not in mismatch_ids:
                    continue

        img_path = resolve_image_path(rec.get("image_path"))
        if not img_path:
            skipped += 1
            continue

        print(f"img_path: {img_path}")

        img_bgr = safe_read_image(img_path)
        if img_bgr is None:
            skipped += 1
            continue

        paddle_data = run_paddlex_ocr_on_image(
            ocr,
            img_bgr,
            min_confidence=MIN_CONFIDENCE,
        )
        rec["paddle"] = paddle_data

        base_name = img_path.stem
        rec_id = rec.get("id")
        prefix = f"{str(rec_id)}_" if rec_id is not None else ""
        safe_name = f"{prefix}{base_name}"

        if SAVE_OVERLAY:
            overlay = draw_overlay_pil(img_bgr, paddle_data)
            overlay.save(img_out / f"{safe_name}__paddlex_overlay.png")

        if SAVE_WHITE_OVERLAY:
            white = draw_overlay_on_white(img_bgr, paddle_data)
            white.save(img_out / f"{safe_name}__white_overlay.png")

        if SAVE_TILT_LIKE:
            tilt_like = make_tilt_like_image(img_bgr, paddle_data)
            tilt_like.save(img_out / f"{safe_name}__tilt_like.png")

        processed += 1

    enriched_path = OUT_DIR / ENRICHED_JSON_NAME
    save_json(enriched_path, records)

    print("Готово.")
    print(f"Обработано записей: {processed}")
    print(f"Пропущено (нет image_path/файл не найден/не прочитан): {skipped}")
    print(f"Enriched JSON: {enriched_path}")
    print(f"Картинки: {img_out}")


if __name__ == "__main__":
    main()
