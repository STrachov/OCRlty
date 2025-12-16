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
DETECT_AND_ROTATE = False

# Визуализация
SAVE_OVERLAY = False          # оверлей поверх оригинала
SAVE_WHITE_OVERLAY = False    # только белый фон
SAVE_TILT_LIKE = False        # «кроп» по области детектов
SAVE_SIDE_BY_SIDE = True      # слева чек, справа белый фон с текстами

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
# PaddleX OCR как в tilt_client + фиксы ориентации
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


def compute_bbox_from_points(box) -> Optional[Tuple[float, float, float, float]]:
    """
    box: список/массив точек [[x,y], ...]
    -> (x1, y1, x2, y2)
    """
    try:
        pts = list(box)
        if not pts:
            return None

        xs: List[float] = []
        ys: List[float] = []

        for p in pts:
            if not hasattr(p, "__len__") or len(p) < 2:
                continue
            xs.append(float(p[0]))
            ys.append(float(p[1]))

        if not xs or not ys:
            return None

        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None


def detect_vertical_page(
    raw_boxes,
    raw_texts,
    min_chars: int = 4,
    aspect_thresh: float = 2.2,
    min_samples: int = 5,
    fraction_thresh: float = 0.6,
) -> bool:
    """
    Возвращает True, если страница выглядит "вертикально сломанной":
    у большинства длинных слов сильно вытянутые по вертикали боксы (h >> w).
    """

    ratios: List[float] = []

    for box, text in zip(raw_boxes, raw_texts):
        if not text:
            continue

        clean = str(text).replace(" ", "")
        if len(clean) < min_chars:
            # по коротким токенам не судим об ориентации
            continue

        bbox = compute_bbox_from_points(box)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        ratios.append(h / w)

    if len(ratios) < min_samples:
        # мало данных – ничего не трогаем
        return False

    ratios.sort()
    median = ratios[len(ratios) // 2]
    frac = sum(r > aspect_thresh for r in ratios) / len(ratios)

    return median > aspect_thresh and frac > fraction_thresh


def fix_box_on_vertical_page(
    bbox: List[float],
    img_w: int,
    img_h: int,
    shallow_factor: float = 1.1,
) -> List[float]:
    """
    Для страницы, признанной вертикально сломанной:
    локально "переворачиваем" бокс: меняем ширину и высоту вокруг центра,
    если он заметно выше, чем шире.

    bbox: [x1, y1, x2, y2] в координатах картинки.
    """

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return bbox

    # если бокс уже почти горизонтальный – не трогаем
    if h <= w * shallow_factor:
        return bbox

    # центр
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # "переворачиваем": новая ширина = старая высота, новая высота = старая ширина
    new_w = h
    new_h = w

    x1 = cx - new_w / 2.0
    x2 = cx + new_w / 2.0
    y1 = cy - new_h / 2.0
    y2 = cy + new_h / 2.0

    # клип в пределах изображения
    x1 = max(0.0, min(float(img_w - 1), x1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y1 = max(0.0, min(float(img_h - 1), y1))
    y2 = max(0.0, min(float(img_h - 1), y2))

    return [float(x1), float(y1), float(x2), float(y2)]

# New funcctional
def compute_bbox_from_points(box) -> Optional[Tuple[float, float, float, float]]:
    """
    box: список/массив точек [[x,y], ...]
    -> (x1, y1, x2, y2)`
    """
    try:
        pts = list(box)
        if not pts:
            return None

        xs: List[float] = []
        ys: List[float] = []

        for p in pts:
            if not hasattr(p, "__len__") or len(p) < 2:
                continue
            xs.append(float(p[0]))
            ys.append(float(p[1]))

        if not xs or not ys:
            return None

        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None


def transform_box_points(box, img_w: int, img_h: int, orient: str) -> List[List[float]]:
    """
    Преобразует точки бокса из координат Paddle в координаты оригинального изображения
    с учётом ориентации:

      orient = "none"  — ничего не делаем
      orient = "cw"    — Paddle работал с изображением, повернутым
                         на 90° по часовой; разворачиваем обратно
      orient = "ccw"   — Paddle работал с изображением, повернутым
                         против часовой

    Все формулы — для преобразования (x_paddle, y_paddle) -> (x_orig, y_orig).
    """

    pts = list(box)
    out: List[List[float]] = []

    for p in pts:
        if not hasattr(p, "__len__") or len(p) < 2:
            continue
        x = float(p[0])
        y = float(p[1])

        if orient == "none":
            x2, y2 = x, y
        elif orient == "cw":
            # inverse( original -> rotated_cw ):
            # original(xo,yo) -> (xr = H-1-yo, yr = xo)
            # => xo = yr, yo = H-1-xr
            x2 = y
            y2 = img_h - 1.0 - x
        elif orient == "ccw":
            # inverse( original -> rotated_ccw ):
            # original(xo,yo) -> (xr = yo, yr = W-1-xo)
            # => xo = W-1-yr, yo = xr
            x2 = img_w - 1.0 - y
            y2 = x
        else:
            x2, y2 = x, y

        # немного страхуемся от выхода за границы
        x2 = max(0.0, min(float(img_w - 1), x2))
        y2 = max(0.0, min(float(img_h - 1), y2))

        out.append([x2, y2])

    return out

def orientation_score(
    raw_boxes,
    raw_texts,
    img_w: int,
    img_h: int,
    orient: str,
    min_chars: int = 3,
    min_samples: int = 5,
) -> float:
    """
    Оцениваем, насколько страница в данной ориентации похожа на нормальный чек
    с горизонтальными строками.

    Чем больше score, тем лучше эта ориентация.
    """
    aspects: List[float] = []

    for box, text in zip(raw_boxes, raw_texts):
        if not text:
            continue
        clean = str(text).replace(" ", "")
        if len(clean) < min_chars:
            continue

        # точки в выбранной ориентации
        pts = transform_box_points(box, img_w, img_h, orient)
        bbox = compute_bbox_from_points(pts)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        # фильтр по центру: выбрасываем боксы у самого края (логотипы, вертикальные бордеры)
        cx = (x1 + x2) / 2.0
        if cx < img_w * 0.1 or cx > img_w * 0.9:
            continue

        aspects.append(w / h)

    if len(aspects) < min_samples:
        return -1e9  # мало данных, ориентация непонятна

    aspects.sort()
    median_ar = aspects[len(aspects) // 2]

    n = len(aspects)
    frac_horiz = sum(a >= 1.3 for a in aspects) / n
    frac_vert = sum(a <= 0.75 for a in aspects) / n

    # комбинируем "насколько строки широкие" и "сколько их по сравнению с вертикальными"
    score = median_ar + 0.5 * (frac_horiz - frac_vert)
    return score


def choose_best_orientation(raw_boxes, raw_texts, img_w: int, img_h: int) -> str:
    """
    Пробуем три варианта: без поворота, cw, ccw.
    Выбираем тот, где медиана w/h максимальна.
    Но поворачиваем только если выигрыш по сравнению с 'none' значимый.
    """
    candidates = ["none", "cw", "ccw"]
    scores: Dict[str, float] = {}

    for o in candidates:
        scores[o] = orientation_score(raw_boxes, raw_texts, img_w, img_h, o)

    best_orient = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_orient]
    base_score = scores.get("none", -1e9)

    # Если "none" и так неплохой, а выигрыш небольшой — не вращаем
    if best_orient != "none" and best_score > 1.3 and best_score - base_score > 0.5:
        return best_orient
    return "none"

def run_paddlex_ocr_on_image(
    ocr_pipeline,
    img_bgr,
    min_confidence: float = 0.0,
    img_path: Path|None = None
) -> Dict[str, Any]:
    """
    Запуск PaddleX OCR на одном изображении с определением ориентации
    (none / 90° cw / 90° ccw) на уровне всей страницы и последующим
    преобразованием всех боксов в координаты оригинального изображения.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    img_w, img_h = pil.size

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pil.save(tmp_path, format="PNG")
        raw_out = list(ocr_pipeline.predict(
            tmp_path, 
            use_doc_orientation_classify=DETECT_AND_ROTATE,
            ))
        first_dict = raw_out[0]
        if 'doc_preprocessor_res' in first_dict and 'angle' in first_dict['doc_preprocessor_res']:
            print(f"angle: {first_dict['doc_preprocessor_res']['angle']}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    
    
    # def json_default(o):
    #     if isinstance(o, np.ndarray):
    #         return o.tolist()
    #     if isinstance(o, (np.integer, np.floating)):
    #         return o.item()
    #     return str(o)  # на крайний случай

    # if img_path is not None and isinstance(img_path, Path):
    #     json_path = img_path.with_suffix(".json")
    #     with open(json_path, "w", encoding="utf-8") as f:
    #         json.dump(list(raw_out), f, ensure_ascii=False, indent=2, default=json_default)


    
    

    texts: List[str] = []
    scores: List[float] = []
    polys: List[List[List[float]]] = []
    words: List[Dict[str, Any]] = []

    if not raw_out:
        return {
            "texts": texts,
            "scores": scores,
            "boxes": polys,
            "words": words,
            "width": img_w,
            "height": img_h,
        }

    page0 = raw_out[0]

    if hasattr(page0, "res"):
        res = page0.res
    elif isinstance(page0, dict) and "res" in page0:
        res = page0["res"]
    else:
        res = page0

    if hasattr(res, "__dict__") and not isinstance(res, dict):
        res = res.__dict__

    if not isinstance(res, dict):
        return {
            "texts": texts,
            "scores": scores,
            "boxes": polys,
            "words": words,
            "width": img_w,
            "height": img_h,
        }

    raw_boxes = _first_nonempty(res, ("dt_polys", "det_boxes", "boxes", "rec_boxes"))
    raw_texts = _first_nonempty(res, ("rec_texts", "rec_text", "texts"))
    raw_scores = _first_nonempty(res, ("rec_scores", "rec_score", "scores"))

    if raw_boxes is None or raw_texts is None:
        return {
            "texts": texts,
            "scores": scores,
            "boxes": polys,
            "words": words,
            "width": img_w,
            "height": img_h,
        }

    boxes_seq = list(raw_boxes)
    texts_seq = list(raw_texts)
    scores_seq = list(raw_scores) if raw_scores is not None else None

    
    # 1) Выбираем ориентацию для ВСЕЙ страницы
    orient = choose_best_orientation(boxes_seq, texts_seq, img_w, img_h)
    print("Chosen orientation:", orient)  # можно раскомментировать для дебага
    
    # 2) Применяем её ко всем боксам
    for idx, (box, text) in enumerate(zip(boxes_seq, texts_seq)):
        if not text:
            continue
        print(f"box: {box}, type: {type(box)}")
        # confidence
        score = 1.0
        if scores_seq is not None and idx < len(scores_seq):
            try:
                score = float(scores_seq[idx])
            except Exception:
                score = 1.0

        if score < min_confidence:
            continue
        
        if DETECT_AND_ROTATE:
            orient = 'none'
        pts = transform_box_points(box, img_w, img_h, orient)
        bbox = compute_bbox_from_points(pts)
        if bbox is None:
            continue
        print(f"pts: {pts}, type: {type(pts)}")
        print(f"box: {list(box)}, type: {type(box)}")
        print(f"bbox: {bbox}, type: {type(bbox)}")


        x1, y1, x2, y2 = bbox
        
        
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        words.append(
            {
                "text": str(text),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
            }
        )
        texts.append(str(text))
        scores.append(float(score))
        polys.append(poly)

    return {
        "texts": texts,
        "scores": scores,
        "boxes": polys,
        "words": words,
        "width": img_w,
        "height": img_h,
    }

# -------------------------
# Визуализация
# -------------------------

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """
    Кросс-версионное измерение текста:
    - Pillow >= 10: draw.textbbox
    - старые версии: font.getsize
    """
    # Новый способ (Pillow 10+)
    if hasattr(draw, "textbbox"):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return int(w), int(h)
        except Exception:
            pass

    # Старый способ
    try:
        w, h = font.getsize(text)
        return int(w), int(h)
    except Exception:
        pass

    # Совсем грубый fallback
    size = getattr(font, "size", 16)
    return len(text) * size, size


def _load_nice_font(size: int) -> ImageFont.ImageFont:
    """
    Пытаемся взять нормальный TTF-шрифт (Windows и Linux).
    Если не нашли — падаем на default.
    """
    # 1) сначала пробуем по имени шрифта (на Windows обычно достаточно arial.ttf)
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass

    # 2) типичные пути Linux + явный путь для Windows
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in font_paths:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue

    # 3) совсем уж запасной вариант — встроенный мелкий шрифт
    return ImageFont.load_default()


def draw_overlay_pil(img_bgr, paddle_data: Dict[str, Any]) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    boxes = paddle_data.get("boxes") or []
    texts = paddle_data.get("texts") or []
    scores = paddle_data.get("scores") or []

    font = _load_nice_font(size=18)

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue

        xy = [(box[j][0], box[j][1]) for j in range(4)]
        draw.line(xy + [xy[0]], width=2, fill=(255, 255, 255))

        label = ""
        if i < len(texts):
            label += texts[i]
        if i < len(scores):
            label += f" ({scores[i]:.2f})"

        if label:
            x0, y0 = xy[0]
            tw, th = _measure_text(draw, label, font)
            draw.rectangle(
                (x0 + 1, y0 + 1, x0 + 1 + tw, y0 + 1 + th),
                fill=(255, 255, 255),
            )
            draw.text((x0 + 1, y0 + 1), label, font=font, fill=(0, 0, 0))

    return pil


def draw_paddle_on_white(img_bgr, paddle_data: Dict[str, Any], font_size: int | None = None) -> Image.Image:
    """
    Рисуем только результаты Paddle на белом фоне,
    чёрный шрифт нормального размера.
    """
    h, w = img_bgr.shape[:2]

    if font_size is None:
        # адаптивный размер: чем выше чек, тем больше шрифт
        font_size = max(12, h // 45)
        #font_size = 12

    pil = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(pil)

    boxes = paddle_data.get("boxes") or []
    texts = paddle_data.get("texts") or []

    font = _load_nice_font(size=font_size)

    line_width = max(2, h // 400)

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue

        xy = [(box[j][0], box[j][1]) for j in range(4)]
        draw.line(xy + [xy[0]], width=line_width, fill=(0, 0, 0))

        label = texts[i] if i < len(texts) else ""
        if not label:
            continue

        x0, y0 = xy[0]
        draw.text((x0 + 4, y0 + 2), label, font=font, fill=(0, 0, 0))

    return pil


def draw_overlay_on_white(img_bgr, paddle_data: Dict[str, Any]) -> Image.Image:
    """
    Старый вариант «просто белый фон», оставлен для совместимости.
    Сейчас он использует тот же крупный шрифт.
    """
    return draw_paddle_on_white(img_bgr, paddle_data, font_size=18)


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


def make_side_by_side_image(img_bgr, paddle_data: Dict[str, Any], font_size: int | None = None) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    left = Image.fromarray(img_rgb)

    right = draw_paddle_on_white(img_bgr, paddle_data, font_size=font_size)

    h = max(left.height, right.height)
    w = left.width + right.width

    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))

    return canvas


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
            img_path=img_path,
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

        if SAVE_SIDE_BY_SIDE:
            sbs = make_side_by_side_image(img_bgr, paddle_data)
            sbs.save(img_out / f"{safe_name}__paddlex_side_by_side.png")

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
