# tests/test_paddle_only.py
from __future__ import annotations
import os, sys, time, logging, tempfile
from typing import Any, Dict, List

# спокойные настройки для CPU/контейнера
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")

def _maybe_render_pdf_to_png(path: str) -> str:
    if not path.lower().endswith(".pdf"):
        return path
    import pypdfium2 as pdfium
    out = os.path.join(tempfile.gettempdir(), "paddle_test_page0.png")
    page = pdfium.PdfDocument(path)[0]
    page.render(scale=2).to_pil().save(out)
    return out

def _normalize_lines(result: Any) -> List[Dict[str, Any]]:
    """
    Приводит результат PaddleOCR к единому списку:
    [{'text': str, 'prob': float|None, 'bbox': ...}, ...]
    Поддерживает новый формат 3.x (dict с rec_texts/rec_scores) и старый list-of-lines.
    """
    lines: List[Dict[str, Any]] = []

    # Новый формат (3.x): dict
    if isinstance(result, dict):
        print('<insert>result: ', result)
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        boxes  = result.get("rec_polys") or result.get("dt_polys") or [None] * len(texts)
        for i, t in enumerate(texts):
            s = scores[i] if i < len(scores) else None
            b = boxes[i]  if i < len(boxes)  else None
            try:
                s = float(s) if s is not None else None
            except Exception:
                s = None
            lines.append({"text": str(t), "prob": s, "bbox": b})
        return lines

    # Иногда 3.x возвращает список страниц, где каждая страница — dict (как выше)
    if isinstance(result, list) and result and isinstance(result[0], dict):
        for page in result:
            lines.extend(_normalize_lines(page))
        return lines

    # Старый формат (2.x): [[ [bbox], (text, prob) ], ...]
    if isinstance(result, list):
        for page in result or []:
            for item in page or []:
                try:
                    bbox, (text, prob) = item
                    lines.append({"text": str(text), "prob": float(prob), "bbox": bbox})
                except Exception:
                    lines.append({"raw": item})
        return lines

    # Запасной случай — вернём сырое
    return [{"raw": result}]

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tests/test_paddle_only.py <image_or_pdf_path> [lang]")
        sys.exit(2)

    src_path = sys.argv[1]
    lang = (sys.argv[2] if len(sys.argv) >= 3 else os.getenv("PADDLE_LANG", "en")).lower()

    if not os.path.exists(src_path):
        print(f"File not found: {src_path}")
        sys.exit(2)

    # приглушаем лишние логи
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddle").setLevel(logging.ERROR)

    img_path = _maybe_render_pdf_to_png(src_path)

    from paddleocr import PaddleOCR

    # Можно добавить HPI/ORT при желании:
    # paddlex_config = {"hpi_config": {"backend": "onnxruntime"}}  # если ставил onnxruntime + hpi deps
    # ocr = PaddleOCR(lang=lang, device="cpu", paddlex_config=paddlex_config)

    ocr = PaddleOCR(
        lang=lang,
        device="cpu",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    t0 = time.time()
    result = ocr.predict(img_path)
    dt = time.time() - t0

    # версии
    try:
        import importlib.metadata as im
        print("paddleocr:", im.version("paddleocr"))
    except Exception:
        pass
    try:
        import paddle
        print("paddlepaddle:", getattr(paddle, "__version__", "unknown"))
    except Exception:
        pass

    print(f"\n=== PaddleOCR OK: {src_path} → {img_path} | elapsed {dt:.2f}s ===")

    lines = _normalize_lines(result)
    if not lines:
        print("No text lines detected.")
        return

    for i, ln in enumerate(lines[:20], 1):
        if "text" in ln:
            print(f"{i:02d}. {ln['text']}  (p={ln.get('prob')})")
        else:
            print(f"{i:02d}. RAW: {ln}")

    if len(lines) > 20:
        print(f"... and {len(lines)-20} more lines")

if __name__ == "__main__":
    main()
