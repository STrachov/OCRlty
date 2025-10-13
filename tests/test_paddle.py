# notebooks/dev_snippet_or_lib_example.py (или в твоём пайплайне)
from paddleocr import PaddleOCR
from lib.pipelines.tilt_prep_1 import build_ocr_document, make_messages, call_tilt
import os
import json
import sys
from typing import Optional, List
import tempfile
import pypdfium2 as pdfium

def _render_pdf_to_pngs(
    pdf_path: str,
    dpi: int = 200,
    max_pages: Optional[int] = None,
    workdir: Optional[str] = None,
    filename_prefix: str = "paddle_page",
) -> List[str]:
    """
    Рендерит все (или первые max_pages) страницы PDF в PNG и возвращает список путей.
    DPI влияет на чёткость (скейл = dpi/72).
    """
    assert pdf_path.lower().endswith(".pdf"), "Это не PDF"

    tmp_dir = workdir or tempfile.mkdtemp(prefix="paddle_pdf_")
    doc = pdfium.PdfDocument(pdf_path)
    n_pages = len(doc)
    pages = range(min(n_pages, max_pages or n_pages))

    scale = dpi / 72  # PDF-единицы → нужный DPI
    out_paths: List[str] = []

    for i in pages:
        page = doc[i]                 # pypdfium2 поддерживает индексирование
        pil = page.render(scale=scale).to_pil()
        out_path = os.path.join(tmp_dir, f"{filename_prefix}_{i+1:04d}.png")
        pil.save(out_path)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tests/test_paddle_only.py <image_or_pdf_path> [lang]")
        sys.exit(2)

    src_path = sys.argv[1]
    if src_path.lower().endswith(".pdf"):
        image_path = _render_pdf_to_pngs(src_path)
    else:
        image_path = [src_path]
    print('++++image_path: ', image_path)
    
    lang = (sys.argv[2] if len(sys.argv) >= 3 else os.getenv("PADDLE_LANG", "en")).lower()

    ocr = PaddleOCR(
        lang=lang, device="cpu",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    pred = ocr.predict(image_path)
    ocr_doc = build_ocr_document(image_path, pred)
    messages = make_messages(
        ocr_doc,
        system_prompt="Extract merchant, date (YYYY-MM-DD), currency (ISO), subtotal, tax_amount, total. Return strict JSON."
    )
    print("++++messages:")
    print(messages)
    #print(json.dumps(messages, indent=2, ensure_ascii=False))
    #messages = json.dumps(messages)

    content = call_tilt("http://mock-vllm:8001/v1", "Snowflake/snowflake-arctic-tilt-v1.3", messages)
    try:
        print(json.loads(content))
    except Exception:
        print({"raw": content})
    
   
if __name__ == "__main__":
    main()