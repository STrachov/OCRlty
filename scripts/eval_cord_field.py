#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # /workspace/src
LIB_ROOT = ROOT / "lib"                         # /workspace/src/lib

for p in (str(LIB_ROOT), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

from lib.pipelines.tilt_client import ArcticTiltClient
from lib.utils.parsing import parse_number, numbers_close


# ---- Константы окружения для твоего проекта ---- #

GT_PATH = Path("data/cord_subset/cord_gt.json")     # твой GT-файл
IMAGES_ROOT = Path(".")                             # корень для image_path из GT
BASE_URL = "http://127.0.0.1:8001/v1"               # tilt_api
MODEL_NAME = "Snowflake/snowflake-arctic-tilt-v1.3" # модель в tilt_api
TIMEOUT_S = 30.0
MAX_RETRIES = 2
TOLERANCE = 0.01                                    # допуск по деньгам
DETAILS_DIR = Path("data/eval")                     # куда складывать подробный JSON
LIMIT_SAMPLES: Optional[int] = None                 # можно поставить, например, 50


log = logging.getLogger("eval_cord_field")

log.info(f"ROOT: {ROOT}")
def guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


class EvalTiltClient(ArcticTiltClient):
    """
    Обёртка над ArcticTiltClient, которая:
      - не ожидает JSON от модели,
      - возвращает просто сырой текст и число:
        {"raw": <str>, "value": <float|None>}
    """

    def _parse_response(self, content: str) -> Dict[str, Any]:  # type: ignore[override]
        return {"raw": content, "value": parse_number(content)}


def load_gt(gt_path: Path) -> List[Dict[str, Any]]:
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {gt_path}, got {type(data)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic CORD mini-benchmark for a single numeric field."
    )
    parser.add_argument(
        "--field",
        required=True,
        help="Имя числового поля в cord_gt.json (например: total_price, cashprice, tax_price).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Промпт для TILT, который должен вернуть одно числовое значение (в текстовом виде).",
    )
    args = parser.parse_args()

    field_name: str = args.field
    question: str = args.prompt

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gt_records = load_gt(GT_PATH)
    log.info("Loaded %d GT records from %s", len(gt_records), GT_PATH)

    client = EvalTiltClient(
        base_url=BASE_URL,
        model=MODEL_NAME,
        timeout=TIMEOUT_S,
        max_retries=MAX_RETRIES,
        question=question,
    )

    n_total = 0   # сколько чеков имеют GT по этому полю
    n_pred = 0    # сколько чеков дали численное предсказание
    n_exact = 0   # сколько чеков попали в заданный допуск
    abs_errors: List[float] = []
    details: List[Dict[str, Any]] = []
    details.append(
                {
                    "field": field_name,
                    "prompt": question
                }
            )

    for idx, rec in enumerate(gt_records):
        if LIMIT_SAMPLES is not None and idx >= LIMIT_SAMPLES:
            break

        gt_val = rec.get(field_name)
        if gt_val is None:
            continue  # в GT нет значения этого поля — не учитываем

        n_total += 1

        sample_id = rec.get("id", f"sample_{idx}")
        image_rel = rec.get("image_file")
        if not image_rel:
            log.warning("[%s] No image_path/image_file in GT, skipping", sample_id)
            continue

        img_path = (IMAGES_ROOT / image_rel).resolve()
        if not img_path.is_file():
            log.warning("[%s] Image not found: %s", sample_id, img_path)
            details.append(
                {
                    "id": sample_id,
                    "status": "missing_image",
                    "image_path": str(img_path),
                    "gt_value": gt_val,
                    "field": field_name,
                }
            )
            continue

        content_type = guess_content_type(img_path)

        try:
            with img_path.open("rb") as f:
                doc_bytes = f.read()
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] Failed to read image: %s", sample_id, e)
            details.append(
                {
                    "id": sample_id,
                    "status": "image_file_read_error",
                    "image_path": str(img_path),
                    "gt_value": gt_val,
                    "field": field_name,
                    "error": str(e),
                }
            )
            continue

        try:
            result = client.infer(doc_bytes, content_type=content_type)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] Inference error: %s", sample_id, e)
            details.append(
                {
                    "id": sample_id,
                    "status": "infer_error",
                    "image_path": str(img_path),
                    "gt_value": gt_val,
                    "field": field_name,
                    "error": str(e),
                }
            )
            continue

        raw = result.get("raw")
        pred_val = result.get("value")

        record: Dict[str, Any] = {
            "id": sample_id,
            "status": "ok" if pred_val is not None else "no_number",
            "image_path": str(img_path),
            "field": field_name,
            "gt_value": gt_val,
            "pred_raw": raw,
            "pred_value": pred_val,
        }

        if pred_val is not None:
            n_pred += 1
            err = abs(pred_val - gt_val)
            abs_errors.append(err)
            record["abs_error"] = err
            record["exact_match"] = numbers_close(pred_val, gt_val, tol=TOLERANCE)

            if record["exact_match"]:
                n_exact += 1

        details.append(record)

    # ---- Метрики ---- #

    print(f"====== CORD mini-benchmark for field: {field_name} ======")
    print(f"Samples with GT {field_name}: {n_total}")

    if n_total > 0:
        coverage = n_pred / n_total
        acc = n_exact / n_total
        res_dict={
                    "coverage": coverage,
                    "acc": acc
                }
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else None

        print(f"Coverage (predicted number / GT-present): {coverage:.3f}")
        print(f"Exact match within tol={TOLERANCE}: {acc:.3f}")
        if mae is not None:
            print(f"MAE (mean absolute error): {mae:.3f}")
            res_dict["mae"] = mae
        else:
            print("MAE: N/A (no valid predictions)")
        
        details.insert(0, res_dict)
    else:
        print(f"No samples with GT {field_name} found.")

    # ---- Сохранение подробностей ---- #

    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = DETAILS_DIR / f"cord_eval_{field_name}.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Per-sample details saved to {details_path}")


if __name__ == "__main__":
    main()
