import json
from pathlib import Path
import re
from typing import Optional, Any

# SUBTOTAL keys + counts:
# subtotal_price: 44
# tax_price: 30
# service_price: 11
# etc: 4  (игнорируем)
# discount_price: 3
SUBTOTAL_FIELDS = ["subtotal_price", "tax_price", "service_price", "discount_price"]

# TOTAL keys + counts:
# total_price: 54
# cashprice: 39
# changeprice: 37
# menuqty_cnt: 17   (игнорируем для бенчмарка)
# creditcardprice: 7
# menutype_cnt: 2   (игнорируем)
# total_etc: 1      (игнорируем)
TOTAL_FIELDS = ["total_price", "cashprice", "changeprice", "creditcardprice"]

IN_PATH = Path("data/cord_subset/cord_subset.jsonl")
OUT_PATH = Path("data/cord_subset/cord_gt.json")


def parse_number(value: Optional[Any]) -> Optional[float]:
    """
    Приводит строку вида '1,234,567', '135,000', '4.50', '1,234.50'
    или даже '1,234 KRW' к float.
    Если value пустое/None/непарсибельное — возвращает None.
    """
    if value is None:
        return None

    # Уже число
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Убираем пробелы и запятые-разделители тысяч
    s = s.replace(",", "").replace(" ", "")

    # Сначала пробуем напрямую
    try:
        return float(s)
    except ValueError:
        pass

    # Пытаемся вытащить первую "похожую на число" подстроку
    #m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    m = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", s)

    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None

    return None


def main() -> None:
    gt_by_id: dict[str, dict[str, Any]] = {}

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # rec.keys() = ['id', 'image_path', 'ground_truth_raw', 'ground_truth_json']

            rec_id = rec["id"]  # это int из твоего предыдущего скрипта
            sample_id = f"cord_train_{rec_id:04d}"

            gt_json = rec.get("ground_truth_json") or {}
            gt_parse = gt_json.get("gt_parse") or {}
            total = gt_parse.get("total") or {}
            subtotal = gt_parse.get("sub_total") or {}

            # Делаем путь к картинке относительным (чтоб работало и на Win, и в контейнере)
            image_path = Path(rec["image_path"])
            # Например: images/cord_0000.jpg
            # Если у тебя уже относительный путь в jsonl — можно просто взять as_posix().
            if image_path.is_absolute():
                image_name = image_path.name
                image_rel = Path("images") / image_name
            else:
                image_rel = image_path

            gt_entry: dict[str, Any] = {
                "id": sample_id,
                "image_file": image_rel.as_posix(),
                # Сырые menu оставляем как есть на этом этапе
                # Потом при желании можно сделать нормализацию:
                # [{"name": ..., "qty": ..., "price": ...}, ...]
                "menu": gt_parse.get("menu") or [],
            }

            # TOTAL поля
            for field in TOTAL_FIELDS:
                gt_entry[field] = parse_number(total.get(field))

            # SUBTOTAL поля
            for field in SUBTOTAL_FIELDS:
                gt_entry[field] = parse_number(subtotal.get(field))

            gt_by_id[sample_id] = gt_entry

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(gt_by_id, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(gt_by_id)} records into {OUT_PATH}")


if __name__ == "__main__":
    main()
