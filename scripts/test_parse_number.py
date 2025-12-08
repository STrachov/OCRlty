#!/usr/bin/env python
import json
import re
from pathlib import Path
from typing import Any, Optional

IN_PATH = Path("data/cord_subset/cord_subset.jsonl")

SUBTOTAL_FIELDS = ["subtotal_price", "tax_price", "service_price", "discount_price"]
TOTAL_FIELDS = ["total_price", "cashprice", "changeprice", "creditcardprice"]


def parse_number(value: Optional[Any]) -> Optional[float]:
    """
    Тот же parse_number, который мы использовали для GT:
    - поддерживает строки с запятыми-разделителями тысяч,
    - возвращает float или None.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Убираем запятые как разделители тысяч
    s_clean = s.replace(",", "")

    # Сначала пробуем прямой float
    try:
        return float(s_clean)
    except ValueError:
        pass

    # Фоллбэк: ищем число в строке
    # Поддерживаем и "123.45", и ".45"
    m = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", s_clean)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None

    return None


def expected_from_simple_digits(raw: str) -> Optional[float]:
    """
    Если строка состоит только из цифр и запятых (наподобие '1,234,567'),
    считаем 'ожидаемым' значением float(raw.replace(',', '')).
    Для всех остальных — возвращаем None (ожидание не определяем).
    """
    s = raw.strip()
    if not s:
        return None

    if not re.fullmatch(r"[0-9,]+", s):
        return None

    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def main() -> None:
    none_cases = []        # parse_number вернул None
    mismatch_cases = []    # parse_number != expected_from_simple_digits

    total_values_count = 0

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec_id = rec["id"]
            sample_id = f"cord_train_{rec_id:04d}"
            gt_json = rec.get("ground_truth_json") or {}
            gt_parse = gt_json.get("gt_parse") or {}
            total = gt_parse.get("total") or {}
            subtotal = gt_parse.get("sub_total") or {}

            # Проверяем TOTAL поля
            for field in TOTAL_FIELDS:
                raw = total.get(field)
                if raw is None or (isinstance(raw, str) and not raw.strip()):
                    continue

                total_values_count += 1
                parsed = parse_number(raw)
                print(f"Raw: {raw}, parsed: {parsed}")

                if parsed is None:
                    none_cases.append(
                        {
                            "sample_id": sample_id,
                            "section": "total",
                            "field": field,
                            "raw": raw,
                        }
                    )
                else:
                    expected = expected_from_simple_digits(str(raw))
                    if expected is not None and expected != parsed:
                        mismatch_cases.append(
                            {
                                "sample_id": sample_id,
                                "section": "total",
                                "field": field,
                                "raw": raw,
                                "parsed": parsed,
                                "expected": expected,
                            }
                        )

            # Проверяем SUBTOTAL поля
            for field in SUBTOTAL_FIELDS:
                raw = subtotal.get(field)
                if raw is None or (isinstance(raw, str) and not raw.strip()):
                    continue

                total_values_count += 1
                parsed = parse_number(raw)
                print(f"Raw: {raw}, parsed: {parsed}")

                if parsed is None:
                    none_cases.append(
                        {
                            "sample_id": sample_id,
                            "section": "sub_total",
                            "field": field,
                            "raw": raw,
                        }
                    )
                else:
                    expected = expected_from_simple_digits(str(raw))
                    if expected is not None and expected != parsed:
                        mismatch_cases.append(
                            {
                                "sample_id": sample_id,
                                "section": "sub_total",
                                "field": field,
                                "raw": raw,
                                "parsed": parsed,
                                "expected": expected,
                            }
                        )

    print(f"Всего числовых значений, которые проверили: {total_values_count}")
    print(f"parse_number вернул None для {len(none_cases)} случаев")
    print(f"parse_number != expected_from_simple_digits для {len(mismatch_cases)} случаев")
    print()

    if none_cases:
        print("=== СЛУЧАИ, ГДЕ parse_number ВЕРНУЛ None ===")
        for case in none_cases:
            print(
                f"[{case['sample_id']}] {case['section']}.{case['field']}: "
                f"raw={case['raw']!r}"
            )
        print()

    if mismatch_cases:
        print("=== СЛУЧАИ НЕОЖИДАННОГО ЗНАЧЕНИЯ (parsed != expected) ===")
        for case in mismatch_cases:
            print(
                f"[{case['sample_id']}] {case['section']}.{case['field']}: "
                f"raw={case['raw']!r} -> parsed={case['parsed']} (expected {case['expected']})"
            )


if __name__ == "__main__":
    main()
