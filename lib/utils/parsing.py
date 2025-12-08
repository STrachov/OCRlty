# lib/utils/number_parsing.py
from __future__ import annotations

import re
from typing import Any, Optional


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def parse_number(value: Optional[Any]) -> Optional[float]:
    """
    Унифицированный парсер чисел из строк:
    - поддерживает int/float → float;
    - строки с запятыми как разделителями тысяч: '1,234.50' → 1234.5;
    - строки с лишним текстом: 'Total: 1,234.50 UAH' → 1234.5;
    - если число не найдено, возвращает None.
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
    s_clean = s.replace(" ", "").replace("\u00A0", "").replace(",", "")

    # Сначала пробуем прямой float
    try:
        return float(s_clean)
    except ValueError:
        pass

    # Фоллбэк: ищем число внутри строки
    m = _NUMBER_RE.search(s_clean)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None

    return None


def numbers_close(a: float, b: float, tol: float = 1e-2) -> bool:
    """
    Сравнение двух чисел с абсолютным допуском (по умолчанию 0.01).
    Для денежных величин обычно этого достаточно.
    """
    return abs(a - b) <= tol
