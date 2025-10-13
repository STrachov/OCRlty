# lib/post/rules.py
from __future__ import annotations

import os
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, Iterable, Optional, Tuple
from datetime import datetime

# Версию правил можно читать в API для метаданных
RULESET_VERSION = os.getenv("RULESET_VERSION", "rules-0.1.0")

# Предпочтение формата даты, если неоднозначно: "DMY" (Европа) или "MDY" (США)
DATE_PREF = os.getenv("DATE_PREF", "DMY").upper()

# Допустимое расхождение при проверке subtotal + tax ≈ total
AMOUNT_EPS = Decimal(os.getenv("AMOUNT_EPS", "0.02"))  # 2 цента по умолчанию


_CURRENCY_SIGNS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₽": "RUB",
    "р": "RUB",  # иногда печатают 'р' вместо символа
    "₴": "UAH",
    "₸": "KZT",
    "₺": "TRY",
    "zł": "PLN",
    "₦": "NGN",
    "₹": "INR",
    "₫": "VND",
    "R$": "BRL",
    "C$": "CAD",
    "A$": "AUD",
}

# Часто встречающиеся алиасы ключей в «сырых» ответах
_ALIASES = {
    "merchant": ["merchant", "seller", "vendor", "store", "supplier", "company"],
    "date": ["date", "invoice_date", "issue_date", "txn_date", "purchase_date"],
    "currency": ["currency", "curr", "ccy"],
    "subtotal": ["subtotal", "net", "net_amount", "amount_net", "sub_total"],
    "tax_amount": ["tax_amount", "tax", "vat", "vat_amount", "taxes"],
    "total": ["total", "grand_total", "amount_total", "sum", "balance_due"],
}


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _strip_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _title_safe(s: str) -> str:
    # простая канонизация названия продавца
    s = _strip_ws(s)
    # не используем .title() слепо (испортит McDonald's и т.п.)
    # здесь можно встроить свои правила; пока просто нормализуем пробелы
    return s


def _detect_currency(raws: Iterable[Any]) -> Optional[str]:
    # 1) символ валюты
    for raw in raws:
        if not isinstance(raw, str):
            continue
        txt = raw.strip()
        # длинные символы сначала (R$, A$), чтобы не сработал одиночный "$"
        for sign in sorted(_CURRENCY_SIGNS.keys(), key=len, reverse=True):
            if sign in txt:
                return _CURRENCY_SIGNS[sign]
    # 2) трёхбуквенные коды
    for raw in raws:
        if not isinstance(raw, str):
            continue
        m = re.search(r"\b([A-Za-z]{3})\b", raw)
        if m:
            return m.group(1).upper()
    return None


def _to_decimal(x: Any) -> Optional[Decimal]:
    if x is None or x == "":
        return None
    if isinstance(x, (int, float, Decimal)):
        try:
            return Decimal(str(x))
        except InvalidOperation:
            return None
    if isinstance(x, str):
        s = x.strip()
        # выкинуть любые буквы и символы валют/пробелы, оставить цифры, точку, запятую, минус
        s = re.sub(r"[^\d,.\-]", "", s)
        if s == "":
            return None
        # определить десятичный разделитель (если и точка и запятая)
        if "," in s and "." in s:
            # heuristic: правый-most символ из {',','.'} — это десятичный
            last_sep = max(s.rfind(","), s.rfind("."))
            dec_sep = s[last_sep]
            thou_sep = "," if dec_sep == "." else "."
            s = s.replace(thou_sep, "")
            s = s.replace(dec_sep, ".")
        else:
            # если только запятая — считаем это десятичным разделителем
            if "," in s and "." not in s:
                s = s.replace(",", ".")
            # если только точка — уже ок
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    # нераспознанный тип
    return None


def _round2(d: Optional[Decimal]) -> Optional[Decimal]:
    if d is None:
        return None
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---- Даты ----

try:
    from dateutil import parser as date_parser  # type: ignore
    _HAS_DATEUTIL = True
except Exception:
    _HAS_DATEUTIL = False

# простые паттерны, если dateutil недоступен
_DATE_PATTERNS = [
    # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    (re.compile(r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*$"), "YMD"),
    # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
    (re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\s*$"), "DMY"),
    # MM-DD-YYYY, MM/DD/YYYY, MM.DD.YYYY
    (re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\s*$"), "MDY"),
]

def _norm_date(raw: Any) -> Optional[str]:
    if not raw:
        return None
    if isinstance(raw, str):
        s = raw.strip()
    else:
        s = str(raw).strip()

    if s == "":
        return None

    # Сначала пробуем dateutil (если есть)
    if _HAS_DATEUTIL:
        try:
            # dayfirst = True для Европы (DATE_PREF="DMY")
            dt = date_parser.parse(s, dayfirst=(DATE_PREF == "DMY"))
            return dt.date().isoformat()
        except Exception:
            pass

    # Фолбэк: пробуем простые регулярки
    for rx, kind in _DATE_PATTERNS:
        m = rx.match(s)
        if not m:
            continue
        g1, g2, g3 = m.groups()
        try:
            if kind == "YMD":
                y, mth, d = int(g1), int(g2), int(g3)
            elif kind == "DMY":
                d, mth, y = int(g1), int(g2), int(g3)
            else:  # MDY
                mth, d, y = int(g1), int(g2), int(g3)
            # Простейшая валидация
            dt = datetime(year=y, month=mth, day=d).date()
            return dt.isoformat()
        except Exception:
            continue
    return None


def _coalesce_amounts(
    subtotal: Optional[Decimal],
    tax: Optional[Decimal],
    total: Optional[Decimal],
) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal], list[str]]:
    """
    Пытается вычислить недостающие суммы и проверить правило:
    subtotal + tax ≈ total (с допуском AMOUNT_EPS).
    """
    warnings: list[str] = []

    # Попробуем вычислить пропущенное
    if subtotal is not None and total is not None and tax is None:
        tax = _round2(total - subtotal)
        warnings.append("computed.tax_amount")
    elif subtotal is not None and tax is not None and total is None:
        total = _round2(subtotal + tax)
        warnings.append("computed.total")
    elif tax is not None and total is not None and subtotal is None:
        subtotal = _round2(total - tax)
        warnings.append("computed.subtotal")

    # Проверка согласованности
    if subtotal is not None and tax is not None and total is not None:
        if (subtotal + tax - total).copy_abs() > AMOUNT_EPS:
            warnings.append("mismatch.subtotal_plus_tax_ne_total")

    return subtotal, tax, total, warnings


def postprocess_rules(raw_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Нормализует поля, вычисляет недостающие суммы и возвращает
    словарь с каноническими ключами + warnings.
    Ожидает, что вход может быть «грязным» (строки, символы валют, лишние пробелы).
    """
    out: Dict[str, Any] = {}
    warnings: list[str] = []

    # --- merchant ---
    merchant_raw = _first_present(raw_fields, _ALIASES["merchant"]) or raw_fields.get("MERCHANT")
    if isinstance(merchant_raw, str):
        out["merchant"] = _title_safe(merchant_raw)
    elif merchant_raw:
        out["merchant"] = _title_safe(str(merchant_raw))
    else:
        out["merchant"] = None

    # --- date ---
    date_raw = _first_present(raw_fields, _ALIASES["date"]) or raw_fields.get("DATE")
    out["date"] = _norm_date(date_raw)

    # --- currency ---
    currency_raw = _first_present(raw_fields, _ALIASES["currency"])
    if isinstance(currency_raw, str):
        currency_norm = currency_raw.strip().upper()
    elif currency_raw:
        currency_norm = str(currency_raw).strip().upper()
    else:
        currency_norm = None

    # Попробуем вытащить валюту из строк сумм (по символам)
    totals_texts = []
    for k in ("subtotal", "tax_amount", "total"):
        for alias in _ALIASES[k]:
            v = raw_fields.get(alias)
            if isinstance(v, str):
                totals_texts.append(v)

    auto_ccy = _detect_currency([currency_norm] + totals_texts if currency_norm else totals_texts)
    out["currency"] = (currency_norm or auto_ccy or None)

    # --- amounts ---
    sub_raw = _first_present(raw_fields, _ALIASES["subtotal"])
    tax_raw = _first_present(raw_fields, _ALIASES["tax_amount"])
    tot_raw = _first_present(raw_fields, _ALIASES["total"])

    sub = _to_decimal(sub_raw)
    tax = _to_decimal(tax_raw)
    tot = _to_decimal(tot_raw)

    # отрицательные → предупреждение и abs()
    for name, val in (("subtotal", sub), ("tax_amount", tax), ("total", tot)):
        if val is not None and val < 0:
            warnings.append(f"negative.{name}")
            if name == "subtotal":
                sub = -val
            elif name == "tax_amount":
                tax = -val
            else:
                tot = -val

    # вычислить недостающее и проверить правило
    sub, tax, tot, w = _coalesce_amounts(sub, tax, tot)
    warnings.extend(w)

    # округление до 2 знаков
    sub = _round2(sub)
    tax = _round2(tax)
    tot = _round2(tot)

    out["subtotal"] = float(sub) if sub is not None else None
    out["tax_amount"] = float(tax) if tax is not None else None
    out["total"] = float(tot) if tot is not None else None

    # --- финальные штрихи ---
    # Если нет даты/валюты — добавим предупреждения (можно использовать для fallback-логики)
    if out.get("date") is None:
        warnings.append("missing.date")
    if out.get("currency") is None:
        warnings.append("missing.currency")

    # прикладываем предупреждения
    if warnings:
        out["warnings"] = warnings

    return out
