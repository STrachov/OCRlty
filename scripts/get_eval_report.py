#!/usr/bin/env python
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================
# НАСТРОЙКИ
# =========================
EVAL_PATH: Path = Path("data\cord_subset\coord_eval_totals.json")
OUT_DIR: Path = Path("out/mismatches_html")
MIN_ERROR: float = 0.0
MAX_SAMPLES: int = 100  # 0 = без лимита
# =========================


def load_eval(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of records in {path}, got {type(data)}")
    return data


def html_escape(value: Any) -> str:
    return html.escape(str(value))


def main() -> None:
    records = load_eval(EVAL_PATH)

    # --- глобальные агрегаты по эксперименту --- #
    with_gt = [r for r in records if r.get("gt_value") is not None]
    #with_pred = [r for r in with_gt if r.get("pred_value") is not None]
    with_pred = [r for r in with_gt if r.get("pred_raw") is not None]
    
    exact = [r for r in with_gt if r.get("exact_match") is True]

    n_total = len(records)
    n_gt = len(with_gt)
    n_pred = len(with_pred)
    n_exact = len(exact)

    coverage = n_pred / n_gt if n_gt else 0.0
    accuracy = n_exact / n_gt if n_gt else 0.0

    field_name: Optional[str] = None
    for r in records:
        if r.get("field"):
            field_name = str(r["field"])
            break

    # --- выбираем только проблемные кейсы --- #
    mismatches: List[Dict[str, Any]] = []
    for rec in records:
        gt = rec.get("gt_value")
        if gt is None:
            continue  # без GT неинтересно

        status = rec.get("status")
        pred = rec.get("pred_value")
        abs_err = rec.get("abs_error")

        reason: str

        # Любой статус не "ok" — сразу считаем промахом
        if status != "ok":
            reason = str(status)
            rec["_reason"] = reason
            mismatches.append(rec)
            continue

        # status == "ok", но предсказанного числа нет
        if pred is None:
            reason = "no_number"
            rec["_reason"] = reason
            mismatches.append(rec)
            continue

        # Есть число, но смотрим на abs_error
        if isinstance(abs_err, (int, float)):
            if abs_err > MIN_ERROR:
                reason = "abs_error"
                rec["_reason"] = reason
                mismatches.append(rec)
        else:
            # нет abs_error, но статус "ok"
            reason = "no_abs_error"
            rec["_reason"] = reason
            mismatches.append(rec)

    # сортируем: сначала большие abs_error, потом no_number/прочие
    def sort_key(r: Dict[str, Any]) -> float:
        err = r.get("abs_error")
        if isinstance(err, (int, float)):
            return float(err)
        return -1.0  # reverse=True => уйдёт в хвост

    mismatches.sort(key=sort_key, reverse=True)

    if MAX_SAMPLES > 0:
        mismatches = mismatches[:MAX_SAMPLES]

    print(f"Всего записей в eval: {n_total}")
    print(f"С GT по полю:        {n_gt}")
    print(f"С предсказанием:     {n_pred}")
    print(f"Точное совпадение:   {n_exact}")
    print(f"Coverage:            {coverage:.3f}")
    print(f"Accuracy:            {accuracy:.3f}")
    print(f"Проблемных кейсов:   {len(mismatches)} (min_error={MIN_ERROR})")

    # --- готовим папку и копируем картинки --- #
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_with_local_img: List[Dict[str, Any]] = []

    for idx, rec in enumerate(mismatches):
        img_path_str = "../../"+rec.get("image_path").replace('/workspace/src','')
        print(f"img_path_str: {img_path_str}")
        local_img_name: Optional[str] = None

        src = Path(img_path_str)
        #print(f"src: {src}")
        #local_img_name = f"{idx:03d}_{src.name}"
        #file:///D:/master/OCRLty/out/data/cord_subset/images/cord_0000.jpg
            
            

        rec["_local_img"] = src
        cases_with_local_img.append(rec)

    # --- генерим HTML --- #
    html_path = OUT_DIR / "index.html"

    lines: List[str] = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang='en'>")
    lines.append("<head>")
    lines.append("  <meta charset='utf-8'/>")
    title = f"CORD eval mismatches for field {field_name or ''}".strip()
    lines.append(f"  <title>{html_escape(title)}</title>")
    lines.append(
        "  <style>"
        "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 16px; }"
        ".summary { margin-bottom: 24px; }"
        ".grid { display: flex; flex-wrap: wrap; gap: 16px; }"
        ".card { border: 1px solid #ddd; border-radius: 8px; padding: 8px; width: 360px; box-sizing: border-box; }"
        ".card img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }"
        ".meta { font-size: 14px; margin-top: 8px; }"
        ".meta table { width: 100%; border-collapse: collapse; }"
        ".meta th, .meta td { text-align: left; padding: 2px 4px; }"
        ".meta th { width: 120px; color: #555; }"
        ".badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 4px; }"
        ".badge-ok { background: #e6f6e6; color: #256029; }"
        ".badge-err { background: #fdecea; color: #b71c1c; }"
        ".badge-warn { background: #fff4e5; color: #8a4b0f; }"
        "</style>"
    )
    lines.append("</head>")
    lines.append("<body>")

    lines.append("<div class='summary'>")
    lines.append(f"  <h1>{html_escape(title)}</h1>")
    lines.append("  <ul>")
    lines.append(f"    <li>Total records in eval: {n_total}</li>")
    lines.append(f"    <li>With GT: {n_gt}</li>")
    lines.append(f"    <li>With prediction: {n_pred}</li>")
    lines.append(f"    <li>Exact matches: {n_exact}</li>")
    lines.append(f"    <li>Coverage (pred/GT): {coverage:.3f}</li>")
    lines.append(f"    <li>Accuracy (exact/GT): {accuracy:.3f}</li>")
    lines.append(f"    <li>Mismatches in this HTML: {len(cases_with_local_img)}</li>")
    lines.append(f"    <li>min_error threshold: {MIN_ERROR}</li>")
    lines.append("  </ul>")
    lines.append("</div>")

    lines.append("<div class='grid'>")

    for rec in cases_with_local_img:
        rid = rec.get("id")
        status = rec.get("status")
        reason = rec.get("_reason")
        gt = rec.get("gt_value")
        pred = rec.get("pred_raw")# ex pred_value
        abs_err = rec.get("abs_error")
        field = rec.get("field")
        img_name = rec.get("_local_img")
        error_text = rec.get("error")

        # стили бейджа по статусу/причине
        if status == "ok" and reason == "abs_error":
            badge_class = "badge-warn"
        elif status == "ok":
            badge_class = "badge-ok"
        else:
            badge_class = "badge-err"

        lines.append("<div class='card'>")

        # картинка
        if img_name:
            lines.append(
                f"  <img src='{html_escape(img_name)}' alt='{html_escape(str(rid))}' />"
            )
        else:
            lines.append(
                "  <div style='height: 200px; display:flex; align-items:center; "
                "justify-content:center; color:#999; border:1px dashed #ccc;'>No image</div>"
            )

        # метаданные
        lines.append("  <div class='meta'>")
        lines.append(
            f"    <div><span class='badge {badge_class}'>"
            f"{html_escape(str(status))}</span>"
            f"<span class='badge badge-warn'>{html_escape(str(reason))}</span></div>"
        )
        lines.append("    <table>")
        lines.append(f"      <tr><th>ID</th><td>{html_escape(rid)}</td></tr>")
        if field is not None:
            lines.append(f"      <tr><th>Field</th><td>{html_escape(field)}</td></tr>")
        lines.append(f"      <tr><th>GT value</th><td>{html_escape(gt)}</td></tr>")
        lines.append(f"      <tr><th>Pred raw</th><td>{html_escape(pred)}</td></tr>")
        lines.append(f"      <tr><th>Abs error</th><td>{html_escape(abs_err)}</td></tr>")
        if error_text:
            lines.append(f"      <tr><th>Error</th><td>{html_escape(error_text)}</td></tr>")
        lines.append("    </table>")
        lines.append("  </div>")

        lines.append("</div>")  # card

    lines.append("</div>")  # grid
    lines.append("</body>")
    lines.append("</html>")

    html_str = "\n".join(lines)
    with html_path.open("w", encoding="utf-8") as f_html:
        f_html.write(html_str)

    print(f"\nHTML отчёт сохранён в: {html_path}")
    print("Скачай эту папку целиком и просто открой index.html в браузере.")


if __name__ == "__main__":
    main()
