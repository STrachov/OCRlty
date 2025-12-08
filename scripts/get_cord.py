# scripts/get_cord_local.py
import argparse
import json
from pathlib import Path
from io import BytesIO

from datasets import load_dataset
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=56,
        help="Сколько примеров взять из шарда",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/cord_subset",
        help="Куда сохранить поднабор (картинки + GT)",
    )
    args = parser.parse_args()

    parquet_path = Path("data/train-00000-of-00004-b4aaeceff1d90ecb.parquet")
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading local parquet: {parquet_path}")
    ds = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",  # для parquet-билдера это просто имя сплита, можно любое
    )

    n = min(args.num_samples, len(ds))
    print(f"Dataset size in this shard: {len(ds)}; taking first {n} samples")

    jsonl_path = out_dir / "cord_subset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f_out:
        for i in range(n):
            ex = ds[i]

            # ex["image"] может быть:
            # 1) уже PIL.Image (если datasets сам декодировал)
            # 2) dict {"bytes": ..., "path": "..."} — если нет
            img_obj = ex["image"]
            if isinstance(img_obj, Image.Image):
                img = img_obj
            elif isinstance(img_obj, dict) and "bytes" in img_obj:
                img = Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
            else:
                raise RuntimeError(f"Неожиданный формат image в примере {i}: {type(img_obj)}")

            img_name = f"cord_{i:04d}.jpg"
            img_path = img_dir / img_name
            img.save(img_path, format="JPEG")

            gt_str = ex.get("ground_truth", "")
            try:
                gt_json = json.loads(gt_str)
            except Exception:
                gt_json = None

            rec = {
                "id": i,
                "image_path": str(img_path),
                "ground_truth_raw": gt_str,
                "ground_truth_json": gt_json,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {n} samples to {out_dir}")
    print(f"- images -> {img_dir}")
    print(f"- metadata -> {jsonl_path}")


if __name__ == "__main__":
    main()
