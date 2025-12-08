import json
from pathlib import Path
from collections import Counter

subtotal_counter = Counter()
total_counter = Counter()

with open(Path("data/cord_subset/cord_subset.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        cur_gt_parse = data["ground_truth_json"]["gt_parse"]

        if "sub_total" in cur_gt_parse and isinstance(cur_gt_parse["sub_total"], dict):
            subtotal_counter.update(cur_gt_parse["sub_total"].keys())

        if "total" in cur_gt_parse and isinstance(cur_gt_parse["total"], dict):
            total_counter.update(cur_gt_parse["total"].keys())

print("SUBTOTAL keys + counts:")
for k, v in subtotal_counter.most_common():
    print(f"{k}: {v}")

print("\nTOTAL keys + counts:")
for k, v in total_counter.most_common():
    print(f"{k}: {v}")
