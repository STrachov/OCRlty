import argparse, os, cv2, json
from lib.config import load_cfg
from lib.pipelines.invoice_extract import run

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--cfg", default="configs/base.yaml")
a = ap.parse_args()

cfg = load_cfg(a.cfg)
out = []
for name in os.listdir(a.input):
    ext = os.path.splitext(name)[1].lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        img = cv2.imread(os.path.join(a.input, name))
        out.append({"file": name, **run(img, cfg)})

print(json.dumps(out, ensure_ascii=False, indent=2))
