# OCRlty

R&D каркас под OCR + Arctic-TILT (GPU).

## Быстрый старт (Pod, NV)
```bash
export RUNPOD_VOLUME_ROOT=/workspace
python -m venv /workspace/venv-gpu
source /workspace/venv-gpu/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
pip install -e .
python scripts/warmup.py
python scripts/infer.py --input ${RUNPOD_VOLUME_ROOT}/data/samples


---

# Как всё создать и запушить (Colab/локально)

Замени `YOUR_USER` и `YOUR_REPO`:

```bash
# 1) Сгенерировать каркас локально/в Colab
export REPO_ROOT="your-project"
mkdir -p $REPO_ROOT/lib/{ocr,tilt,pipelines,utils} \
         $REPO_ROOT/apps/api-gpu \
         $REPO_ROOT/scripts \
         $REPO_ROOT/notebooks/gpu \
         $REPO_ROOT/configs

# файлы
cat > $REPO_ROOT/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "yourproject-core"
version = "0.1.0"
description = "Core logic (OCR + Arctic-TILT pipelines)"
requires-python = ">=3.10"
license = { text = "Proprietary" }
[tool.setuptools.packages.find]
where = ["."]
include = ["lib", "lib.*"]
[tool.setuptools]
package-dir = { "" = "." }
EOF

cat > $REPO_ROOT/requirements-gpu.txt <<'EOF'
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.4.0+cu124
torchvision==0.19.0+cu124
torchaudio==2.4.0+cu124
vllm==0.8.3
huggingface_hub>=0.23
tokenizers>=0.15,<0.20
sentencepiece>=0.1.99
tiktoken>=0.6
paddlepaddle-gpu>=2.6.0
paddleocr==2.7.0.3
onnxruntime-gpu>=1.17,<1.19
opencv-python>=4.9
shapely>=2.0
pyclipper>=1.3
rapidfuzz>=3.0
pypdfium2>=4.20
pdfminer.six>=20221105
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.5
loguru>=0.7
numpy>=1.25,<3
EOF

cat > $REPO_ROOT/configs/base.yaml <<'EOF'
paths: { models: "models", data: "data", cache: ".cache" }
ocr: { lang: "en" }
limits: { max_pages: 3 }
use_gpu: true
EOF

printf "" > $REPO_ROOT/lib/__init__.py

cat > $REPO_ROOT/lib/config.py <<'EOF'
import os, pathlib, yaml
def root():
    return pathlib.Path(os.getenv("RUNPOD_VOLUME_ROOT", "/content/drive/MyDrive/runpod")).expanduser()
def load_cfg(path="configs/base.yaml"):
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)
EOF

cat > $REPO_ROOT/lib/pipelines/invoice_extract.py <<'EOF'
def run(image_bgr, cfg: dict):
    # TODO: добавить OCR + Arctic-TILT
    return {"total": None, "vendor": None, "_meta": {"lines": 0}}
EOF

cat > $REPO_ROOT/scripts/warmup.py <<'EOF'
import os, pathlib
from huggingface_hub import snapshot_download
VROOT = pathlib.Path(os.getenv("RUNPOD_VOLUME_ROOT", "/content/drive/MyDrive/runpod"))
MODELS = VROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
snapshot_download("Snowflake/snowflake-arctic-tilt-v1.3",
                  local_dir=str(MODELS/"snowflake-arctic-tilt-v1.3"),
                  local_dir_use_symlinks=False)
print("✅ Arctic-TILT ready at", MODELS)
EOF

cat > $REPO_ROOT/scripts/infer.py <<'EOF'
import argparse, os, cv2, json
from lib.config import load_cfg
from lib.pipelines.invoice_extract import run
ap=argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--cfg", default="configs/base.yaml")
a=ap.parse_args(); cfg=load_cfg(a.cfg); out=[]
for name in os.listdir(a.input):
    if os.path.splitext(name)[1].lower() in {".png",".jpg",".jpeg"}:
        out.append({"file": name, **run(cv2.imread(os.path.join(a.input, name)), cfg)})
print(json.dumps(out, ensure_ascii=False, indent=2))
EOF

cat > $REPO_ROOT/apps/api-gpu/main.py <<'EOF'
from fastapi import FastAPI, UploadFile, File
import numpy as np, cv2
from lib.config import load_cfg
from lib.pipelines.invoice_extract import run
app = FastAPI(); cfg = load_cfg("configs/base.yaml")
@app.get("/health") 
def health(): return {"ok": True}
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return {"fields": run(img, cfg)}
EOF

cat > $REPO_ROOT/.gitignore <<'EOF'
__pycache__/
*.pyc
.env*
.ipynb_checkpoints/
notebooks/**/outputs/
# артефакты/данные — вне гита
/data/
/models/
/outputs/
/wheelhouse*/
/workspace/
/.runpod/
/.cache/
EOF

cat > $REPO_ROOT/README.md <<'EOF'
# your-project (R&D)

## Pod quick start
```bash
export RUNPOD_VOLUME_ROOT=/workspace
python -m venv /workspace/venv-gpu
source /workspace/venv-gpu/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
pip install -e .
python scripts/warmup.py
python scripts/infer.py --input ${RUNPOD_VOLUME_ROOT}/data/samples

EOF

```bash
# 2) Инициализировать git и запушить в новый репозиторий
cd "$REPO_ROOT"
git init
git add -A
git commit -m "bootstrap (lib+apps R&D skeleton)"
git branch -M main
git remote add origin https://github.com/STrachov/OCRlty.git
git push -u origin main

## Вариант с wheelhouse
export RUNPOD_VOLUME_ROOT=/workspace
python -m venv ${RUNPOD_VOLUME_ROOT}/venv-gpu
source ${RUNPOD_VOLUME_ROOT}/venv-gpu/bin/activate
export PIP_CACHE_DIR=/workspace/.cache/pip
mkdir -p ${RUNPOD_VOLUME_ROOT}/wheelhouse-gpu

# если wheelhouse ещё пустой — собрать
if [ ! "$(ls -A ${RUNPOD_VOLUME_ROOT}/wheelhouse-gpu 2>/dev/null)" ]; then
  pip install -U pip
  pip download -r requirements-gpu.txt -d ${RUNPOD_VOLUME_ROOT}/wheelhouse-gpu
fi

# установка офлайн из wheelhouse
pip install --no-index --find-links=${RUNPOD_VOLUME_ROOT}/wheelhouse-gpu -r requirements-gpu.txt

# Для Windows
# 1) Переменная окружения (на текущую сессию)
$env:RUNPOD_VOLUME_ROOT = "D:\master\ocrlty"   # укажи свой путь
# (опционально) создать папку, если нет
New-Item -ItemType Directory -Force -Path $env:RUNPOD_VOLUME_ROOT | Out-Null
# 2) Создать venv на нужной версии Python
py -3.11 -m venv "$env:RUNPOD_VOLUME_ROOT\venv-gpu"
# 3) Активировать venv
& "$env:RUNPOD_VOLUME_ROOT\venv-gpu\Scripts\Activate.ps1"
# Проверка
python --version




# подключить твой пакет (lib/) и прогреть веса
pip install -e .
python scripts/warmup.py
python scripts/infer.py --input ${RUNPOD_VOLUME_ROOT}/data/samples
