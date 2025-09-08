import os, pathlib, yaml

def root() -> pathlib.Path:
    # на Pod/Serverless выставляй RUNPOD_VOLUME_ROOT: /workspace или /runpod-volume
    return pathlib.Path(os.getenv("RUNPOD_VOLUME_ROOT", "/content/drive/MyDrive/runpod")).expanduser()

def load_cfg(path: str = "configs/base.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
