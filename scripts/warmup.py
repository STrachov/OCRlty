import os, pathlib
from huggingface_hub import snapshot_download
VROOT = pathlib.Path(os.getenv("RUNPOD_VOLUME_ROOT", "/content/drive/MyDrive/runpod"))
MODELS = VROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
snapshot_download("Snowflake/snowflake-arctic-tilt-v1.3",
                  local_dir=str(MODELS / "snowflake-arctic-tilt-v1.3"),
                  local_dir_use_symlinks=False)
print("✅ Arctic-TILT ready at", MODELS)
