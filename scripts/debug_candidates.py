from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]      # /workspace/src
LIB_ROOT = ROOT / "lib"                         # /workspace/src/lib

for p in (str(LIB_ROOT), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipelines.tilt_client import ArcticTiltClient


client = ArcticTiltClient(
    base_url="http://127.0.0.1:8001",  # любой, не используется
    model="snowflake-arctic-tilt",     # любой, не используется
)

img = Path("data/cord_subset/images/cord_0046.jpg").read_bytes()

dbg = client.debug_candidates(
    img,
    content_type="image/png",
    field_name="total_price",
    question='Pick the TOTAL amount.\nReturn EXACTLY ONE of these strings:\n{{}}\nOutput only the chosen string.',
    max_neighbours=3,
)

print("CANDIDATES:", dbg["candidates"])
print("USED QUESTION:\n", dbg["used_question"])
print("PREVIEW:", dbg["pages_payload_preview"])
