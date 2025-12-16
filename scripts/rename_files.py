import re
from pathlib import Path

FOLDER = r"D:\master\OCRLty\out\paddle_debug\images"
PAT = re.compile(r".*(cord_\d{4,}).*", re.IGNORECASE)

def main():
    folder = Path(FOLDER)
    for p in folder.glob("*.png"):
        print(f"file: {p.name}")
        m = PAT.match(p.name)
        print(m)
        if not m:
            continue
        cord_id = m.group(1).lower()
        target = folder / f"{cord_id}.png"
        print(f"RENAME: {p.name} -> {target.name}")
        p.rename(target)

if __name__ == "__main__":
    main()
