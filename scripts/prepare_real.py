# scripts/prepare_real.py
import zipfile
from pathlib import Path
import pandas as pd
import shutil
import re

RAW = Path("data/raw_zips")
OUT = Path("data/real")
IMAGES = OUT / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

DEFAULT_VIEW = "white"  # change later if you have both viewpoints

FRAME_RE = re.compile(r"(^|/)tagged_images/frame_(\d+)\.(jpg|png)$", re.IGNORECASE)

def _index_tagged_images(z: zipfile.ZipFile):
    """
    Build a mapping: frame_index (int) -> zip member name (str)
    for all files under tagged_images/frame_XXXXXX.(jpg|png)
    """
    mapping = {}
    for name in z.namelist():
        m = FRAME_RE.search(name)
        if not m:
            continue
        frame = int(m.group(2))
        # if duplicates exist (rare), keep the first
        mapping.setdefault(frame, name)
    return mapping

def main():
    rows = []
    total_missing = 0
    total_copied = 0

    zips = sorted(RAW.glob("game*_per_frame.zip"))
    if not zips:
        raise FileNotFoundError(
            "No zips found in data/raw_zips. Expected files like game2_per_frame.zip"
        )

    for zpath in zips:
        game = zpath.stem.replace("_per_frame", "")

        with zipfile.ZipFile(zpath, "r") as z:
            # find csv
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise RuntimeError(f"No CSV found inside {zpath.name}")
            csv_name = csvs[0]

            df = pd.read_csv(z.open(csv_name))
            if not {"from_frame", "fen"}.issubset(df.columns):
                raise RuntimeError(f"{csv_name} columns are {df.columns.tolist()} (missing from_frame/fen)")

            frame_to_member = _index_tagged_images(z)

            missing_here = []
            copied_here = 0

            for _, r in df.iterrows():
                frame = int(r["from_frame"])
                fen = str(r["fen"])

                member = frame_to_member.get(frame)
                if member is None:
                    missing_here.append(frame)
                    continue

                ext = Path(member).suffix.lower()  # .jpg or .png
                out_name = f"{game}_frame_{frame:06d}{ext}"
                out_path = IMAGES / out_name

                with z.open(member) as fsrc, open(out_path, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)

                rows.append([out_name, fen, DEFAULT_VIEW])
                copied_here += 1

            total_missing += len(missing_here)
            total_copied += copied_here

            if missing_here:
                # print a short warning with a few examples
                preview = ", ".join(str(x) for x in missing_here[:10])
                tail = " ..." if len(missing_here) > 10 else ""
                print(f"[WARN] {zpath.name}: missing {len(missing_here)} tagged images. Examples: {preview}{tail}")

            print(f"[OK]   {zpath.name}: copied {copied_here}/{len(df)} labeled frames")

    OUT.mkdir(parents=True, exist_ok=True)
    gt = pd.DataFrame(rows, columns=["image_name", "fen", "view"])
    gt.to_csv(OUT / "gt.csv", index=False)

    print("\n==== Summary ====")
    print(f"Total copied : {total_copied}")
    print(f"Total missing: {total_missing}")
    print(f"Saved: {OUT/'gt.csv'}")
    print(f"Images dir: {IMAGES.resolve()}")

if __name__ == "__main__":
    main()
