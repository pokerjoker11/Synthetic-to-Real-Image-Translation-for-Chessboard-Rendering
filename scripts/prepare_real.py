import csv
import re
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw_real")
OUT_DIR = Path("data/real")
IMG_DIR = OUT_DIR / "images"
GT_PATH = OUT_DIR / "gt.csv"

FRAME_RE = re.compile(r"tagged_images/frame_(\d+)\.jpg$", re.IGNORECASE)

def read_csv_rows(z: zipfile.ZipFile, csv_name: str):
    text = z.read(csv_name).decode("utf-8", errors="replace").splitlines()
    reader = csv.DictReader(text)
    rows = list(reader)
    # Build mapping: frame -> fen (handle duplicates)
    frame_to_fen = {}
    dup_count = 0
    for r in rows:
        fr = int(r["from_frame"])
        fen = r["fen"].strip()
        if fr in frame_to_fen and frame_to_fen[fr] != fen:
            dup_count += 1
        # keep last occurrence (deterministic)
        frame_to_fen[fr] = fen
    return frame_to_fen, dup_count

def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"[ERR] Missing folder: {RAW_DIR.resolve()}")

    zips = sorted(RAW_DIR.glob("game*_per_frame.zip"))
    if not zips:
        raise SystemExit(f"[ERR] No game*_per_frame.zip found in {RAW_DIR.resolve()}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total_copied = 0
    total_skipped_no_fen = 0
    total_dups = 0

    for zp in zips:
        m = re.search(r"(game\d+)_per_frame\.zip$", zp.name)
        if not m:
            print(f"[WARN] Unexpected zip name: {zp.name} (skipping)")
            continue
        game = m.group(1)
        csv_name = f"{game}.csv"

        with zipfile.ZipFile(zp, "r") as z:
            names = z.namelist()
            if csv_name not in names:
                print(f"[ERR] {zp.name}: missing {csv_name} (skipping)")
                continue

            frame_to_fen, dups = read_csv_rows(z, csv_name)
            total_dups += dups

            tagged = [n for n in names if FRAME_RE.search(n)]
            tagged.sort()

            copied = 0
            skipped_no_fen = 0

            for n in tagged:
                fr = int(FRAME_RE.search(n).group(1))
                fen = frame_to_fen.get(fr)
                if fen is None:
                    skipped_no_fen += 1
                    continue

                out_name = f"{game}_frame_{fr:06d}.jpg"
                out_path = IMG_DIR / out_name

                # extract into correct destination
                with z.open(n) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

                records.append({
                    "image": str(Path("images") / out_name).replace("\\", "/"),
                    "fen": fen,
                    "viewpoint": "white",   # real data is white-view in the provided dataset
                    "game": game,
                    "frame": fr,
                })
                copied += 1

            total_copied += copied
            total_skipped_no_fen += skipped_no_fen

            print(f"[OK] {zp.name}: copied {copied}/{len(tagged)} tagged images"
                  + (f" (skipped {skipped_no_fen} w/o fen)" if skipped_no_fen else "")
                  + (f" | csv duplicates w/ conflicting fen: {dups}" if dups else ""))

    # Write gt.csv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(GT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "fen", "viewpoint", "game", "frame"])
        w.writeheader()
        w.writerows(records)

    print("\n==== Summary ====")
    print(f"Total copied            : {total_copied}")
    print(f"Total skipped (no fen)  : {total_skipped_no_fen}")
    print(f"Total conflicting dups  : {total_dups}")
    print(f"Saved: {GT_PATH}")
    print(f"Images dir: {IMG_DIR}")

if __name__ == "__main__":
    main()
