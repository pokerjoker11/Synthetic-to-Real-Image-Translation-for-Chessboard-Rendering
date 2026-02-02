import csv
import re
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw_real")

def read_game_csv(z: zipfile.ZipFile, csv_name: str):
    with z.open(csv_name) as f:
        # bytes -> text
        text = f.read().decode("utf-8", errors="replace").splitlines()
    reader = csv.DictReader(text)
    rows = list(reader)
    # expecting columns: from_frame,to_frame,fen
    frame_to_fen = {}
    for r in rows:
        fr = int(r["from_frame"])
        fen = r["fen"].strip()
        frame_to_fen[fr] = fen
    return rows, frame_to_fen

def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"[ERR] Missing folder: {RAW_DIR.resolve()}")

    zips = sorted(RAW_DIR.glob("game*_per_frame.zip"))
    if not zips:
        raise SystemExit(f"[ERR] No game*_per_frame.zip found in {RAW_DIR.resolve()}")

    print(f"[OK] Found {len(zips)} zip(s) in {RAW_DIR}")

    total_imgs = 0
    total_rows = 0

    for zp in zips:
        with zipfile.ZipFile(zp, "r") as z:
            names = z.namelist()

            m = re.search(r"(game\d+)_per_frame\.zip$", zp.name)
            if not m:
                print(f"[WARN] Unexpected zip name: {zp.name}")
                continue
            game = m.group(1)  # e.g. game2
            csv_name = f"{game}.csv"
            if csv_name not in names:
                print(f"[ERR] {zp.name}: missing {csv_name}")
                continue

            rows, frame_to_fen = read_game_csv(z, csv_name)

            tagged = [n for n in names if n.startswith("tagged_images/") and n.lower().endswith(".jpg")]
            # extract frame ids from filenames like frame_000200.jpg
            frame_ids = []
            for n in tagged:
                mm = re.search(r"frame_(\d+)\.jpg$", n)
                if mm:
                    frame_ids.append(int(mm.group(1)))

            frame_ids_set = set(frame_ids)
            csv_frames_set = set(frame_to_fen.keys())

            missing_imgs = sorted(csv_frames_set - frame_ids_set)
            extra_imgs = sorted(frame_ids_set - csv_frames_set)

            print(f"\n== {game} ==")
            print(f"  csv rows      : {len(rows)}")
            print(f"  tagged images : {len(tagged)}")

            if missing_imgs:
                print(f"  [WARN] Missing images for {len(missing_imgs)} frame(s). Examples: {missing_imgs[:10]}")
            if extra_imgs:
                print(f"  [WARN] Extra tagged images not in csv: {len(extra_imgs)}. Examples: {extra_imgs[:10]}")

            if frame_ids:
                print(f"  frame id range: {min(frame_ids)} .. {max(frame_ids)}")

            total_imgs += len(tagged)
            total_rows += len(rows)

    print("\n==== TOTAL ====")
    print(f"csv rows total      : {total_rows}")
    print(f"tagged images total : {total_imgs}")

if __name__ == "__main__":
    main()
