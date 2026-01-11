# scripts/manual_mark_hands.py
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / pp).resolve()


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _make_side_by_side(real_img: Image.Image, synth_img: Image.Image, thumb: int, caption: str) -> Image.Image:
    # square thumbnails
    real_t = real_img.convert("RGB").resize((thumb, thumb), Image.BICUBIC)
    synth_t = synth_img.convert("RGB").resize((thumb, thumb), Image.BICUBIC)

    pad = 8
    cap_h = 52
    out = Image.new("RGB", (thumb * 2 + pad * 3, thumb + cap_h + pad * 2), (18, 18, 18))
    out.paste(real_t, (pad, cap_h + pad))
    out.paste(synth_t, (pad * 2 + thumb, cap_h + pad))

    d = ImageDraw.Draw(out)
    d.text((pad, pad), caption, fill=(240, 240, 240))
    d.text((pad, cap_h - 20), "REAL (mark hands/arms here)", fill=(220, 220, 220))
    d.text((pad * 2 + thumb, cap_h - 20), "SYNTH (reference)", fill=(220, 220, 220))
    return out


def cmd_make(args: argparse.Namespace) -> None:
    split_csv = _resolve_repo(args.split_csv)
    out_dir = _resolve_repo(args.out_dir)
    images_dir = out_dir / "images"
    drop_dir = out_dir / "DROP_HERE"
    keep_dir = out_dir / "KEEP_HERE"
    mapping_csv = out_dir / "mapping.csv"
    instructions = out_dir / "INSTRUCTIONS.txt"

    rows = _read_rows(split_csv)
    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    if args.limit > 0:
        rows = rows[: args.limit]

    images_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)
    keep_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["review_file", "real", "synth", "fen", "game", "frame", "viewpoint"])
        w.writeheader()

        for r in rows:
            real_p = _resolve_repo(r["real"])
            synth_p = _resolve_repo(r["synth"])

            if not real_p.exists() or not synth_p.exists():
                continue

            # stable review filename based on real stem (unique in this project)
            review_name = f"{Path(r['real']).stem}.png"
            out_path = images_dir / review_name

            if out_path.exists() and not args.overwrite:
                continue

            real_img = Image.open(real_p).convert("RGB")
            synth_img = Image.open(synth_p).convert("RGB")

            caption = f"{r.get('game','')} frame={r.get('frame','')} view={r.get('viewpoint','')}"
            sbs = _make_side_by_side(real_img, synth_img, thumb=args.thumb, caption=caption)
            sbs.save(out_path)

            w.writerow({
                "review_file": str((Path("images") / review_name).as_posix()),
                "real": r.get("real", ""),
                "synth": r.get("synth", ""),
                "fen": r.get("fen", ""),
                "game": r.get("game", ""),
                "frame": r.get("frame", ""),
                "viewpoint": r.get("viewpoint", ""),
            })
            written += 1

    instructions.write_text(
        "Manual hands/arms marking workflow\n"
        "===============================\n\n"
        "1) Open the folder:\n"
        f"   {images_dir}\n\n"
        "2) For each image that shows a hand/arm (or big occlusion), MOVE that PNG into:\n"
        f"   {drop_dir}\n\n"
        "   (Optional) If you want, you can move 'clean' ones into KEEP_HERE, but it's not required.\n\n"
        "3) After you're done, run:\n"
        f"   python scripts/manual_mark_hands.py collect --review_dir \"{out_dir}\" --out_list \"data/filters/manual_drop.txt\"\n\n"
        "Notes:\n"
        "- The review images are thumbnails (real|synth) to make scanning fast.\n"
        "- Your actual dataset is not modified by this step.\n",
        encoding="utf-8",
    )

    print("[OK] Review set created")
    print(f" - split csv : {split_csv}")
    print(f" - out dir   : {out_dir}")
    print(f" - images    : {images_dir}")
    print(f" - drop here : {drop_dir}")
    print(f" - mapping   : {mapping_csv}")
    print(f" - written   : {written}")
    print(f" - instructions: {instructions}")


def cmd_collect(args: argparse.Namespace) -> None:
    review_dir = _resolve_repo(args.review_dir)
    mapping_csv = review_dir / "mapping.csv"
    drop_dir = review_dir / "DROP_HERE"

    if not mapping_csv.exists():
        raise FileNotFoundError(f"Missing mapping.csv in {review_dir}")
    if not drop_dir.exists():
        raise FileNotFoundError(f"Missing DROP_HERE in {review_dir}")

    dropped_files = {p.name for p in drop_dir.glob("*.png")}

    # map review filename -> original real path
    review_to_real: Dict[str, str] = {}
    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            review_name = Path(r["review_file"]).name
            review_to_real[review_name] = r["real"]

    dropped_real = []
    unknown = []
    for fn in sorted(dropped_files):
        if fn in review_to_real:
            dropped_real.append(review_to_real[fn])
        else:
            unknown.append(fn)

    out_list = _resolve_repo(args.out_list)
    out_list.parent.mkdir(parents=True, exist_ok=True)

    # write as exact "real" paths (most precise), one per line
    with open(out_list, "w", encoding="utf-8") as f:
        for rp in dropped_real:
            f.write(rp.strip() + "\n")

    print("[OK] Collected manual drop list")
    print(f" - review dir : {review_dir}")
    print(f" - dropped png: {len(dropped_files)}")
    print(f" - mapped rows: {len(dropped_real)}")
    print(f" - out list   : {out_list}")
    if unknown:
        print(f"[WARN] {len(unknown)} dropped files were not found in mapping.csv (ignored). Examples: {unknown[:5]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_make = sub.add_parser("make", help="Create review thumbnails (real|synth) and mapping.csv")
    ap_make.add_argument("--split_csv", type=str, default="data/splits/train.csv")
    ap_make.add_argument("--out_dir", type=str, default="results/manual_review_train")
    ap_make.add_argument("--thumb", type=int, default=256)
    ap_make.add_argument("--limit", type=int, default=0, help="0 = all")
    ap_make.add_argument("--shuffle", action="store_true")
    ap_make.add_argument("--seed", type=int, default=123)
    ap_make.add_argument("--overwrite", action="store_true")
    ap_make.set_defaults(func=cmd_make)

    ap_col = sub.add_parser("collect", help="Collect files moved to DROP_HERE and write manual drop list")
    ap_col.add_argument("--review_dir", type=str, default="results/manual_review_train")
    ap_col.add_argument("--out_list", type=str, default="data/filters/manual_drop.txt")
    ap_col.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
