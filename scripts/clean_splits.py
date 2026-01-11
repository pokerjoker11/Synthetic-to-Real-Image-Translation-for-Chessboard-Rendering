# scripts/clean_splits.py
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
from PIL import Image

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


def _write_rows(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _load_manual_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    s = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        s.add(t)
    return s


def _blur_score_var_laplacian(img_rgb: Image.Image, size: int = 256) -> float:
    """
    Variance of Laplacian on a resized grayscale image.
    Higher = sharper, Lower = blurrier.

    Implemented without OpenCV: uses a 4-neighbor Laplacian on the interior pixels.
    """
    img = img_rgb.convert("L").resize((size, size), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32)

    # interior laplacian (avoid borders)
    c = x[1:-1, 1:-1]
    up = x[:-2, 1:-1]
    dn = x[2:, 1:-1]
    lf = x[1:-1, :-2]
    rt = x[1:-1, 2:]

    lap = (-4.0 * c) + up + dn + lf + rt
    return float(lap.var())


def _filter_rows(
    rows: List[Dict[str, str]],
    manual_drop: Set[str],
    min_blur_score: float,
    blur_size: int,
    save_scores_csv: Path | None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    kept = []
    removed_manual = 0
    removed_blur = 0

    scores_out = []
    for r in rows:
        real_rel = r.get("real", "")
        real_path = _resolve_repo(real_rel)

        # manual match by exact real path OR basename
        basename = Path(real_rel).name
        if (real_rel in manual_drop) or (basename in manual_drop):
            removed_manual += 1
            continue

        if not real_path.exists():
            # missing real file => drop (shouldn't happen)
            removed_blur += 1
            continue

        img = Image.open(real_path).convert("RGB")
        score = _blur_score_var_laplacian(img, size=blur_size)

        scores_out.append({
            "real": real_rel,
            "score": f"{score:.6f}",
            "game": r.get("game", ""),
            "frame": r.get("frame", ""),
        })

        if min_blur_score > 0 and score < min_blur_score:
            removed_blur += 1
            continue

        kept.append(r)

    if save_scores_csv is not None:
        save_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_scores_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["real", "score", "game", "frame"])
            w.writeheader()
            w.writerows(scores_out)

    stats = {
        "kept": len(kept),
        "removed_manual": removed_manual,
        "removed_blur": removed_blur,
        "total_in": len(rows),
    }
    return kept, stats


def _print_quantiles(scores_csv: Path) -> None:
    if not scores_csv.exists():
        return
    scores = []
    with open(scores_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                scores.append(float(r["score"]))
            except:
                pass
    if not scores:
        return

    arr = np.array(scores, dtype=np.float32)
    qs = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    vals = np.quantile(arr, qs)
    print("Blur score quantiles (var(Laplacian)):")
    for q, v in zip(qs, vals):
        print(f"  q={q:>4.2f}: {v:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    ap.add_argument("--val_csv", type=str, default="data/splits/val.csv")
    ap.add_argument("--out_dir", type=str, default="data/splits_clean")

    ap.add_argument("--manual_drop", type=str, default="data/filters/manual_drop.txt",
                    help="Lines can be either exact 'real' paths OR basenames.")
    ap.add_argument("--min_blur_score", type=float, default=0.0,
                    help="0 disables blur filtering. Otherwise drop if score < threshold.")
    ap.add_argument("--blur_size", type=int, default=256, help="Resize used for blur scoring (consistent scale).")

    args = ap.parse_args()

    train_csv = _resolve_repo(args.train_csv)
    val_csv = _resolve_repo(args.val_csv)
    out_dir = _resolve_repo(args.out_dir)

    manual_drop_path = _resolve_repo(args.manual_drop)
    manual_drop = _load_manual_list(manual_drop_path)

    train_rows = _read_rows(train_csv)
    val_rows = _read_rows(val_csv)

    if not train_rows or not val_rows:
        raise SystemExit("[ERR] train/val csv empty")

    fieldnames = list(train_rows[0].keys())

    train_scores_csv = out_dir / "blur_scores_train.csv"
    val_scores_csv = out_dir / "blur_scores_val.csv"

    kept_train, st_train = _filter_rows(
        train_rows,
        manual_drop=manual_drop,
        min_blur_score=args.min_blur_score,
        blur_size=args.blur_size,
        save_scores_csv=train_scores_csv,
    )
    kept_val, st_val = _filter_rows(
        val_rows,
        manual_drop=manual_drop,
        min_blur_score=args.min_blur_score,
        blur_size=args.blur_size,
        save_scores_csv=val_scores_csv,
    )

    out_train = out_dir / "train_clean.csv"
    out_val = out_dir / "val_clean.csv"

    _write_rows(out_train, kept_train, fieldnames)
    _write_rows(out_val, kept_val, fieldnames)

    print("==== Clean Splits Summary ====")
    print(f"Manual drop list       : {manual_drop_path} ({len(manual_drop)} entries)")
    print(f"Blur filter            : min_blur_score={args.min_blur_score} (0 = disabled), blur_size={args.blur_size}")
    print("")
    print("[TRAIN]")
    print(f"  in   : {st_train['total_in']}")
    print(f"  kept : {st_train['kept']}")
    print(f"  drop manual : {st_train['removed_manual']}")
    print(f"  drop blur   : {st_train['removed_blur']}")
    print(f"  out  : {out_train}")
    print("")
    print("[VAL]")
    print(f"  in   : {st_val['total_in']}")
    print(f"  kept : {st_val['kept']}")
    print(f"  drop manual : {st_val['removed_manual']}")
    print(f"  drop blur   : {st_val['removed_blur']}")
    print(f"  out  : {out_val}")
    print("")
    print(f"Saved blur scores:")
    print(f" - {train_scores_csv}")
    print(f" - {val_scores_csv}")
    print("")
    _print_quantiles(train_scores_csv)
    print("")
    _print_quantiles(val_scores_csv)


if __name__ == "__main__":
    main()
