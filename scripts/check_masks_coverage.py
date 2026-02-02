# scripts/check_masks_coverage.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["data/pairs/train.csv", "data/pairs/val.csv"])
    ap.add_argument("--mask_dir", default="data/masks_manual")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--fail", action="store_true", help="Exit with error code if anything is missing.")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    mask_dir = (repo / args.mask_dir).resolve()

    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir does not exist: {mask_dir}")

    missing = []
    checked = 0
    uniq_real = set()

    for csv_path in args.pairs:
        p = (repo / csv_path).resolve()
        df = pd.read_csv(p)
        if "real" not in df.columns:
            raise ValueError(f"{p} missing 'real' column")

        for rp in df["real"].astype(str):
            rp = rp.replace("\\", "/")
            real_path = Path(rp)
            if not real_path.is_absolute():
                real_path = (repo / real_path).resolve()
            if not real_path.exists():
                missing.append(("REAL_MISSING", str(real_path)))
                continue

            stem = real_path.stem
            uniq_real.add(stem)

    for stem in sorted(uniq_real):
        checked += 1
        mp = mask_dir / f"{stem}.png"
        if not mp.exists():
            missing.append(("MASK_MISSING", str(mp)))

    print("mask_dir:", mask_dir)
    print("unique real images referenced:", len(uniq_real))
    print("checked masks:", checked)
    print("missing items:", len(missing))
    if missing:
        print("\nExamples:")
        for t, s in missing[:25]:
            print(f"  {t}: {s}")

    if args.fail and missing:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
