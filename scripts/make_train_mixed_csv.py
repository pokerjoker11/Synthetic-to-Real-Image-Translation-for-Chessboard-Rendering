#!/usr/bin/env python3
"""make_train_mixed_csv.py

Mix a base training CSV with an additional training CSV (e.g., ChessReD2K)
into a single shuffled training CSV.

All CSVs must use the canonical columns:
    real,synth,fen,viewpoint

Usage:
  python scripts/make_train_mixed_csv.py \
    --base_train data/splits_rect/train_clean.csv \
    --extra_train data/splits_rect/train_chessred2k.csv \
    --out_csv data/splits_rect/train_mixed.csv

Notes:
- By default, duplicates are removed by (real,synth) pair.
- Paths are normalized to forward slashes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLS = ["real", "synth", "fen", "viewpoint"]


def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _load_csv(rel_path: str) -> pd.DataFrame:
    p = REPO_ROOT / rel_path
    if not p.exists():
        raise SystemExit(f"[ERROR] CSV not found: {rel_path}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {rel_path} missing columns: {missing}")

    # Normalize and basic cleanup
    for c in ("real", "synth"):
        df[c] = df[c].astype(str).map(_norm_path)
    df["fen"] = df["fen"].astype(str).str.strip()
    df["viewpoint"] = df["viewpoint"].astype(str).str.strip().str.lower()

    # Drop obvious bad rows
    df = df[(df["fen"] != "") & (df["real"] != "") & (df["synth"] != "")]
    df = df[df["viewpoint"].isin(["white", "black"])]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Mix base train CSV with extra train CSV")
    ap.add_argument("--base_train", type=str, default="data/splits_rect/train_clean.csv")
    ap.add_argument("--extra_train", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="data/splits_rect/train_mixed.csv")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep_duplicates", action="store_true", help="Do not drop duplicate (real,synth) pairs")
    args = ap.parse_args()

    base_df = _load_csv(args.base_train)
    extra_df = _load_csv(args.extra_train)

    base_df["source"] = "course"
    extra_df["source"] = "extra"

    mixed = pd.concat([base_df, extra_df], ignore_index=True)

    before = len(mixed)
    if not args.keep_duplicates:
        mixed = mixed.drop_duplicates(subset=["real", "synth"], keep="first")
    after = len(mixed)

    mixed = mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_path = REPO_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write only canonical columns (plus source is optional but can be useful)
    mixed.to_csv(out_path, index=False)

    print("==== Mixed train CSV ====")
    print(f"Base : {args.base_train} ({len(base_df)} rows)")
    print(f"Extra: {args.extra_train} ({len(extra_df)} rows)")
    print(f"Out  : {args.out_csv} ({len(mixed)} rows)")
    if not args.keep_duplicates:
        print(f"De-dup: {before} -> {after}")
    print("========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
