#!/usr/bin/env python3
"""build_chessred2k_train_csv.py

Convert a ChessReD2K "pairs" CSV (produced by prepare_chessred2k_pairs.py / manual POV tooling)
into the canonical training CSV format used by this repo:

    real,synth,fen,viewpoint

Key details:
- ChessReD2K CSVs in this project historically used columns like real_path/synth_path,
  and sometimes also include expected_path/actual_path (because top-POV selection fixed paths).
- This script chooses a usable real image path in a robust way:
    1) actual_path (if present and exists)
    2) real_path (if exists)
    3) expected_path (if exists)
  Otherwise the row is skipped.
- Rows are skipped if synth_path is missing/empty or the synth file does not exist.

Typical usage after rendering ChessReD2K synth images:

  python scripts/build_chessred2k_train_csv.py \
    --pairs_csv data/chessred2k_rect_pov/pairs_top_pov_subset_with_synth.csv \
    --out_csv   data/splits_rect/train_chessred2k.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def _norm(s: str) -> str:
    return str(s).replace("\\", "/")


def _exists(rel: str) -> bool:
    p = REPO_ROOT / _norm(rel)
    return p.exists()


def _pick_real_path(row: dict) -> str:
    # Preference order matters.
    for key in ("actual_path", "real_path", "expected_path"):
        v = row.get(key, "")
        if isinstance(v, str) and v.strip():
            rel = _norm(v.strip())
            if _exists(rel):
                return rel
    # If none exists, return empty (caller will skip).
    return ""


def _pick_viewpoint(row: dict) -> str:
    for key in ("pov", "viewpoint"):
        v = row.get(key, "")
        if isinstance(v, str) and v.strip():
            t = v.strip().lower()
            if t in ("white", "black"):
                return t

    # Heuristic fallback from id naming.
    sid = str(row.get("id", "")).lower()
    if "bpov" in sid or "black" in sid:
        return "black"
    return "white"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert ChessReD2K pairs CSV into repo training CSV")
    ap.add_argument("--pairs_csv", type=str, required=True, help="Input pairs CSV with real/synth paths")
    ap.add_argument("--out_csv", type=str, default="data/splits_rect/train_chessred2k.csv", help="Output CSV")
    ap.add_argument("--require_exists", action="store_true", help="Skip rows where real/synth files are missing")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap (0 = no cap)")
    args = ap.parse_args()

    in_csv = REPO_ROOT / args.pairs_csv
    out_csv = REPO_ROOT / args.out_csv
    if not in_csv.exists():
        raise SystemExit(f"[ERROR] pairs_csv not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if len(df) == 0:
        raise SystemExit(f"[ERROR] Empty CSV: {in_csv}")

    out_rows = []
    skipped = {
        "no_fen": 0,
        "no_synth": 0,
        "missing_synth": 0,
        "no_real": 0,
        "missing_real": 0,
    }

    for _, r in df.iterrows():
        row = r.to_dict()

        fen = row.get("fen", "")
        if not isinstance(fen, str) or not fen.strip():
            skipped["no_fen"] += 1
            continue

        synth = row.get("synth_path", "")
        if not isinstance(synth, str) or not synth.strip() or synth.strip().lower() == "nan":
            skipped["no_synth"] += 1
            continue
        synth = _norm(synth.strip())

        real = _pick_real_path(row)
        if not real:
            skipped["no_real"] += 1
            continue

        if args.require_exists:
            if not _exists(synth):
                skipped["missing_synth"] += 1
                continue
            if not _exists(real):
                skipped["missing_real"] += 1
                continue

        viewpoint = _pick_viewpoint(row)

        out_rows.append(
            {
                "real": real,
                "synth": synth,
                "fen": fen.strip(),
                "viewpoint": viewpoint,
            }
        )

        if args.max_rows and len(out_rows) >= args.max_rows:
            break

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)

    print("==== ChessReD2K CSV Conversion ====")
    print(f"Input : {args.pairs_csv} ({len(df)} rows)")
    print(f"Output: {args.out_csv} ({len(out_df)} rows)")
    print("Skipped:")
    for k, v in skipped.items():
        print(f"  {k}: {v}")
    print("===================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
