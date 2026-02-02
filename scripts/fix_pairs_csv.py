# scripts/fix_pairs_csv.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


ROT_SUFFIX_RE = re.compile(r"_rot180(?=\.[^./\\]+$)")  # strip only right before extension


def _unique(seq: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in seq:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _norm_slashes(p: str) -> str:
    return str(p).strip().replace("\\", "/")


def _abs_or_repo_path(repo_root: Path, p: str) -> Path:
    """Interpret p as either absolute, or relative to repo_root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (repo_root / pp)


def _to_repo_relative_posix(repo_root: Path, p: Path) -> str:
    """If p is under repo_root, return relative POSIX path; else return POSIX of p."""
    try:
        rel = p.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        return p.as_posix()


def _candidates_real(path_str: str) -> List[str]:
    """
    Generate candidate fixes for REAL paths:
      - normalize slashes
      - images_rot180 -> images
      - remove _rot180 suffix from filename
      - combinations of the above
    """
    p = _norm_slashes(path_str)
    cands = [p]

    if "images_rot180" in p:
        cands.append(p.replace("images_rot180", "images"))

    # strip _rot180 in basename
    for x in list(cands):
        y = ROT_SUFFIX_RE.sub("", x)
        if y != x:
            cands.append(y)

    # combo: (images_rot180 -> images) then strip _rot180
    if "images_rot180" in p:
        x = p.replace("images_rot180", "images")
        y = ROT_SUFFIX_RE.sub("", x)
        if y != x:
            cands.append(y)

    return _unique(cands)


def _candidates_generic(path_str: str) -> List[str]:
    """Candidates for SYNTH paths: just normalize slashes."""
    p = _norm_slashes(path_str)
    return _unique([p])


def resolve_existing(repo_root: Path, path_str: str, is_real: bool) -> Optional[str]:
    cands = _candidates_real(path_str) if is_real else _candidates_generic(path_str)
    for c in cands:
        pp = _abs_or_repo_path(repo_root, c)
        if pp.exists():
            return _to_repo_relative_posix(repo_root, pp)
    return None


def fix_csv(repo_root: Path, in_csv: Path, out_csv: Path, drop_missing: bool) -> Tuple[int, int, int, int]:
    df = pd.read_csv(in_csv)
    if "real" not in df.columns or "synth" not in df.columns:
        raise ValueError(f"{in_csv} must have columns: real,synth (found: {list(df.columns)})")

    n0 = len(df)

    # Resolve + normalize
    df["real_fixed"] = df["real"].astype(str).map(lambda s: resolve_existing(repo_root, s, is_real=True))
    df["synth_fixed"] = df["synth"].astype(str).map(lambda s: resolve_existing(repo_root, s, is_real=False))

    miss_real = int(df["real_fixed"].isna().sum())
    miss_synth = int(df["synth_fixed"].isna().sum())

    if drop_missing:
        df2 = df[df["real_fixed"].notna() & df["synth_fixed"].notna()].copy()
    else:
        df2 = df.copy()

    # Overwrite with fixed paths (keep original columns order)
    df2["real"] = df2["real_fixed"].fillna(df2["real"].astype(str)).map(_norm_slashes)
    df2["synth"] = df2["synth_fixed"].fillna(df2["synth"].astype(str)).map(_norm_slashes)

    df2 = df2.drop(columns=["real_fixed", "synth_fixed"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_csv, index=False)

    kept = len(df2)
    dropped = n0 - kept if drop_missing else 0
    return n0, kept, dropped, (miss_real + miss_synth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=["data/pairs/train.csv", "data/pairs/val.csv"],
        help="Input CSV(s) to fix (default: data/pairs/train.csv data/pairs/val.csv)",
    )
    ap.add_argument(
        "--suffix",
        default="_fixed",
        help="Suffix appended before .csv (default: _fixed) => train_fixed.csv",
    )
    ap.add_argument(
        "--drop_missing",
        action="store_true",
        help="Drop rows whose real/synth paths still don't exist after fixing (recommended).",
    )
    args = ap.parse_args()

    repo_root = Path(".").resolve()

    print("Repo root:", repo_root)
    print("drop_missing:", bool(args.drop_missing))
    print()

    for inp in args.inputs:
        in_csv = Path(inp)
        if not in_csv.exists():
            raise FileNotFoundError(f"Missing input csv: {in_csv}")

        out_csv = in_csv.with_name(in_csv.stem + args.suffix + in_csv.suffix)
        n0, kept, dropped, miss_total = fix_csv(repo_root, in_csv, out_csv, drop_missing=bool(args.drop_missing))

        print(f"{in_csv.as_posix()} -> {out_csv.as_posix()}")
        print(f"  rows: {n0}  kept: {kept}  dropped: {dropped}")
        print(f"  unresolved cells (real+synth before drop): {miss_total}")
        print()


if __name__ == "__main__":
    main()
