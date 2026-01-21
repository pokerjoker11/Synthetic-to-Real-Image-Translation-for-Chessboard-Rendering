# scripts/rewrite_synth_paths.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Rewrite the 'synth' column paths in a pairs CSV.")
    ap.add_argument("--in_csv", required=True, help="Input CSV (must contain 'synth' column)")
    ap.add_argument("--out_csv", required=True, help="Output CSV to write")
    ap.add_argument("--old_prefix", required=True, help="Old synth prefix to replace (e.g. data/synth_v3_cropped/images)")
    ap.add_argument("--new_prefix", required=True, help="New synth prefix (e.g. data/synth_v4_rect256/images)")
    ap.add_argument("--verify_exists", action="store_true", help="Verify new synth files exist (slower but safer)")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    oldp = args.old_prefix.replace("\\", "/").rstrip("/")
    newp = args.new_prefix.replace("\\", "/").rstrip("/")

    df = pd.read_csv(in_csv)
    if "synth" not in df.columns:
        raise SystemExit(f"[ERROR] CSV missing 'synth' column. Columns: {df.columns.tolist()}")

    def rewrite(p: str) -> str:
        s = str(p).replace("\\", "/")
        if s.startswith(oldp + "/") or s == oldp:
            s2 = newp + s[len(oldp):]
            return s2
        # also handle relative paths without leading prefix slashes differences
        if oldp in s:
            return s.replace(oldp, newp, 1)
        return s

    df["synth_old"] = df["synth"]
    df["synth"] = df["synth"].apply(rewrite)

    changed = (df["synth_old"] != df["synth"]).sum()
    print(f"[OK] rows={len(df)} changed={changed}")

    if args.verify_exists:
        missing = 0
        for p in df["synth"].tolist():
            if not Path(p).exists():
                missing += 1
        print(f"[VERIFY] missing_new_files={missing}")
        if missing > 0:
            print("[WARN] Some rewritten synth paths do not exist. Check your crop output folder.")
    df.drop(columns=["synth_old"]).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

if __name__ == "__main__":
    main()
