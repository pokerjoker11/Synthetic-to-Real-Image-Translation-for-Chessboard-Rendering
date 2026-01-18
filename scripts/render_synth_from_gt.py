import re
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REAL_GT = REPO / "data" / "real" / "gt.csv"
OUT_SYNTH_DIR = REPO / "data" / "synth" / "images"
OUT_SYNTH_GT = REPO / "data" / "synth" / "gt.csv"
OUT_REAL_WITH_SYNTH = REPO / "data" / "real" / "gt_with_synth.csv"

BLEND = REPO / "assets" / "chess-set.blend"
SCRIPT = REPO / "assets" / "chess_position_api_v2.py"


def fen_key(fen: str, view: str) -> str:
    # Stable ID so the same (fen,view) always maps to the same filename.
    h = hashlib.sha1(f"{fen}|{view}".encode("utf-8")).hexdigest()[:16]
    return f"{h}_{view}"


SAVED_OVERHEAD_RE = re.compile(r"Saved:\s*'([^']*1_overhead\.png)'", re.IGNORECASE)

def render_one(blender_exe: str, fen: str, view: str) -> Path:
    cmd = [
        blender_exe,
        str(BLEND),
        "--background",
        "--python",
        str(SCRIPT),
        "--",
        "--fen",
        fen,
        "--view",
        view,
    ]

    # Capture Blender output so we can parse the real output path
    p = subprocess.run(
        cmd,
        check=True,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log = (p.stdout or "") + "\n" + (p.stderr or "")

    m = SAVED_OVERHEAD_RE.search(log)
    if m:
        overhead = Path(m.group(1))
        if overhead.exists():
            return overhead

    # Fallback: your log strongly suggests this default path
    fallback = Path(r"C:\renders\1_overhead.png")
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Blender finished but overhead render wasn't found.\n"
        "Tried parsing output and also C:\\renders\\1_overhead.png.\n"
        "Tip: search manually for 1_overhead.png and tell me where it lands."
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--blender",
        required=True,
        help=r'Full path to blender.exe, e.g. "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"',
    )
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: render only first N unique FENs (0 = all)")
    args = ap.parse_args()

    if not REAL_GT.exists():
        raise FileNotFoundError(f"Missing {REAL_GT}. Run scripts/prepare_real.py first.")
    if not BLEND.exists():
        raise FileNotFoundError(f"Missing {BLEND}. Put chess-set.blend under assets/.")
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Missing {SCRIPT}. Put chess_position_api_v2.py under assets/.")

    OUT_SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "synth").mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(REAL_GT)
    if not {"fen", "view", "image_name"}.issubset(real.columns):
        raise RuntimeError(f"Unexpected columns in {REAL_GT}: {real.columns.tolist()}")

    combos = real[["fen", "view"]].drop_duplicates().reset_index(drop=True)
    if args.limit and args.limit > 0:
        combos = combos.iloc[: args.limit].copy()

    synth_rows = []
    key_to_synthname = {}

    for i, (fen, view) in enumerate(combos.itertuples(index=False), start=1):
        key = fen_key(fen, view)
        out_name = f"{key}.png"
        out_path = OUT_SYNTH_DIR / out_name

        if out_path.exists():
            key_to_synthname[(fen, view)] = out_name
            synth_rows.append([out_name, fen, view])
            print(f"[SKIP] {i}/{len(combos)} already exists: {out_name}")
            continue

        print(f"[RENDER] {i}/{len(combos)} view={view}  fen={fen[:40]}...")
        overhead = render_one(args.blender, fen, view)

        shutil.copy2(overhead, out_path)

        key_to_synthname[(fen, view)] = out_name
        synth_rows.append([out_name, fen, view])

    pd.DataFrame(synth_rows, columns=["image_name", "fen", "view"]).to_csv(OUT_SYNTH_GT, index=False)

    # Add synth_name column to real gt (safe for --limit smoke tests)
    map_df = pd.DataFrame(
        [(fen, view, name) for (fen, view), name in key_to_synthname.items()],
        columns=["fen", "view", "synth_name"],
    )

    real2 = real.merge(map_df, on=["fen", "view"], how="left")

    missing = int(real2["synth_name"].isna().sum())
    if missing:
        print(f"[WARN] {missing} real rows have no synth_name (expected when using --limit).")

    # Write a different filename when using --limit so you don't accidentally train on incomplete pairing
    if args.limit and args.limit > 0:
        out_real = REPO / "data" / "real" / f"gt_with_synth_limit{args.limit}.csv"
    else:
        out_real = OUT_REAL_WITH_SYNTH

    real2.to_csv(out_real, index=False)
    print(f"Wrote: {out_real}")


    print("\n==== Done ====")
    print(f"Synthetic images: {OUT_SYNTH_DIR}")
    print(f"Wrote: {OUT_SYNTH_GT}")
    print(f"Wrote: {OUT_REAL_WITH_SYNTH}")
    print(f"Unique (fen,view) rendered: {len(combos)}")


if __name__ == "__main__":
    main()
