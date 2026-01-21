#!/usr/bin/env python3
"""
Render synthetic chess boards for ChessReD2K pairs_ok.csv.

Reads data/chessred2k_rect_pov/pairs_ok.csv and renders synthetic images
for all entries, updating the synth_path column.

Usage:
    python scripts/render_chessred2k_synth.py
    python scripts/render_chessred2k_synth.py --csv data/chessred2k_rect_pov/pairs_ok.csv
"""

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
BLEND_FILE = ASSETS_DIR / "chess-set.blend"
BLENDER_SCRIPT = ASSETS_DIR / "chess_position_api_v2.py"

TMP_RENDERS_DIR = REPO_ROOT / "renders"
SYN_VIEW_NAME = "1_overhead.png"

# Training-time render settings (faster than eval)
RESOLUTION = 512
SAMPLES = 16

# Progress reporting
REPORT_EVERY = 5  # update progress every N items


def _find_blender() -> str:
    p = os.environ.get("BLENDER_PATH")
    if p and Path(p).exists():
        return p
    return "blender"


def _py_single_quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _render_one(fen: str, viewpoint: str) -> Path:
    """Render a single FEN position using Blender. Returns path to rendered image."""
    blender_bin = _find_blender()
    if not BLEND_FILE.exists():
        raise FileNotFoundError(f"Missing: {BLEND_FILE}")
    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {BLENDER_SCRIPT}")

    # Ensure Blender treats repo as CWD (so ./renders is repo-local)
    chdir_expr = f"import os; os.chdir('{_py_single_quote(str(REPO_ROOT))}')"

    # Clean old renders to avoid stale reads
    if TMP_RENDERS_DIR.exists():
        shutil.rmtree(TMP_RENDERS_DIR, ignore_errors=True)

    cmd = [
        blender_bin,
        str(BLEND_FILE),
        "--background",
        "--python-expr",
        chdir_expr,
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--fen", fen,
        "--view", viewpoint,
        "--resolution", str(RESOLUTION),
        "--samples", str(SAMPLES),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Blender failed.\nCMD: {' '.join(cmd)}\n\n{proc.stdout}")

    # Normal expected location
    p = TMP_RENDERS_DIR / SYN_VIEW_NAME
    if p.exists():
        return p

    # Fallback: parse "Saved: '...1_overhead.png'"
    m = re.search(r"Saved:\s*'([^']*1_overhead\.png)'", proc.stdout)
    if m:
        q = Path(m.group(1))
        if q.exists():
            return q

    raise FileNotFoundError(
        f"Render missing. Expected: {p}\nBlender output:\n{proc.stdout}"
    )


def main():
    parser = argparse.ArgumentParser(description="Render synthetic boards: all from pairs_top_pov.csv + 75 from pairs_ok.csv")
    parser.add_argument("--top-pov-csv", type=str, default="data/chessred2k_rect_pov/pairs_top_pov.csv",
                       help="CSV file with top POV images (all will be rendered)")
    parser.add_argument("--ok-csv", type=str, default="data/chessred2k_rect_pov/pairs_ok.csv",
                       help="CSV file with OK images (75 will be randomly selected)")
    parser.add_argument("--output-dir", type=str, default="data/chessred2k_rect_pov/synth_top_pov",
                       help="Output directory for synthetic images")
    parser.add_argument("--num-additional", type=int, default=75,
                       help="Number of additional images from pairs_ok.csv to include (default: 75)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip rendering if synthetic image already exists")
    
    args = parser.parse_args()
    
    top_pov_csv = REPO_ROOT / args.top_pov_csv
    ok_csv = REPO_ROOT / args.ok_csv
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not top_pov_csv.exists():
        raise SystemExit(f"[ERROR] Top POV CSV not found: {top_pov_csv}")
    if not ok_csv.exists():
        raise SystemExit(f"[ERROR] OK CSV not found: {ok_csv}")
    
    # Read top POV CSV (all rows)
    with open(top_pov_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        top_pov_rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"Loaded {len(top_pov_rows)} rows from {top_pov_csv}")
    
    # Read OK CSV (for random selection)
    with open(ok_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ok_rows = list(reader)
        # Ensure fieldnames match (use top_pov fieldnames as base)
        if reader.fieldnames != fieldnames:
            print(f"[WARN] Fieldnames differ between CSVs, using top_pov CSV fieldnames")
    
    print(f"Loaded {len(ok_rows)} rows from {ok_csv}")
    
    # Select random images from pairs_ok.csv
    num_additional = args.num_additional
    if len(ok_rows) < num_additional:
        print(f"[WARN] Only {len(ok_rows)} images in pairs_ok.csv, using all of them")
        selected_ok = ok_rows
    else:
        random.seed(42)  # For reproducibility
        selected_ok = random.sample(ok_rows, num_additional)
    
    # Combine: all top POV + selected OK images
    rows = top_pov_rows + selected_ok
    
    print(f"\nWill render:")
    print(f"  - {len(top_pov_rows)} top POV images (from {top_pov_csv.name})")
    print(f"  - {len(selected_ok)} additional images (from {ok_csv.name})")
    print(f"  - Total: {len(rows)} images\n")
    
    n = len(rows)
    rendered = 0
    skipped_existing = 0
    failed = 0
    
    start_t = time.perf_counter()
    
    # Process each row
    for idx, row in enumerate(rows):
        fen = row.get("fen", "").strip()
        pov = row.get("pov", "white").strip().lower()
        real_path = row.get("real_path", "")
        image_id = row.get("id", f"img_{idx}")
        
        # Normalize viewpoint
        if pov == "black":
            viewpoint = "black"
        else:
            viewpoint = "white"
        
        if not fen:
            print(f"  [{idx+1}/{n}] [SKIP] No FEN for {image_id}")
            failed += 1
            continue
        
        # Generate output filename based on image_id
        out_filename = f"{image_id}.png"
        synth_out = output_dir / out_filename
        synth_path_rel = f"data/chessred2k_rect_pov/synth_top_pov/{out_filename}"
        
        # Check if already exists
        if synth_out.exists() and args.skip_existing:
            print(f"  [{idx+1}/{n}] [SKIP] Already exists: {out_filename}")
            skipped_existing += 1
            # Update synth_path in row
            row["synth_path"] = synth_path_rel
            continue
        
        # Render
        try:
            print(f"  [{idx+1}/{n}] Rendering: {fen[:40]}... ({viewpoint})")
            synth_tmp = _render_one(fen, viewpoint)
            shutil.copyfile(synth_tmp, synth_out)
            row["synth_path"] = synth_path_rel
            rendered += 1
            print(f"    [OK] Saved: {out_filename}")
        except Exception as e:
            print(f"    [FAIL] Error: {e}")
            failed += 1
            # Leave synth_path empty or keep existing value
            if not row.get("synth_path"):
                row["synth_path"] = ""
            continue
        
        # Progress update
        if (idx + 1) % REPORT_EVERY == 0 or (idx + 1) == n:
            elapsed = time.perf_counter() - start_t
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            remaining = n - (idx + 1)
            eta_sec = remaining / rate if rate > 0 else float("inf")
            
            if eta_sec == float("inf"):
                eta_str = "??:??"
            else:
                eta_min = int(eta_sec // 60)
                eta_s = int(eta_sec % 60)
                eta_str = f"{eta_min:02d}:{eta_s:02d}"
            
            pct = 100.0 * (idx + 1) / n
            msg = (
                f"\r[PROGRESS] {idx+1}/{n} ({pct:5.1f}%) | "
                f"rendered={rendered} skipped={skipped_existing} failed={failed} | "
                f"{rate:5.2f} it/s | ETA {eta_str}"
            )
            print(msg, end="", flush=True)
    
    print()  # newline after carriage-return progress line
    
    # Write updated CSV
    output_csv = top_pov_csv.parent / "pairs_top_pov_subset_with_synth.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print("\n==== Summary ====")
    print(f"Total rows processed    : {n}")
    print(f"Rendered newly         : {rendered}")
    print(f"Skipped existing       : {skipped_existing}")
    print(f"Failed                 : {failed}")
    print(f"Synth images dir       : {output_dir}")
    print(f"Updated CSV            : {output_csv}")
    print(f"Render settings        : res={RESOLUTION}, samples={SAMPLES}")
    
    if rendered > 0:
        print(f"\n[OK] Rendered {rendered} synthetic images")
    if failed > 0:
        print(f"[WARN] {failed} rows failed to render")


if __name__ == "__main__":
    main()
