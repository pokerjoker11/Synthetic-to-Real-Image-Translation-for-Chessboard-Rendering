#!/usr/bin/env python3
"""
Re-render synthetic images for all training positions using the v3 Blender script.

This creates a new synthetic dataset that matches the eval_api rendering style,
ensuring the model is trained on the same domain it will see at inference.

Usage:
    python scripts/rerender_synth_dataset.py --input_csv data/splits_rect/train.csv
    python scripts/rerender_synth_dataset.py --input_csv data/splits_rect/val.csv
    
Or render all at once:
    python scripts/rerender_synth_dataset.py --all
"""

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
BLEND_FILE = ASSETS_DIR / "chess-set.blend"
BLENDER_SCRIPT = ASSETS_DIR / "chess_position_api_v3.py"

# Output directory for new synthetic images
OUTPUT_DIR = REPO_ROOT / "data" / "synth_v3" / "images"
TEMP_RENDERS = REPO_ROOT / "renders"

# Render settings (balanced for quality and speed)
RENDER_RES = 512  # Resolution - will be resized to 256 during training anyway
RENDER_SAMPLES = 64  # Cycles samples - lower for faster batch rendering


def find_blender():
    """Find Blender executable"""
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    
    # Common paths
    common_paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        "/usr/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    for p in common_paths:
        if Path(p).exists():
            return p
    
    return "blender"  # Hope it's in PATH


def render_single(fen: str, viewpoint: str, output_path: Path, blender_bin: str) -> bool:
    """
    Render a single FEN position.
    Returns True on success, False on failure.
    """
    # Clean temp directory
    if TEMP_RENDERS.exists():
        shutil.rmtree(TEMP_RENDERS, ignore_errors=True)
    
    # Build Blender command
    chdir_expr = f"import os; os.chdir('{str(REPO_ROOT).replace(chr(92), '/')}')"
    
    cmd = [
        blender_bin,
        str(BLEND_FILE),
        "--background",
        "--python-expr", chdir_expr,
        "--python", str(BLENDER_SCRIPT),
        "--",
        "--fen", fen,
        "--view", viewpoint,
        "--resolution", str(RENDER_RES),
        "--samples", str(RENDER_SAMPLES),
        "--output", str(TEMP_RENDERS),
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,  # 5 minute timeout
        )
        
        if proc.returncode != 0:
            print(f"  [ERROR] Blender failed for {fen[:20]}...")
            print(proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout)
            return False
        
        # Find rendered image
        rendered = TEMP_RENDERS / "1_overhead.png"
        if not rendered.exists():
            print(f"  [ERROR] Render not found: {rendered}")
            return False
        
        # Move to output location
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rendered), str(output_path))
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout rendering {fen[:20]}...")
        return False
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return False


def generate_output_filename(row_idx: int, viewpoint: str, fen: str) -> str:
    """Generate unique filename for rendered image"""
    # Create short hash of FEN for uniqueness
    fen_hash = hashlib.md5(fen.encode()).hexdigest()[:8]
    return f"row{row_idx:06d}_{viewpoint}_{fen_hash}.png"


def process_csv(input_csv: Path, blender_bin: str) -> Path:
    """
    Process a CSV file and render all synthetic images.
    Returns path to the new CSV with updated synth paths.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_csv}")
    print(f"{'='*70}")
    
    if not input_csv.exists():
        print(f"[ERROR] CSV not found: {input_csv}")
        return None
    
    # Read input CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} rows to process")
    
    # Determine output CSV path
    output_csv = input_csv.parent / f"{input_csv.stem}_v3.csv"
    
    # Process each row
    new_rows = []
    success_count = 0
    fail_count = 0
    
    for idx, row in enumerate(rows):
        fen = row.get('fen', '')
        viewpoint = row.get('viewpoint', 'white')
        real_path = row.get('real', '')
        
        if not fen:
            print(f"  [SKIP] Row {idx}: No FEN")
            fail_count += 1
            continue
        
        # Generate output filename
        out_filename = generate_output_filename(idx, viewpoint, fen)
        out_path = OUTPUT_DIR / out_filename
        
        # Check if already rendered
        if out_path.exists():
            print(f"  [{idx+1}/{len(rows)}] Already exists: {out_filename}")
            synth_rel = f"data/synth_v3/images/{out_filename}"
        else:
            # Render
            print(f"  [{idx+1}/{len(rows)}] Rendering: {fen[:30]}... ({viewpoint})")
            
            if render_single(fen, viewpoint, out_path, blender_bin):
                print(f"    [OK] Saved: {out_filename}")
                success_count += 1
            else:
                print(f"    [FAIL] Failed")
                fail_count += 1
                continue
            
            synth_rel = f"data/synth_v3/images/{out_filename}"
        
        # Add to new rows
        new_row = {
            'real': real_path,
            'synth': synth_rel,
            'fen': fen,
            'viewpoint': viewpoint,
        }
        new_rows.append(new_row)
    
    # Write output CSV
    if new_rows:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['real', 'synth', 'fen', 'viewpoint'])
            writer.writeheader()
            writer.writerows(new_rows)
        
        print(f"\n[OK] Wrote {len(new_rows)} rows to {output_csv}")
    
    print(f"\nSummary: {success_count} rendered, {fail_count} failed, {len(new_rows)} total")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Re-render synthetic images for training")
    parser.add_argument('--input_csv', type=str, help="Input CSV file to process")
    parser.add_argument('--all', action='store_true', help="Process both train.csv and val.csv")
    parser.add_argument('--resolution', type=int, default=512, help="Render resolution")
    parser.add_argument('--samples', type=int, default=64, help="Cycles render samples")
    
    args = parser.parse_args()
    
    global RENDER_RES, RENDER_SAMPLES
    RENDER_RES = args.resolution
    RENDER_SAMPLES = args.samples
    
    # Find Blender
    blender_bin = find_blender()
    print(f"Using Blender: {blender_bin}")
    
    # Verify Blender script exists
    if not BLENDER_SCRIPT.exists():
        print(f"[ERROR] Blender script not found: {BLENDER_SCRIPT}")
        return 1
    
    if not BLEND_FILE.exists():
        print(f"[ERROR] Blend file not found: {BLEND_FILE}")
        return 1
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which CSVs to process
    csvs_to_process = []
    
    if args.all:
        csvs_to_process = [
            REPO_ROOT / "data" / "splits_rect" / "train.csv",
            REPO_ROOT / "data" / "splits_rect" / "val.csv",
        ]
    elif args.input_csv:
        csvs_to_process = [Path(args.input_csv)]
    else:
        print("Usage: Specify --input_csv or --all")
        return 1
    
    # Process each CSV
    start_time = time.time()
    
    output_csvs = []
    for csv_path in csvs_to_process:
        result = process_csv(csv_path, blender_bin)
        if result:
            output_csvs.append(result)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output CSVs:")
    for p in output_csvs:
        print(f"  {p}")
    
    print(f"\nNext steps:")
    print(f"  1. Review rendered images in {OUTPUT_DIR}")
    print(f"  2. Copy the _v3.csv files to replace train.csv and val.csv, or use them directly")
    print(f"  3. Start training: python train.py --train_csv <path_to_train_v3.csv>")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
