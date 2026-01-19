#!/usr/bin/env python3
"""
Test the model on an actual training sample to verify it works on seen data.
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_api import generate_chessboard_image
import os

def main():
    # Set checkpoint path
    ckpt_dir = REPO_ROOT / "checkpoints_clean"
    best_pt = ckpt_dir / "best.pt"
    latest_pt = ckpt_dir / "latest.pt"
    
    if latest_pt.exists():
        os.environ['CKPT_PATH'] = str(latest_pt)
        print(f"Using checkpoint: {latest_pt.name}")
    elif best_pt.exists():
        os.environ['CKPT_PATH'] = str(best_pt)
        print(f"Using checkpoint: {best_pt.name}")
    else:
        print(f"[ERROR] No checkpoint found in {ckpt_dir}")
        return 1
    # Load training CSV
    train_csv = REPO_ROOT / "data" / "splits_rect" / "train_clean.csv"
    if not train_csv.exists():
        print(f"[ERROR] Training CSV not found: {train_csv}")
        return 1
    
    df = pd.read_csv(train_csv)
    if len(df) == 0:
        print("[ERROR] Training CSV is empty")
        return 1
    
    # Get first sample
    row = df.iloc[0]
    fen = row['fen']
    viewpoint = row.get('viewpoint', 'white')
    
    print(f"Testing on training sample:")
    print(f"  FEN: {fen}")
    print(f"  Viewpoint: {viewpoint}")
    print(f"  Real: {row['real']}")
    print(f"  Synth: {row['synth']}")
    
    # Generate with model
    try:
        generate_chessboard_image(fen, viewpoint)
        
        # Copy results
        output_dir = REPO_ROOT / "results" / "test_training_sample"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_dir = REPO_ROOT / "results"
        shutil.copy2(results_dir / "synthetic.png", output_dir / "synthetic.png")
        shutil.copy2(results_dir / "realistic.png", output_dir / "realistic.png")
        shutil.copy2(results_dir / "side_by_side.png", output_dir / "side_by_side.png")
        
        # Also copy the actual training images for comparison
        real_path = REPO_ROOT / row['real']
        synth_path = REPO_ROOT / row['synth']
        
        if real_path.exists():
            shutil.copy2(real_path, output_dir / "training_real.png")
        if synth_path.exists():
            shutil.copy2(synth_path, output_dir / "training_synth.png")
        
        print(f"\n[OK] Results saved to: {output_dir}")
        print("Compare:")
        print("  - synthetic.png (model input) vs training_synth.png (original training synth)")
        print("  - realistic.png (model output) vs training_real.png (original training real)")
        print("  - side_by_side.png (model input vs output)")
        
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
