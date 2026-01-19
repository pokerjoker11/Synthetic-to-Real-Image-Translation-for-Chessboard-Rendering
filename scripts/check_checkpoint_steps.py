#!/usr/bin/env python3
"""
Check what step number each checkpoint is from.
"""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    ckpt_dir = REPO_ROOT / "checkpoints_clean"
    
    if not ckpt_dir.exists():
        print(f"[ERROR] Checkpoint directory not found: {ckpt_dir}")
        return 1
    
    pt_files = sorted(ckpt_dir.glob("*.pt"))
    
    if len(pt_files) == 0:
        print(f"[ERROR] No checkpoints found in {ckpt_dir}")
        return 1
    
    print(f"Found {len(pt_files)} checkpoints in {ckpt_dir}\n")
    print(f"{'Checkpoint':<40} {'Step':<10} {'Best Val L1':<15} {'Size (MB)':<12}")
    print("=" * 80)
    
    for pt_file in pt_files:
        try:
            ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
            step = ckpt.get("step", "unknown")
            best_val = ckpt.get("best_val", "unknown")
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            
            print(f"{pt_file.name:<40} {str(step):<10} {str(best_val):<15} {size_mb:.2f}")
        except Exception as e:
            print(f"{pt_file.name:<40} ERROR: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
