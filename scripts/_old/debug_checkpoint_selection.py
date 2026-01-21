#!/usr/bin/env python3
"""Debug checkpoint selection logic."""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_api import DEFAULT_CKPT_DIR

ckpt_dir = DEFAULT_CKPT_DIR
print(f"Checking directory: {ckpt_dir}")
print(f"Directory exists: {ckpt_dir.exists()}\n")

plateau_checkpoints = []
all_checkpoints = []

for pt_file in sorted(ckpt_dir.glob("*.pt")):
    try:
        ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
        step = ckpt.get("step", 0)
        all_checkpoints.append((pt_file.name, step))
        
        if 35000 <= step <= 50000:
            plateau_checkpoints.append((pt_file, step))
            print(f"[PLATEAU] {pt_file.name}: step {step}")
        else:
            print(f"[OTHER]   {pt_file.name}: step {step}")
    except Exception as e:
        print(f"[ERROR]   {pt_file.name}: {e}")

print(f"\nFound {len(plateau_checkpoints)} plateau checkpoints")
if plateau_checkpoints:
    plateau_checkpoints.sort(key=lambda x: abs(x[1] - 43000))
    selected = plateau_checkpoints[0][0]
    print(f"Would select: {selected.name} (step {plateau_checkpoints[0][1]})")
else:
    print("No plateau checkpoints found!")
