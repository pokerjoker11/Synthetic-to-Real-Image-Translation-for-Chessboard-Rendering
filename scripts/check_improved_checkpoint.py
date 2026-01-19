#!/usr/bin/env python3
"""Check improved checkpoint step."""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

ckpt_path = REPO_ROOT / "checkpoints_improved" / "latest.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", 0)
    print(f"Improved checkpoint step: {step}")
    print(f"Started from step 43k, so {step - 43000} steps with new weights")
else:
    print("No improved checkpoint found")
