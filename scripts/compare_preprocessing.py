#!/usr/bin/env python3
"""
Compare preprocessing between training data and eval_api output.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load training synth
train_path = REPO_ROOT / "data" / "synth_v3_cropped" / "images" / "row000000_white_4fdc8e8c.png"
train_img = Image.open(train_path).convert("RGB")
print(f"Training image: {train_img.size}")

# Load test synthetic
test_path = REPO_ROOT / "results" / "synthetic.png"
if test_path.exists():
    test_img = Image.open(test_path).convert("RGB")
    print(f"Test image: {test_img.size}")
    
    # Resize test to match training for comparison
    if test_img.size != train_img.size:
        test_img = test_img.resize(train_img.size, Image.BICUBIC)
        print(f"Resized test to: {test_img.size}")
    
    # Compare statistics
    train_arr = np.array(train_img, dtype=np.float32)
    test_arr = np.array(test_img, dtype=np.float32)
    
    print(f"\nTraining stats:")
    print(f"  Mean: {train_arr.mean():.1f}")
    print(f"  Std: {train_arr.std():.1f}")
    print(f"  Min: {train_arr.min():.0f}, Max: {train_arr.max():.0f}")
    
    print(f"\nTest stats:")
    print(f"  Mean: {test_arr.mean():.1f}")
    print(f"  Std: {test_arr.std():.1f}")
    print(f"  Min: {test_arr.min():.0f}, Max: {test_arr.max():.0f}")
    
    # Check what happens after resize to 256x256 (model input)
    train_256 = train_img.resize((256, 256), Image.BICUBIC)
    test_256 = test_img.resize((256, 256), Image.BICUBIC)
    
    train_256_arr = np.array(train_256, dtype=np.float32)
    test_256_arr = np.array(test_256, dtype=np.float32)
    
    print(f"\nAfter resize to 256x256:")
    print(f"Training 256: mean={train_256_arr.mean():.1f}, std={train_256_arr.std():.1f}")
    print(f"Test 256:     mean={test_256_arr.mean():.1f}, std={test_256_arr.std():.1f}")
    
    # Check normalization (what model sees)
    train_t = TF.to_tensor(train_256)  # [0,1]
    train_norm = train_t * 2.0 - 1.0  # [-1,1]
    test_t = TF.to_tensor(test_256)
    test_norm = test_t * 2.0 - 1.0
    
    print(f"\nNormalized to [-1,1] (model input):")
    print(f"Training: mean={train_norm.mean():.3f}, std={train_norm.std():.3f}")
    print(f"Test:     mean={test_norm.mean():.3f}, std={test_norm.std():.3f}")
    
    # Create side-by-side comparison
    canvas = Image.new("RGB", (train_img.width * 2 + 10, train_img.height))
    canvas.paste(train_img, (0, 0))
    canvas.paste(test_img, (train_img.width + 10, 0))
    out_path = REPO_ROOT / "results" / "preprocessing_comparison.png"
    canvas.save(out_path)
    print(f"\nSaved comparison: {out_path}")
    
else:
    print(f"[ERROR] Test image not found: {test_path}")
    print("Run eval_api first to generate synthetic.png")
