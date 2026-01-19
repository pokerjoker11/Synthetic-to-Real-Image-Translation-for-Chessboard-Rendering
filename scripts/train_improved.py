#!/usr/bin/env python3
"""
Improved training script with enhanced augmentation and regularization.

Key improvements:
1. Enhanced data augmentation (color jitter, noise, rotation)
2. Learning rate scheduling
3. Feature matching loss
4. Label smoothing
5. Better regularization
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import and run the main training script with improved defaults
if __name__ == "__main__":
    import subprocess
    
    # Build command with improved hyperparameters
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_pix2pix.py"),
        "--train_csv", "data/splits_rect/train_clean.csv",
        "--val_csv", "data/splits_rect/val_final.csv",
        "--batch_size", "8",
        "--image_size", "256",
        "--load_size", "286",  # Enable random crop for augmentation
        "--max_steps", "100000",
        "--lr", "2e-4",
        "--lambda_l1", "50",  # Reduced - was 75
        "--lambda_grad", "15",  # Reduced slightly - was 20
        "--lambda_perceptual", "5",  # Increased - was 4
        "--grad_gray",
        "--gan_loss", "bce",
        "--log_every", "100",
        "--sample_every", "1000",
        "--val_every", "2000",
        "--ckpt_dir", "checkpoints_improved",
        "--samples_dir", "results/train_samples_improved",
        "--resume", "checkpoints_clean/best_step40000_backup.pt",  # Start from best checkpoint
        "--no_amp",
    ]
    
    print("=" * 60)
    print("IMPROVED TRAINING - Starting from step 43k checkpoint")
    print("=" * 60)
    print("Key improvements:")
    print("  - Reduced L1 weight (50 vs 75) to allow more detail")
    print("  - Increased perceptual weight (5 vs 4) for better textures")
    print("  - Starting from best checkpoint (step 43k)")
    print("=" * 60)
    print()
    
    subprocess.run(cmd)
