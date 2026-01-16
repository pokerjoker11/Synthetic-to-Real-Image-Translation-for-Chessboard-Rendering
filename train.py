#!/usr/bin/env python3
"""
Top-level training entrypoint for Project 3: Synthetic-to-Real Chess Translation.

This script wraps scripts/train_pix2pix.py via subprocess for robust cross-platform
execution (no PYTHONPATH or package issues). All arguments are passed through.

Usage:
    python train.py --train_csv data/splits_rect/train.csv --val_csv data/splits_rect/val.csv

For full argument list:
    python train.py --help
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_pix2pix.py"


def main() -> int:
    # Validate that the training script exists
    if not TRAIN_SCRIPT.exists():
        print(f"[ERROR] Training script not found: {TRAIN_SCRIPT}", file=sys.stderr)
        return 1

    # Build argument parser that mirrors scripts/train_pix2pix.py
    # This provides --help and validation at this level too
    ap = argparse.ArgumentParser(
        description="Train Pix2Pix model for synthetic-to-real chess translation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths - use actual repo structure as defaults
    ap.add_argument("--train_csv", type=str, default="data/splits_rect/train.csv",
                    help="Path to training CSV (columns: real, synth, fen, viewpoint)")
    ap.add_argument("--val_csv", type=str, default="data/splits_rect/val.csv",
                    help="Path to validation CSV")

    # Output directories
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints",
                    help="Directory for saving checkpoints")
    ap.add_argument("--samples_dir", type=str, default="results/train_samples",
                    help="Directory for saving sample images during training")

    # Device and batch
    ap.add_argument("--device", type=str, default="auto",
                    help="Device: auto|cpu|cuda")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)

    # Image sizes
    ap.add_argument("--image_size", type=int, default=256,
                    help="Final crop size for training")
    ap.add_argument("--load_size", type=int, default=286,
                    help="Resize to this before random crop (set ==image_size to disable crop)")

    # Training schedule
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)

    # Loss weights
    ap.add_argument("--lambda_l1", type=float, default=100.0,
                    help="Weight for L1 reconstruction loss")
    ap.add_argument("--lambda_grad", type=float, default=0.0,
                    help="Weight for Sobel edge loss (0 disables)")
    ap.add_argument("--grad_gray", action="store_true",
                    help="Compute Sobel loss on grayscale instead of RGB")

    # GAN loss type
    ap.add_argument("--gan_loss", type=str, default="bce", choices=["bce", "lsgan"])
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable automatic mixed precision (AMP) even on CUDA")

    # Logging
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=1000)

    # Resume
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from (if not set, auto-resumes from ckpt_dir/latest.pt if exists)")

    # Seed
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    # Validate input files exist
    train_csv_path = (REPO_ROOT / args.train_csv) if not Path(args.train_csv).is_absolute() else Path(args.train_csv)
    val_csv_path = (REPO_ROOT / args.val_csv) if not Path(args.val_csv).is_absolute() else Path(args.val_csv)

    if not train_csv_path.exists():
        print(f"[ERROR] Training CSV not found: {train_csv_path}", file=sys.stderr)
        print(f"        Expected format: columns 'real', 'synth', 'fen', 'viewpoint'", file=sys.stderr)
        print(f"        See README.md for data preparation instructions.", file=sys.stderr)
        return 1

    if not val_csv_path.exists():
        print(f"[ERROR] Validation CSV not found: {val_csv_path}", file=sys.stderr)
        return 1

    # Build command for subprocess
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_csv", args.train_csv,
        "--val_csv", args.val_csv,
        "--ckpt_dir", args.ckpt_dir,
        "--samples_dir", args.samples_dir,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--image_size", str(args.image_size),
        "--load_size", str(args.load_size),
        "--max_steps", str(args.max_steps),
        "--lr", str(args.lr),
        "--beta1", str(args.beta1),
        "--beta2", str(args.beta2),
        "--lambda_l1", str(args.lambda_l1),
        "--lambda_grad", str(args.lambda_grad),
        "--gan_loss", args.gan_loss,
        "--log_every", str(args.log_every),
        "--sample_every", str(args.sample_every),
        "--val_every", str(args.val_every),
        "--seed", str(args.seed),
    ]

    if args.grad_gray:
        cmd.append("--grad_gray")
    if args.no_amp:
        cmd.append("--no_amp")

    # Handle explicit resume checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume checkpoint not found: {resume_path}", file=sys.stderr)
            return 1
        cmd.extend(["--resume", str(resume_path)])
        print(f"[INFO] Resuming from: {resume_path}")

    print(f"[INFO] Running: {' '.join(cmd[:3])} ...")
    print(f"[INFO] Training CSV: {args.train_csv}")
    print(f"[INFO] Validation CSV: {args.val_csv}")
    print(f"[INFO] Checkpoint dir: {args.ckpt_dir}")
    print(f"[INFO] Max steps: {args.max_steps}")
    print()

    # Run the training script
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
