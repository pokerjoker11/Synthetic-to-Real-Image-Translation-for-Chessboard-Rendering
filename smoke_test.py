#!/usr/bin/env python3
"""
Smoke test to verify the installation and basic functionality.

This script:
1. Checks all imports work
2. Verifies train.py --help works
3. Verifies eval_api.py --help works
4. (Optional) Runs a 2-step training if data exists

Usage:
    python smoke_test.py
    python smoke_test.py --with-training  # Also run 2 training steps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def check_imports() -> bool:
    """Check that all required imports work."""
    print("[1/4] Checking imports...")
    
    try:
        import torch
        print(f"      torch: {torch.__version__}")
    except ImportError as e:
        print(f"      [FAIL] torch: {e}")
        return False

    try:
        import torchvision
        print(f"      torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"      [FAIL] torchvision: {e}")
        return False

    try:
        import numpy
        print(f"      numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"      [FAIL] numpy: {e}")
        return False

    try:
        import PIL
        print(f"      pillow: {PIL.__version__}")
    except ImportError as e:
        print(f"      [FAIL] pillow: {e}")
        return False

    try:
        from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator
        from src.datasets.pairs_dataset import PairedChessDataset
        print("      src.models: OK")
        print("      src.datasets: OK")
    except ImportError as e:
        print(f"      [FAIL] src modules: {e}")
        return False

    print("      [OK] All imports successful")
    return True


def check_train_help() -> bool:
    """Check that train.py --help works."""
    print("[2/4] Checking train.py --help...")
    
    train_py = REPO_ROOT / "train.py"
    if not train_py.exists():
        print(f"      [FAIL] {train_py} not found")
        return False

    result = subprocess.run(
        [sys.executable, str(train_py), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if result.returncode != 0:
        print(f"      [FAIL] Exit code {result.returncode}")
        print(f"      stderr: {result.stderr[:500]}")
        return False

    if "--train_csv" not in result.stdout:
        print("      [FAIL] --train_csv not in help output")
        return False

    if "--mask_dir" not in result.stdout:
        print("      [FAIL] --mask_dir not in help output")
        return False

    print("      [OK] train.py --help works")
    return True


def check_eval_help() -> bool:
    """Check that eval_api.py --help works."""
    print("[3/4] Checking eval_api.py --help...")
    
    eval_py = REPO_ROOT / "eval_api.py"
    if not eval_py.exists():
        print(f"      [FAIL] {eval_py} not found")
        return False

    result = subprocess.run(
        [sys.executable, str(eval_py), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if result.returncode != 0:
        print(f"      [FAIL] Exit code {result.returncode}")
        print(f"      stderr: {result.stderr[:500]}")
        return False

    if "--fen" not in result.stdout:
        print("      [FAIL] --fen not in help output")
        return False

    print("      [OK] eval_api.py --help works")
    return True


def check_model_forward() -> bool:
    """Check that model forward pass works (no data needed)."""
    print("[4/4] Checking model forward pass...")

    try:
        import torch
        from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator, init_weights

        device = torch.device("cpu")
        G = UNetGenerator().to(device)
        D = PatchDiscriminator().to(device)
        init_weights(G)
        init_weights(D)

        # Dummy input
        x = torch.randn(1, 3, 256, 256, device=device)

        with torch.no_grad():
            fake = G(x)
            pred = D(x, fake)

        assert fake.shape == (1, 3, 256, 256), f"Generator output shape wrong: {fake.shape}"
        assert pred.shape[0] == 1, f"Discriminator output shape wrong: {pred.shape}"

        print(f"      G(x) shape: {tuple(fake.shape)}")
        print(f"      D(x, G(x)) shape: {tuple(pred.shape)}")
        print("      [OK] Model forward pass works")
        return True

    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

def run_mini_training() -> bool:
    """Run 2 training steps if data exists."""
    print("[OPTIONAL] Running 2-step training test...")

    train_csv = REPO_ROOT / "data" / "pairs" / "train.csv"
    val_csv = REPO_ROOT / "data" / "pairs" / "val.csv"
    masks_dir = REPO_ROOT / "data" / "masks_manual"

    if not train_csv.exists():
        print(f"      [SKIP] {train_csv} not found")
        return True
    if not val_csv.exists():
        print(f"      [SKIP] {val_csv} not found")
        return True
    if not masks_dir.exists():
        print(f"      [SKIP] {masks_dir} not found (masks required for training)")
        return True

    ckpt_dir = "checkpoints_smoke_test"

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "train.py"),
            "--train_csv", str(train_csv),
            "--val_csv", str(val_csv),
            "--max_steps", "2",
            "--device", "cpu",
            "--num_workers", "0",
            "--log_every", "1",
            "--sample_every", "2",
            "--sample_nocrop",
            "--val_every", "2",
            "--ckpt_dir", ckpt_dir,
            "--mask_dir", str(masks_dir),
            "--lambda_piece", "5",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )

    if result.returncode != 0:
        print(f"      [FAIL] Exit code {result.returncode}")
        print(f"      stdout: {result.stdout[-1000:]}")
        print(f"      stderr: {result.stderr[-500:]}")
        return False

    # Confirm it actually wrote a checkpoint
    ckpt_path = REPO_ROOT / ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        print(f"      [FAIL] training finished but checkpoint not found: {ckpt_path}")
        return False

    print("      [OK] 2-step training completed and checkpoint written")

    # Cleanup
    ckpt_dir_path = REPO_ROOT / ckpt_dir
    if ckpt_dir_path.exists():
        import shutil
        shutil.rmtree(ckpt_dir_path, ignore_errors=True)

    return True

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-training", action="store_true",
                    help="Also run 2 training steps (requires data)")
    args = ap.parse_args()

    print("=" * 60)
    print("Smoke Test: Project 3 - Synthetic-to-Real Chess")
    print("=" * 60)
    print()

    all_passed = True

    if not check_imports():
        all_passed = False

    if not check_train_help():
        all_passed = False

    if not check_eval_help():
        all_passed = False

    if not check_model_forward():
        all_passed = False

    if args.with_training:
        if not run_mini_training():
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        return 0
    else:
        print("RESULT: SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
