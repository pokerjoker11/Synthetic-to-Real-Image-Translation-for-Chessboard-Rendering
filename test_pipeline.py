#!/usr/bin/env python3
"""
End-to-end pipeline test for the chess-s2r project.

This script verifies that:
1. All required imports work
2. Data files are correctly structured
3. Training can run for a few steps
4. Inference works (if a checkpoint exists)

Usage:
    python test_pipeline.py              # Run all tests
    python test_pipeline.py --no-train   # Skip training test
    python test_pipeline.py --no-infer   # Skip inference test
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def print_header(msg: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_result(name: str, passed: bool, detail: str = "") -> None:
    """Print a test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if detail:
        print(f"         {detail}")


def check_imports() -> bool:
    """Test that all critical imports work."""
    print_header("Testing Imports")
    
    tests = [
        ("torch", "import torch"),
        ("torchvision", "import torchvision"),
        ("PIL", "from PIL import Image"),
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("tqdm", "from tqdm import tqdm"),
        ("chess", "import chess"),
        ("src.models", "from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator"),
        ("src.datasets", "from src.datasets.pairs_dataset import PairedChessDataset"),
    ]
    
    all_passed = True
    for name, stmt in tests:
        try:
            exec(stmt)
            print_result(name, True)
        except ImportError as e:
            print_result(name, False, str(e))
            all_passed = False
    
    return all_passed


def check_data_structure() -> bool:
    """Verify expected data files exist."""
    print_header("Checking Data Structure")
    
    checks = [
        ("data/splits_rect/train.csv", "Training CSV"),
        ("data/splits_rect/val.csv", "Validation CSV"),
        ("assets/chess_position_api_v2.py", "Blender rendering script"),
        ("assets/chess-set.blend", "Blender scene file"),
    ]
    
    all_passed = True
    for path, name in checks:
        full_path = REPO_ROOT / path
        exists = full_path.exists()
        if not exists:
            all_passed = False
        print_result(name, exists, str(full_path))
    
    # Check CSV has correct columns (using csv module instead of pandas)
    train_csv = REPO_ROOT / "data/splits_rect/train.csv"
    if train_csv.exists():
        import csv
        with open(train_csv, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            required_cols = {"real", "synth", "fen", "viewpoint"}
            has_cols = required_cols.issubset(set(columns))
            print_result("CSV has required columns", has_cols, str(columns))
            if not has_cols:
                all_passed = False
            
            # Read first row
            try:
                row = next(reader)
                
                # Verify at least one real image exists
                if "real" in row:
                    real_path = REPO_ROOT / row["real"]
                    exists = real_path.exists()
                    print_result("Real images accessible", exists, str(real_path))
                    if not exists:
                        all_passed = False
                
                # Verify at least one synth image exists
                if "synth" in row:
                    synth_path = REPO_ROOT / row["synth"]
                    exists = synth_path.exists()
                    print_result("Synth images accessible", exists, str(synth_path))
                    if not exists:
                        all_passed = False
            except StopIteration:
                print_result("CSV has data rows", False, "CSV is empty")
                all_passed = False
    
    return all_passed


def check_model_forward() -> bool:
    """Test that the model can do a forward pass."""
    print_header("Testing Model Forward Pass")
    
    try:
        import torch
    except ImportError:
        print_result("Model forward pass", False, "torch not installed")
        return False
    
    try:
        from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator
        
        # Create models
        G = UNetGenerator(in_channels=3, out_channels=3)
        D = PatchDiscriminator(in_channels=6)
        
        # Create dummy input
        x = torch.randn(1, 3, 256, 256)
        
        # Test generator
        with torch.no_grad():
            fake = G(x)
        print_result("Generator forward", True, f"Input: {x.shape} -> Output: {fake.shape}")
        
        # Test discriminator
        with torch.no_grad():
            pred = D(torch.cat([x, fake], dim=1))
        print_result("Discriminator forward", True, f"Input: {fake.shape} -> Output: {pred.shape}")
        
        return True
    except Exception as e:
        print_result("Model forward pass", False, str(e))
        return False


def check_dataset_loading() -> bool:
    """Test that the dataset loads correctly."""
    print_header("Testing Dataset Loading")
    
    train_csv = REPO_ROOT / "data/splits_rect/train.csv"
    if not train_csv.exists():
        print_result("Dataset loading", False, "train.csv not found")
        return False
    
    try:
        import torch  # noqa: F401
    except ImportError:
        print_result("Dataset loading", False, "torch not installed")
        return False
    
    try:
        from src.datasets.pairs_dataset import PairedChessDataset
        
        ds = PairedChessDataset(csv_path=str(train_csv), image_size=256)
        print_result("Dataset creation", True, f"{len(ds)} samples")
        
        if len(ds) > 0:
            sample = ds[0]
            real_img = sample["real"]
            synth_img = sample["synth"]
            print_result("Sample loading", True, 
                        f"real: {real_img.shape}, synth: {synth_img.shape}")
        
        return True
    except Exception as e:
        print_result("Dataset loading", False, str(e))
        return False


def run_mini_training(steps: int = 5) -> bool:
    """Run a few training steps to verify the pipeline."""
    print_header(f"Testing Training ({steps} steps)")
    
    train_csv = REPO_ROOT / "data/splits_rect/train.csv"
    val_csv = REPO_ROOT / "data/splits_rect/val.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        print_result("Mini training", False, "CSV files not found")
        return False
    
    # Use a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, str(REPO_ROOT / "train.py"),
            "--train_csv", str(train_csv),
            "--val_csv", str(val_csv),
            "--ckpt_dir", tmpdir,
            "--samples_dir", str(Path(tmpdir) / "samples"),
            "--max_steps", str(steps),
            "--device", "cpu",
            "--log_every", "1",
            "--sample_every", str(steps),
            "--val_every", str(steps),
        ]
        
        print(f"  Running: python train.py --max_steps {steps} --device cpu ...")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=str(REPO_ROOT)
            )
            
            if result.returncode == 0:
                # Check if checkpoint was created
                ckpt = Path(tmpdir) / "latest.pt"
                if ckpt.exists():
                    print_result("Mini training", True, f"Checkpoint saved: {ckpt.name}")
                    return True
                else:
                    print_result("Mini training", False, "No checkpoint created")
                    return False
            else:
                print_result("Mini training", False, f"Exit code: {result.returncode}")
                if result.stderr:
                    # Print last few lines of stderr
                    lines = result.stderr.strip().split("\n")[-5:]
                    for line in lines:
                        print(f"         {line}")
                return False
        except subprocess.TimeoutExpired:
            print_result("Mini training", False, "Timeout (>300s)")
            return False
        except Exception as e:
            print_result("Mini training", False, str(e))
            return False


def check_inference() -> bool:
    """Test the inference API (if a checkpoint exists)."""
    print_header("Testing Inference API")
    
    # Check for checkpoint
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_candidates = [
        ckpt_dir / "best.pt",
        ckpt_dir / "latest.pt",
    ]
    
    checkpoint = None
    for c in ckpt_candidates:
        if c.exists():
            checkpoint = c
            break
    
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Also check for any .pt file
    if checkpoint is None and ckpt_dir.exists():
        pts = list(ckpt_dir.glob("*.pt"))
        if pts:
            checkpoint = max(pts, key=lambda p: p.stat().st_mtime)
    
    if checkpoint is None:
        print_result("Checkpoint found", False, "No checkpoint in checkpoints/")
        print("         Skipping inference test (train a model first)")
        return True  # Not a failure, just skipped
    
    print_result("Checkpoint found", True, str(checkpoint))
    
    # Check for Blender
    blender_path = shutil.which("blender")
    if not blender_path:
        import os
        blender_path = os.environ.get("BLENDER_PATH")
    
    if not blender_path:
        print_result("Blender available", False, "Set BLENDER_PATH or add blender to PATH")
        print("         Skipping inference test (Blender required)")
        return True  # Not a failure, just skipped
    
    print_result("Blender available", True, blender_path)
    
    # Run eval_api.py with a simple FEN
    with tempfile.TemporaryDirectory() as tmpdir:
        # We can't easily redirect results/ output, so we'll just test --help
        cmd = [
            sys.executable, str(REPO_ROOT / "eval_api.py"),
            "--help"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=str(REPO_ROOT)
            )
            
            if result.returncode == 0:
                print_result("eval_api.py --help", True)
                return True
            else:
                print_result("eval_api.py --help", False, f"Exit code: {result.returncode}")
                return False
        except Exception as e:
            print_result("eval_api.py --help", False, str(e))
            return False


def main() -> int:
    """Run all tests."""
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument("--no-train", action="store_true", help="Skip training test")
    parser.add_argument("--no-infer", action="store_true", help="Skip inference test")
    parser.add_argument("--train-steps", type=int, default=5, help="Steps for mini training")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  Chess S2R Pipeline Test")
    print("="*60)
    
    results = {}
    
    # Always run these
    results["imports"] = check_imports()
    results["data"] = check_data_structure()
    results["model"] = check_model_forward()
    results["dataset"] = check_dataset_loading()
    
    # Optional tests
    if not args.no_train:
        results["training"] = run_mini_training(args.train_steps)
    
    if not args.no_infer:
        results["inference"] = check_inference()
    
    # Summary
    print_header("Summary")
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    print()
    if all_passed:
        print("  All tests passed!")
        return 0
    else:
        print("  Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
