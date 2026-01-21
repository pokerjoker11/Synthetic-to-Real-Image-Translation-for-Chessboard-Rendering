#!/usr/bin/env python3
"""
Smoke test for piece-focused supervision features.

Tests:
1. Dataset loads masks correctly (or defaults to all-ones)
2. Weighted losses compute without errors
3. Piece discriminator works (if enabled)
4. Hinge loss works
5. R1 regularization computes (if enabled)
6. Validation sharpness metric works

Usage:
    python scripts/smoke_test_piece_supervision.py
    python scripts/smoke_test_piece_supervision.py --use_piece_mask --piece_mask_dir data/masks
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.pairs_dataset import PairedChessDataset
from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator, init_weights
from scripts.train_pix2pix import (
    make_weight_map, weighted_l1_loss, weighted_perceptual_loss,
    hinge_loss_D, hinge_loss_G, sample_piece_patches, compute_r1_penalty,
    sobel, to_grayscale, VGGPerceptualLoss, validate_l1
)


def test_dataset_with_masks(csv_path: Path, piece_mask_dir: Path = None, use_piece_mask: bool = False):
    """Test dataset loads masks correctly."""
    print(f"\n=== Testing Dataset with Masks ===")
    
    ds = PairedChessDataset(
        csv_path,
        repo_root=".",
        image_size=256,
        train=True,
        seed=123,
        load_size=256,
        piece_mask_dir=piece_mask_dir,
        use_piece_mask=use_piece_mask,
    )
    
    if len(ds) == 0:
        print(f"[ERROR] Dataset is empty: {csv_path}")
        return False
    
    # Test loading a batch
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    
    assert "A" in batch, "Missing 'A' (synthetic input)"
    assert "B" in batch, "Missing 'B' (real target)"
    assert "mask" in batch, "Missing 'mask'"
    
    A = batch["A"]
    B = batch["B"]
    mask = batch["mask"]
    
    print(f"  A shape: {A.shape} (expected: [N, 3, 256, 256])")
    print(f"  B shape: {B.shape} (expected: [N, 3, 256, 256])")
    print(f"  Mask shape: {mask.shape} (expected: [N, 1, 256, 256])")
    print(f"  Mask range: [{mask.min().item():.2f}, {mask.max().item():.2f}] (expected: [0.0, 1.0])")
    
    # Verify mask values are in {0, 1}
    mask_unique = torch.unique(mask)
    if not all(v in [0.0, 1.0] for v in mask_unique.tolist()):
        print(f"  [WARN] Mask has non-binary values: {mask_unique.tolist()}")
    
    print(f"  [OK] Dataset loaded successfully")
    return True


def test_weighted_losses():
    """Test weighted loss functions."""
    print(f"\n=== Testing Weighted Losses ===")
    
    device = torch.device("cpu")
    N, C, H, W = 2, 3, 256, 256
    
    # Create dummy data
    fake = torch.randn(N, C, H, W, device=device)
    real = torch.randn(N, C, H, W, device=device)
    mask = torch.randint(0, 2, (N, 1, H, W), dtype=torch.float32, device=device)
    
    # Test weight map
    weight_map = make_weight_map(mask, piece_weight=6.0, bg_weight=1.0)
    print(f"  Weight map shape: {weight_map.shape}")
    print(f"  Weight map range: [{weight_map.min().item():.2f}, {weight_map.max().item():.2f}]")
    
    # Test weighted L1
    w_l1 = weighted_l1_loss(fake, real, weight_map)
    print(f"  Weighted L1 loss: {w_l1.item():.4f}")
    
    # Test Sobel (for gradient loss)
    fake_gray = to_grayscale(fake)
    real_gray = to_grayscale(real)
    g_fake = sobel(fake_gray)
    g_real = sobel(real_gray)
    w_grad = weighted_l1_loss(g_fake, g_real, weight_map)
    print(f"  Weighted gradient loss: {w_grad.item():.4f}")
    
    print(f"  [OK] Weighted losses computed successfully")
    return True


def test_hinge_loss():
    """Test hinge loss functions."""
    print(f"\n=== Testing Hinge Loss ===")
    
    device = torch.device("cpu")
    N = 4
    H, W = 32, 32
    
    # Dummy discriminator outputs
    pred_real = torch.randn(N, 1, H, W, device=device) + 1.0  # Should be positive
    pred_fake = torch.randn(N, 1, H, W, device=device) - 1.0  # Should be negative
    
    # Test discriminator hinge loss
    d_real, d_fake = hinge_loss_D(pred_real, pred_fake)
    print(f"  D hinge (real): {d_real.item():.4f}")
    print(f"  D hinge (fake): {d_fake.item():.4f}")
    
    # Test generator hinge loss
    g_gan = hinge_loss_G(pred_fake)
    print(f"  G hinge: {g_gan.item():.4f}")
    
    print(f"  [OK] Hinge loss computed successfully")
    return True


def test_piece_discriminator():
    """Test piece patch sampling and discriminator."""
    print(f"\n=== Testing Piece Discriminator ===")
    
    device = torch.device("cpu")
    N, C, H, W = 2, 3, 256, 256
    crop_size = 96
    num_patches = 2
    
    # Create dummy data
    A = torch.randn(N, C, H, W, device=device)
    B = torch.randn(N, C, H, W, device=device)
    mask = torch.randint(0, 2, (N, 1, H, W), dtype=torch.float32, device=device)
    
    # Test patch sampling
    A_patches, B_patches = sample_piece_patches(A, B, mask, crop_size, num_patches)
    expected_patches = N * num_patches
    print(f"  A_patches shape: {A_patches.shape} (expected: [{expected_patches}, 3, {crop_size}, {crop_size}])")
    print(f"  B_patches shape: {B_patches.shape} (expected: [{expected_patches}, 3, {crop_size}, {crop_size}])")
    
    # Test piece discriminator forward
    D_piece = PatchDiscriminator().to(device)
    pred = D_piece(A_patches, B_patches)
    print(f"  D_piece output shape: {pred.shape}")
    
    print(f"  [OK] Piece discriminator works")
    return True


def test_r1_regularization():
    """Test R1 regularization computation."""
    print(f"\n=== Testing R1 Regularization ===")
    
    device = torch.device("cpu")
    N, C, H, W = 2, 3, 256, 256
    
    # Create dummy data
    D = PatchDiscriminator().to(device)
    A = torch.randn(N, C, H, W, device=device)
    B_real = torch.randn(N, C, H, W, device=device, requires_grad=True)
    
    # Forward pass
    pred = D(A, B_real)
    
    # Compute R1
    r1 = compute_r1_penalty(pred, B_real, gamma=10.0)
    print(f"  R1 penalty: {r1.item():.4f}")
    
    print(f"  [OK] R1 regularization computed successfully")
    return True


def test_full_forward_pass():
    """Test complete forward pass with all features enabled."""
    print(f"\n=== Testing Full Forward Pass ===")
    
    device = torch.device("cpu")
    N, C, H, W = 2, 3, 256, 256
    
    # Models
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    D_piece = PatchDiscriminator().to(device)
    init_weights(G)
    init_weights(D)
    init_weights(D_piece)
    
    # Create dummy batch
    A = torch.randn(N, C, H, W, device=device)
    B = torch.randn(N, C, H, W, device=device)
    mask = torch.randint(0, 2, (N, 1, H, W), dtype=torch.float32, device=device)
    
    # Weight map
    weight_map = make_weight_map(mask, piece_weight=6.0)
    weight_map_norm = weight_map / (weight_map.mean() + 1e-8)
    
    # Generator forward
    fake_B = G(A)
    
    # Discriminator (global)
    pred_real = D(A, B)
    pred_fake = D(A, fake_B)
    d_real, d_fake = hinge_loss_D(pred_real, pred_fake)
    d_loss = 0.5 * (d_real + d_fake)
    
    # Generator losses
    g_gan = hinge_loss_G(pred_fake)
    
    # Piece discriminator
    A_patches, B_patches = sample_piece_patches(A, B, mask, crop_size=96, num_patches=2)
    A_patches_fake, fake_B_patches = sample_piece_patches(A, fake_B, mask, crop_size=96, num_patches=2)
    pred_fake_piece = D_piece(A_patches_fake, fake_B_patches)
    g_gan_piece = hinge_loss_G(pred_fake_piece)
    g_gan = g_gan + 1.0 * g_gan_piece
    
    # Weighted losses
    g_l1 = weighted_l1_loss(fake_B, B, weight_map_norm) * 12.0
    
    fake_gray = to_grayscale(fake_B)
    real_gray = to_grayscale(B)
    g_fake = sobel(fake_gray)
    g_real = sobel(real_gray)
    g_grad = weighted_l1_loss(g_fake, g_real, weight_map_norm) * 60.0
    
    g_loss = g_gan + g_l1 + g_grad
    
    print(f"  Generator loss: {g_loss.item():.4f}")
    print(f"    GAN: {g_gan.item():.4f}")
    print(f"    L1: {g_l1.item():.4f}")
    print(f"    Grad: {g_grad.item():.4f}")
    print(f"  Discriminator loss: {d_loss.item():.4f}")
    
    print(f"  [OK] Full forward pass completed")
    return True


def test_validation_sharpness():
    """Test validation sharpness metric."""
    print(f"\n=== Testing Validation Sharpness Metric ===")
    
    device = torch.device("cpu")
    
    # Create dummy dataset
    train_csv = REPO_ROOT / "data/splits_rect/train_clean.csv"
    if not train_csv.exists():
        print(f"  [SKIP] CSV not found: {train_csv}")
        return True
    
    ds = PairedChessDataset(
        train_csv,
        repo_root=".",
        image_size=256,
        train=False,
        seed=123,
        load_size=256,
    )
    
    if len(ds) == 0:
        print(f"  [SKIP] Dataset is empty")
        return True
    
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    G = UNetGenerator().to(device)
    init_weights(G)
    G.eval()
    
    # Test validation (with and without sharpness)
    mean_l1_no_sharp, _ = validate_l1(G, dl, device, compute_sharpness=False)
    print(f"  Validation L1 (no sharpness): {mean_l1_no_sharp:.4f}")
    
    mean_l1_with_sharp, metrics = validate_l1(G, dl, device, compute_sharpness=True)
    print(f"  Validation L1 (with sharpness): {mean_l1_with_sharp:.4f}")
    if metrics:
        print(f"  Piece sharpness: {metrics.get('piece_sharpness', 'N/A')}")
        print(f"  BG sharpness: {metrics.get('bg_sharpness', 'N/A')}")
    
    print(f"  [OK] Validation sharpness metric works")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test for piece-focused supervision")
    parser.add_argument("--train_csv", type=str, default=None,
                       help="Training CSV for dataset tests")
    parser.add_argument("--piece_mask_dir", type=str, default=None,
                       help="Piece mask directory (optional)")
    parser.add_argument("--use_piece_mask", action="store_true",
                       help="Enable piece mask loading")
    
    args = parser.parse_args()
    
    train_csv = REPO_ROOT / args.train_csv
    mask_dir = REPO_ROOT / args.piece_mask_dir if args.piece_mask_dir else None
    
    print("=" * 60)
    print("PIECE-FOCUSED SUPERVISION SMOKE TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Dataset with masks
    if train_csv.exists():
        all_passed &= test_dataset_with_masks(train_csv, mask_dir, args.use_piece_mask)
    else:
        print(f"\n[SKIP] Train CSV not found: {train_csv}")
    
    # Test 2: Weighted losses
    all_passed &= test_weighted_losses()
    
    # Test 3: Hinge loss
    all_passed &= test_hinge_loss()
    
    # Test 4: Piece discriminator
    all_passed &= test_piece_discriminator()
    
    # Test 5: R1 regularization
    all_passed &= test_r1_regularization()
    
    # Test 6: Full forward pass
    all_passed &= test_full_forward_pass()
    
    # Test 7: Validation sharpness
    all_passed &= test_validation_sharpness()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[PASS] All smoke tests passed!")
    else:
        print("[FAIL] Some tests failed")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
