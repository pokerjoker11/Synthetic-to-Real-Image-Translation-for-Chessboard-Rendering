#!/usr/bin/env python3
"""
Train Pix2Pix model for chessboard image translation.

Example with piece-focused supervision:
    python scripts/train_pix2pix.py \
        --train_csv data/splits_rect/train_clean.csv \
        --val_csv data/splits_rect/val_final.csv \
        --gan_loss hinge \
        --use_piece_mask \
        --piece_mask_dir data/masks \
        --piece_weight 6.0 \
        --use_piece_D \
        --piece_crop_size 96 \
        --piece_patches_per_image 2 \
        --r1_gamma 10.0 \
        --lambda_l1 12 \
        --lambda_grad 60 \
        --lambda_perceptual 0.15
"""

# scripts/train_pix2pix.py
from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.pairs_dataset import PairedChessDataset
from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator, init_weights


# ----------------------------
# Image helpers
# ----------------------------
def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) in [-1,1] -> uint8 HxWx3
    """
    x = x.detach().float().cpu()
    x = (x + 1.0) * 0.5  # [0,1]
    x = x.clamp(0, 1)
    x = (x * 255.0).byte()
    return x.permute(1, 2, 0).numpy()


@torch.no_grad()
def save_samples(G: nn.Module, batch: dict, device: torch.device, out_path: Path) -> None:
    """
    Save side-by-side: [A | G(A) | B]
    """
    was_training = G.training
    G.eval()

    A = batch["A"].to(device)
    B = batch["B"].to(device)
    fake_B = G(A)

    a = tensor_to_uint8(A[0])
    f = tensor_to_uint8(fake_B[0])
    b = tensor_to_uint8(B[0])

    H, W, _ = a.shape
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(Image.fromarray(a), (0, 0))
    canvas.paste(Image.fromarray(f), (W, 0))
    canvas.paste(Image.fromarray(b), (W * 2, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    if was_training:
        G.train()


# ----------------------------
# VGG Perceptual Loss
# ----------------------------
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Compares features at multiple layers for texture/detail preservation.
    """
    def __init__(self, layers=None, weights=None):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        
        # Default layers: conv1_2, conv2_2, conv3_2, conv4_2
        if layers is None:
            layers = [2, 7, 12, 21]
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
        
        self.layers = layers
        self.weights = weights
        
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        
        prev = 0
        for layer_idx in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, layer_idx + 1)]))
            prev = layer_idx + 1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Convert from [-1,1] to ImageNet normalized"""
        x = (x + 1) / 2  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        """Return list of VGG feature maps at configured layers.

        Args:
            x: (N,3,H,W) in [-1,1]

        Returns:
            feats: list of tensors, one per configured layer
        """
        x = self.normalize(x)
        feats = []
        h = x
        for slice_net in self.slices:
            h = slice_net(h)
            feats.append(h)
        return feats

    def forward(self, fake, real):
        fake_feats = self.extract_features(fake)
        real_feats = self.extract_features(real)
        loss = 0.0
        for w, ff, rf in zip(self.weights, fake_feats, real_feats):
            loss += w * F.l1_loss(ff, rf)
        return loss


# ----------------------------
# Sobel / gradient loss helper
# ----------------------------
def sobel(
    x: torch.Tensor,
    *,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Sobel gradients for a batch of images.

    x: (N,C,H,W)
    Returns:
      - magnitude: (N,C,H,W)  if return_components=False
      - (gx, gy): both (N,C,H,W) if return_components=True

    Notes:
      - Uses grouped conv so each channel is filtered independently.
      - Kernel is normalized by 8.0 (common).
    """
    if x.dim() != 4:
        raise ValueError(f"sobel() expects NCHW, got shape={tuple(x.shape)}")

    _, C, _, _ = x.shape
    dtype = x.dtype
    device = x.device

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ) / 8.0

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=dtype,
        device=device,
    ) / 8.0

    # (2,1,3,3) then repeat per-channel => (2*C,1,3,3)
    weight = torch.stack([kx, ky], dim=0).unsqueeze(1).repeat(C, 1, 1, 1)

    g = F.conv2d(x, weight, bias=None, stride=1, padding=1, groups=C)  # (N,2C,H,W)
    gx = g[:, 0:C, :, :]
    gy = g[:, C:2 * C, :, :]

    if return_components:
        return gx, gy

    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,3,H,W) -> (N,1,H,W) using luminance weights.
    If C != 3, falls back to mean over channels.
    """
    if x.dim() != 4:
        raise ValueError(f"to_grayscale expects NCHW, got shape={tuple(x.shape)}")

    if x.size(1) != 3:
        return x.mean(dim=1, keepdim=True)

    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ----------------------------
# Piece mask weighted losses
# ----------------------------
def make_weight_map(mask: torch.Tensor, piece_weight: float, bg_weight: float = 1.0) -> torch.Tensor:
    """
    Create weight map from mask for loss weighting.
    
    Args:
        mask: [N, 1, H, W] in {0, 1}, where 1 = piece region
        piece_weight: weight multiplier for piece pixels
        bg_weight: weight multiplier for background pixels (default 1.0)
    
    Returns:
        weight: [N, 1, H, W] where weight = bg_weight + (piece_weight - bg_weight) * mask
    """
    return bg_weight + (piece_weight - bg_weight) * mask


def mask_boundary_band(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Approximate boundary band around piece mask using morphological gradient.

    mask: (N,1,H,W) float in {0,1}

    Returns:
        band: (N,1,H,W) float in [0,1] where 1 indicates pixels near the mask boundary
    """
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    pad = k // 2
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)
    band = (dil - ero).clamp(0.0, 1.0)
    return band


def weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Weighted L1 loss.
    
    Args:
        pred: [N, C, H, W]
        target: [N, C, H, W]
        weight: [N, 1, H, W] or [N, C, H, W]
    
    Returns:
        weighted mean absolute error
    """
    diff = torch.abs(pred - target)
    # Broadcast weight to match pred channels if needed
    if weight.shape[1] == 1 and pred.shape[1] > 1:
        weight = weight.repeat(1, pred.shape[1], 1, 1)
    weighted_diff = diff * weight
    return weighted_diff.mean()


def weighted_perceptual_loss(vgg_loss_fn: VGGPerceptualLoss, pred: torch.Tensor, target: torch.Tensor,
                            weight: torch.Tensor) -> torch.Tensor:
    """Spatially-weighted perceptual loss using VGG features.

    We downsample the (H,W) weight map to each VGG feature map resolution and
    apply it elementwise before averaging. This makes the perceptual loss
    actually care more about piece regions (and not just multiply by ~1).

    Args:
        vgg_loss_fn: VGGPerceptualLoss module
        pred:   (N,3,H,W) in [-1,1]
        target: (N,3,H,W) in [-1,1]
        weight: (N,1,H,W) weight map (ideally normalized to mean ~1)

    Returns:
        weighted perceptual loss (scalar)
    """
    pred_feats = vgg_loss_fn.extract_features(pred)
    tgt_feats = vgg_loss_fn.extract_features(target)

    loss = 0.0
    for w_layer, pf, tf in zip(vgg_loss_fn.weights, pred_feats, tgt_feats):
        # Downsample weight map to this feature resolution
        wm = F.interpolate(weight, size=pf.shape[-2:], mode='bilinear', align_corners=False)
        # Keep magnitudes stable: normalize per-layer to mean ~1
        wm = wm / (wm.mean() + 1e-8)
        # Broadcast to channels
        if wm.shape[1] == 1 and pf.shape[1] > 1:
            wm = wm.repeat(1, pf.shape[1], 1, 1)
        diff = torch.abs(pf - tf) * wm
        loss = loss + w_layer * diff.mean()
    return loss


# ----------------------------
# Hinge loss helpers
# ----------------------------
def hinge_loss_D(real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hinge loss for discriminator.
    
    Args:
        real: D(real) predictions
        fake: D(fake) predictions (detached)
    
    Returns:
        (real_loss, fake_loss)
    """
    real_loss = F.relu(1.0 - real).mean()
    fake_loss = F.relu(1.0 + fake).mean()
    return real_loss, fake_loss


def hinge_loss_G(fake: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for generator.
    
    Args:
        fake: D(fake) predictions
    
    Returns:
        generator loss (negative mean of fake predictions)
    """
    return -fake.mean()


# ----------------------------
# Patch sampling for piece discriminator
# ----------------------------

# We intentionally keep patch extraction deterministic within a step by
# sampling patch coordinates once and reusing them for A/B/fake_B.

def sample_piece_patch_coords(mask: torch.Tensor, crop_size: int, num_patches: int):
    # mask: [N, 1, H, W] in {0,1}
    N, _, H, W = mask.shape
    half = crop_size // 2
    coords = []  # list of (n, top, left)
    for n in range(N):
        m = mask[n, 0]
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        for _ in range(num_patches):
            if ys.numel() > 0:
                j = torch.randint(0, ys.numel(), (1,), device=mask.device).item()
                cy = int(ys[j].item())
                cx = int(xs[j].item())
            else:
                cy = int(torch.randint(0, H, (1,), device=mask.device).item())
                cx = int(torch.randint(0, W, (1,), device=mask.device).item())
            top = max(0, min(cy - half, H - crop_size))
            left = max(0, min(cx - half, W - crop_size))
            coords.append((n, top, left))
    return coords


def extract_patches(X: torch.Tensor, coords, crop_size: int) -> torch.Tensor:
    # X: [N, C, H, W]
    if not coords:
        return X.new_empty((0, X.shape[1], crop_size, crop_size))
    patches = [X[n:n+1, :, top:top+crop_size, left:left+crop_size] for (n, top, left) in coords]
    return torch.cat(patches, dim=0)


def sample_piece_patches(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor,
                         crop_size: int, num_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Backward-compatible helper (coords not returned)
    coords = sample_piece_patch_coords(mask, crop_size, num_patches)
    return extract_patches(A, coords, crop_size), extract_patches(B, coords, crop_size)


def sample_piece_patches_with_coords(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor,
                                     crop_size: int, num_patches: int):
    coords = sample_piece_patch_coords(mask, crop_size, num_patches)
    A_p = extract_patches(A, coords, crop_size)
    B_p = extract_patches(B, coords, crop_size)
    return A_p, B_p, coords



# ----------------------------
# R1 regularization
# ----------------------------
def compute_r1_penalty(d_output: torch.Tensor, real_input: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute R1 regularization penalty for discriminator.
    
    Args:
        d_output: D(real) output [N, ...]
        real_input: Real input tensor [N, C, H, W]
        gamma: R1 regularization weight
    
    Returns:
        R1 penalty: 0.5 * gamma * ||grad||^2
    """
    # Compute gradients
    grad = torch.autograd.grad(
        outputs=d_output.sum(),
        inputs=real_input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # R1 = 0.5 * gamma * ||grad||^2
    grad_sq = grad.pow(2).view(real_input.shape[0], -1).sum(1).mean()
    return 0.5 * gamma * grad_sq


# ----------------------------
# Checkpointing
# ----------------------------
def save_checkpoint(path: Path, step: int, best_val: float, G, D, optG, optD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_val": best_val,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, G, D, optG, optD, strict: bool = True) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    G.load_state_dict(ckpt["G"], strict=strict)
    
    # Handle discriminator state dict mismatch (old Sequential vs new layer-based architecture)
    d_state = ckpt.get("D", {})
    d_loaded = False
    try:
        D.load_state_dict(d_state, strict=strict)
        d_loaded = True
    except RuntimeError as e:
        # If strict loading fails, try loading only compatible weights
        if not strict:
            raise
        print(f"[WARN] Discriminator state dict mismatch: {e}")
        print("[INFO] Loading discriminator with strict=False (only compatible weights)")
        try:
            D.load_state_dict(d_state, strict=False)
            d_loaded = True
            print("[INFO] Successfully loaded compatible discriminator weights")
        except Exception as e2:
            print(f"[WARN] Could not load discriminator weights: {e2}")
            print("[INFO] Using randomly initialized discriminator")
            d_loaded = False
    
    # Load generator optimizer (should be compatible)
    try:
        optG.load_state_dict(ckpt.get("optG", {}))
    except Exception as e:
        print(f"[WARN] Could not load generator optimizer state: {e}")
        print("[INFO] Using fresh generator optimizer")
    
    # Only load discriminator optimizer if discriminator was loaded successfully
    if d_loaded:
        try:
            optD.load_state_dict(ckpt.get("optD", {}))
        except Exception as e:
            print(f"[WARN] Could not load discriminator optimizer state: {e}")
            print("[INFO] Using fresh discriminator optimizer")
    else:
        print("[INFO] Using fresh discriminator optimizer (discriminator not loaded)")
    
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", math.inf))


@torch.no_grad()
def validate_l1(G: nn.Module, dl: DataLoader, device: torch.device,
                compute_sharpness: bool = False) -> tuple[float, dict]:
    """
    Validate model and compute L1 loss. Optionally compute piece sharpness metrics.

    Returns:
        (mean_l1, metrics_dict) where metrics_dict contains:
            - 'piece_sharpness': mean Sobel magnitude inside piece regions for the *generated* image
            - 'bg_sharpness': mean Sobel magnitude in background regions for the *generated* image
            - 'real_piece_sharpness': same metric computed on the real target (baseline)
            - 'real_bg_sharpness': baseline background sharpness

    Note:
        These sharpness metrics are simple (Sobel magnitude on grayscale). They’re useful
        for trend monitoring, not as a perfect perceptual score.
    """
    was_training = G.training
    G.eval()

    losses: list[float] = []
    fake_piece_sharpness: list[float] = []
    fake_bg_sharpness: list[float] = []
    real_piece_sharpness: list[float] = []
    real_bg_sharpness: list[float] = []

    for batch in dl:
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        fake_B = G(A)
        losses.append(F.l1_loss(fake_B, B).item())

        if compute_sharpness and ("mask" in batch):
            mask = batch["mask"].to(device)  # [N,1,H,W]

            piece_mask = (mask > 0.5).float()
            bg_mask = (mask <= 0.5).float()

            piece_pixels = piece_mask.sum() + 1e-8
            bg_pixels = bg_mask.sum() + 1e-8

            sobel_fake = sobel(to_grayscale(fake_B))
            sobel_real = sobel(to_grayscale(B))

            fake_piece = (sobel_fake * piece_mask).sum() / piece_pixels
            fake_bg = (sobel_fake * bg_mask).sum() / bg_pixels
            real_piece = (sobel_real * piece_mask).sum() / piece_pixels
            real_bg = (sobel_real * bg_mask).sum() / bg_pixels

            fake_piece_sharpness.append(float(fake_piece.item()))
            fake_bg_sharpness.append(float(fake_bg.item()))
            real_piece_sharpness.append(float(real_piece.item()))
            real_bg_sharpness.append(float(real_bg.item()))

    metrics: dict = {}
    if compute_sharpness and fake_piece_sharpness:
        metrics["piece_sharpness"] = float(__import__("numpy").mean(fake_piece_sharpness))
        metrics["bg_sharpness"] = float(__import__("numpy").mean(fake_bg_sharpness))
    if compute_sharpness and real_piece_sharpness:
        metrics["real_piece_sharpness"] = float(__import__("numpy").mean(real_piece_sharpness))
        metrics["real_bg_sharpness"] = float(__import__("numpy").mean(real_bg_sharpness))

    if was_training:
        G.train()

    mean_l1 = float(__import__("numpy").mean(losses)) if losses else float("nan")
    return mean_l1, metrics


def get_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--train_csv", type=str, default="data/splits_rect/train.csv")
    ap.add_argument("--val_csv", type=str, default="data/splits_rect/val.csv")

    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--load_size", type=int, default=286, help="resize before random crop; set ==image_size to disable crop")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)

    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--lambda_grad", type=float, default=0.0, help="Sobel/edge loss weight (0 disables)")
    ap.add_argument("--grad_gray", action="store_true", help="compute Sobel loss on grayscale instead of RGB")
    ap.add_argument("--lambda_perceptual", type=float, default=0.0, help="VGG perceptual loss weight (0 disables)")
    ap.add_argument("--lambda_feature_match", type=float, default=0.0,
                   help="Feature matching loss weight (0 disables). Matches intermediate D features.")
    ap.add_argument("--d_train_freq", type=int, default=1,
                   help="Train discriminator every N steps (default: 1 = every step, 2 = every other step)")
    ap.add_argument("--label_smooth", type=float, default=0.0,
                   help="Label smoothing for discriminator (0.0 = no smoothing, 0.1 = 10% smoothing)")
    ap.add_argument("--spectral_norm", action="store_true",
                   help="Use spectral normalization in discriminator for stable training")

    ap.add_argument("--gan_loss", type=str, default="bce", choices=["bce", "lsgan", "hinge"],
                   help="GAN loss type: bce, lsgan, or hinge (default: bce)")
    ap.add_argument("--no_amp", action="store_true", help="disable AMP even on CUDA")
    
    # Piece-focused supervision
    ap.add_argument("--piece_mask_dir", type=str, default=None,
                   help="Directory containing piece masks (same stem as synthetic images)")
    ap.add_argument("--use_piece_mask", action="store_true",
                   help="Enable piece mask weighting for losses")
    ap.add_argument("--piece_weight", type=float, default=6.0,
                   help="Weight multiplier for piece pixels in losses (default: 6.0)")
    ap.add_argument("--piece_edge_weight", type=float, default=0.0,
                   help="Extra weight for pixels near piece-mask boundaries (0 disables). Typical: 10-20.")
    ap.add_argument("--piece_edge_width", type=int, default=7,
                   help="Kernel size for boundary band (odd int, default: 7).")
    ap.add_argument("--use_piece_D", action="store_true",
                   help="Use additional piece-patch discriminator")
    ap.add_argument("--piece_crop_size", type=int, default=96,
                   help="Crop size for piece patches (default: 96)")
    ap.add_argument("--piece_patches_per_image", type=int, default=2,
                   help="Number of piece patches per image (default: 2)")
    ap.add_argument("--lambda_piece_gan", type=float, default=1.0,
                   help="Weight for piece GAN loss in generator (default: 1.0)")
    ap.add_argument("--lambda_piece_l1", type=float, default=0.0,
                   help="Additional L1 loss on piece patches (0 disables)")
    ap.add_argument("--lambda_piece_grad", type=float, default=0.0,
                   help="Additional Sobel/edge loss on piece patches (0 disables)")
    ap.add_argument("--lambda_piece_perceptual", type=float, default=0.0,
                   help="Additional VGG perceptual loss on piece patches (0 disables)")
    ap.add_argument("--r1_gamma", type=float, default=0.0,
                   help="R1 regularization weight for discriminator (0 disables, default: 0.0)")

    # Option 1: refine piece mask on-the-fly using real-image edges (spill-aware)
    ap.add_argument("--refine_real_mask", action="store_true",
                   help="Refine coarse piece masks using real-image edges (spill-aware).")
    ap.add_argument("--refine_quantile", type=float, default=0.85,
                   help="Edge quantile used for centroiding (higher = tighter, default: 0.85).")
    ap.add_argument("--refine_sigma", type=float, default=8.0,
                   help="Gaussian sigma in pixels for refined mask (default: 8.0).")
    ap.add_argument("--refine_border", type=int, default=2,
                   help="Ignore this many pixels from square borders when estimating edges (default: 2).")
    ap.add_argument("--refine_strength", type=float, default=1.0,
                   help="Blend strength between coarse and refined mask (0=coarse, 1=refined).")
    ap.add_argument("--refine_occ_thr", type=float, default=0.08,
                   help="Per-square MEAN mask threshold for occupancy (default: 0.08).")
    ap.add_argument("--refine_spill_px", type=int, default=8,
                   help="Allow refined mask to spill this many pixels outside occupied squares (default: 8).")

    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=1000)

    ap.add_argument("--samples_dir", type=str, default="results/train_samples")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from. If not set, auto-resumes from ckpt_dir/latest.pt if exists.")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"[INFO] device={device}")

    # Dataset
    ds_train = PairedChessDataset(
        args.train_csv,
        repo_root=".",
        image_size=args.image_size,
        train=True,
        seed=args.seed,
        load_size=args.load_size,
        piece_mask_dir=args.piece_mask_dir,
        use_piece_mask=args.use_piece_mask,
        refine_real_mask=args.refine_real_mask,
        refine_quantile=args.refine_quantile,
        refine_sigma=args.refine_sigma,
        refine_border=args.refine_border,
        refine_strength=args.refine_strength,
        refine_occ_thr=args.refine_occ_thr,
        refine_spill_px=args.refine_spill_px,
    )
    ds_val = PairedChessDataset(
        args.val_csv,
        repo_root=".",
        image_size=args.image_size,
        train=False,
        seed=args.seed,
        load_size=args.load_size,
        piece_mask_dir=args.piece_mask_dir,
        use_piece_mask=args.use_piece_mask,
        refine_real_mask=args.refine_real_mask,
        refine_quantile=args.refine_quantile,
        refine_sigma=args.refine_sigma,
        refine_border=args.refine_border,
        refine_strength=args.refine_strength,
        refine_occ_thr=args.refine_occ_thr,
        refine_spill_px=args.refine_spill_px,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Models
    G = UNetGenerator().to(device)
    D = PatchDiscriminator(use_spectral_norm=args.spectral_norm).to(device)
    init_weights(G)
    if not args.spectral_norm:
        init_weights(D)  # Don't init if using spectral norm (it handles it)
    
    # Optional piece discriminator
    D_piece = None
    if args.use_piece_D:
        D_piece = PatchDiscriminator(use_spectral_norm=args.spectral_norm).to(device)
        if not args.spectral_norm:
            init_weights(D_piece)
    
    # Log features
    if args.spectral_norm:
        print(f"[INFO] Spectral normalization enabled in discriminator")
    if args.lambda_feature_match > 0:
        print(f"[INFO] Feature matching loss enabled (weight={args.lambda_feature_match})")
    if args.use_piece_mask:
        print(f"[INFO] Piece mask weighting enabled (piece_weight={args.piece_weight})")
        if args.refine_real_mask:
            print(f"[INFO] Option1 mask refinement enabled (spill_px={args.refine_spill_px}, occ_thr={args.refine_occ_thr}, sigma={args.refine_sigma}, q={args.refine_quantile})")
    if args.use_piece_D:
        print(f"[INFO] Piece-patch discriminator enabled (crop_size={args.piece_crop_size}, patches={args.piece_patches_per_image})")
    if args.gan_loss == "hinge":
        print(f"[INFO] Using hinge loss (label smoothing disabled)")
        # Disable label smoothing for hinge loss
        args.label_smooth = 0.0
    if args.r1_gamma > 0:
        print(f"[INFO] R1 regularization enabled (gamma={args.r1_gamma})")

    # VGG for perceptual loss (if enabled)
    vgg_loss = None
    if (args.lambda_perceptual > 0) or (args.lambda_piece_perceptual > 0):
        vgg_loss = VGGPerceptualLoss().to(device)
        vgg_loss.eval()
        print(f"[INFO] VGG perceptual loss enabled (weight={args.lambda_perceptual})")

    # Losses
    if args.gan_loss == "lsgan":
        adv_crit = nn.MSELoss()
    else:
        adv_crit = nn.BCEWithLogitsLoss()

    l1_crit = nn.L1Loss()

    # Opt
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    # D optimizer: combine D and D_piece if both exist
    d_params = list(D.parameters())
    if D_piece is not None:
        d_params += list(D_piece.parameters())
    optD = torch.optim.Adam(d_params, lr=args.lr, betas=(args.beta1, args.beta2))
    
    # Log discriminator training frequency
    if args.d_train_freq > 1:
        print(f"[INFO] Training discriminator every {args.d_train_freq} steps (G trains every step)")
    if args.label_smooth > 0:
        print(f"[INFO] Label smoothing enabled: {args.label_smooth}")

    # Resume if exists
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_latest = ckpt_dir / "latest.pt"
    ckpt_best = ckpt_dir / "best.pt"
    ckpt_best_piece = ckpt_dir / "best_piece.pt"

    step = 0
    best_val = float("inf")
    best_piece_delta = float("-inf")

    # Determine which checkpoint to resume from
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif ckpt_latest.exists():
        resume_path = ckpt_latest

    if resume_path is not None:
        # Use strict=False when resuming if discriminator architecture changed (e.g., spectral norm, layer structure)
        step, best_val = load_checkpoint(resume_path, G, D, optG, optD, strict=False)
        print(f"[INFO] resumed from {resume_path} (step={step}, best_val={best_val:.4f})")

    # AMP setup (on by default on CUDA, unless disabled)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    try:
        scaler = torch.amp.GradScaler(enabled=use_amp)  # torch>=2.0
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # legacy fallback

    if use_amp and hasattr(torch, "autocast"):
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)  # noqa: E731
    else:
        try:
            autocast = lambda: torch.autocast(device_type="cpu", enabled=False)  # noqa: E731
        except Exception:
            autocast = lambda: nullcontext()  # noqa: E731

    samples_dir = Path(args.samples_dir)

    t0 = time.time()
    it = iter(dl_train)

    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_train)
            batch = next(it)

        step += 1
        A = batch["A"].to(device, non_blocking=True)
        B = batch["B"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)  # [N, 1, H, W] in {0, 1}

        # ----------------------------
        # Train D (only every d_train_freq steps)
        # ----------------------------
        if step % args.d_train_freq == 0:
            optD.zero_grad(set_to_none=True)

            with autocast():
                fake_B = G(A).detach()
                
                # Global discriminator
                if args.gan_loss == "hinge":
                    pred_real = D(A, B)
                    pred_fake = D(A, fake_B)
                    d_real_global, d_fake_global = hinge_loss_D(pred_real, pred_fake)
                    d_loss_global = 0.5 * (d_real_global + d_fake_global)
                else:
                    # BCE or LSGAN
                    pred_real = D(A, B)
                    pred_fake = D(A, fake_B)
                    real_targets = torch.ones_like(pred_real) * (1.0 - args.label_smooth)
                    fake_targets = torch.zeros_like(pred_fake) + args.label_smooth
                    d_real_global = adv_crit(pred_real, real_targets)
                    d_fake_global = adv_crit(pred_fake, fake_targets)
                    d_loss_global = 0.5 * (d_real_global + d_fake_global)
                
                d_loss = d_loss_global
                # Store for logging
                d_real = d_real_global
                d_fake = d_fake_global
                
                # Piece discriminator
                d_loss_piece = torch.tensor(0.0, device=device)
                A_patches = None
                B_patches = None
                if D_piece is not None:
                    # Sample coords once and reuse for (A, B, fake_B) for aligned patches
                    A_patches, B_patches, _coords_piece = sample_piece_patches_with_coords(
                        A, B, mask, args.piece_crop_size, args.piece_patches_per_image
                    )
                    A_patches_fake = extract_patches(A, _coords_piece, args.piece_crop_size)
                    fake_B_patches = extract_patches(fake_B, _coords_piece, args.piece_crop_size)
                    
                    if args.gan_loss == "hinge":
                        pred_real_piece = D_piece(A_patches, B_patches)
                        pred_fake_piece = D_piece(A_patches_fake, fake_B_patches)
                        d_real_piece, d_fake_piece = hinge_loss_D(pred_real_piece, pred_fake_piece)
                        d_loss_piece = 0.5 * (d_real_piece + d_fake_piece)
                    else:
                        pred_real_piece = D_piece(A_patches, B_patches)
                        pred_fake_piece = D_piece(A_patches_fake, fake_B_patches)
                        real_targets_piece = torch.ones_like(pred_real_piece) * (1.0 - args.label_smooth)
                        fake_targets_piece = torch.zeros_like(pred_fake_piece) + args.label_smooth
                        d_real_piece = adv_crit(pred_real_piece, real_targets_piece)
                        d_fake_piece = adv_crit(pred_fake_piece, fake_targets_piece)
                        d_loss_piece = 0.5 * (d_real_piece + d_fake_piece)
                    
                    d_loss = d_loss + d_loss_piece
                
                # R1 regularization (only on real data)
                r1_penalty = torch.tensor(0.0, device=device)
                if args.r1_gamma > 0:
                    # R1 for global discriminator
                    B_real_r1 = B.clone().requires_grad_(True)
                    pred_real_r1 = D(A, B_real_r1)
                    r1_penalty = compute_r1_penalty(pred_real_r1, B_real_r1, args.r1_gamma)
                    
                    # R1 for piece discriminator (if enabled)
                    if D_piece is not None and A_patches is not None and len(A_patches) > 0:
                        B_patches_r1 = B_patches.clone().requires_grad_(True)
                        pred_real_piece_r1 = D_piece(A_patches, B_patches_r1)
                        r1_piece = compute_r1_penalty(pred_real_piece_r1, B_patches_r1, args.r1_gamma)
                        r1_penalty = r1_penalty + r1_piece
                    
                    d_loss = d_loss + r1_penalty

            if use_amp:
                scaler.scale(d_loss).backward()
                scaler.step(optD)
            else:
                d_loss.backward()
                optD.step()
        else:
            # Skip D training this step, but still compute loss for logging
            with torch.no_grad():
                fake_B_detached = G(A).detach()
                if args.gan_loss == "hinge":
                    pred_real = D(A, B)
                    pred_fake = D(A, fake_B_detached)
                    d_real, d_fake = hinge_loss_D(pred_real, pred_fake)
                    d_loss = 0.5 * (d_real + d_fake)
                else:
                    pred_real = D(A, B)
                    pred_fake = D(A, fake_B_detached)
                    real_targets = torch.ones_like(pred_real) * (1.0 - args.label_smooth)
                    fake_targets = torch.zeros_like(pred_fake) + args.label_smooth
                    d_real = adv_crit(pred_real, real_targets)
                    d_fake = adv_crit(pred_fake, fake_targets)
                    d_loss = 0.5 * (d_real + d_fake)

        # ----------------------------
        # Train G
        # ----------------------------
        optG.zero_grad(set_to_none=True)

        with autocast():
            fake_B = G(A)
            
            # Create weight map for piece-focused losses
            if args.use_piece_mask:
                weight_map = make_weight_map(mask, args.piece_weight, bg_weight=1.0)

                # Optional: emphasize mask boundaries (helps crisp silhouettes)
                if args.piece_edge_weight and args.piece_edge_weight > 0:
                    # boost boundary pixels instead of replacing
                    band = mask_boundary_band(mask, k=args.piece_edge_width)
                    weight_map = weight_map * (1.0 + args.piece_edge_weight * band)

                # Normalize weights to keep loss magnitudes stable (mean ~1)
                weight_map_norm = weight_map / (weight_map.mean() + 1e-8)
            else:
                weight_map_norm = torch.ones_like(mask)
            
            # Feature matching loss: extract intermediate D features
            g_feat_match = torch.tensor(0.0, device=device)
            if args.lambda_feature_match > 0:
                # Get features from real and fake pairs
                real_features, _ = D(A, B, return_features=True)
                fake_features, pred_fake = D(A, fake_B, return_features=True)
                
                # Match features at each intermediate layer
                for rf, ff in zip(real_features, fake_features):
                    g_feat_match += l1_crit(ff, rf.detach())
                g_feat_match *= args.lambda_feature_match
            else:
                pred_fake = D(A, fake_B)

            # GAN loss (global)
            if args.gan_loss == "hinge":
                g_gan_global = hinge_loss_G(pred_fake)
            else:
                gan_targets = torch.ones_like(pred_fake)
                g_gan_global = adv_crit(pred_fake, gan_targets)
            
            g_gan = g_gan_global
            
            # Piece discriminator GAN loss (+ optional patch-level reconstruction losses)
            g_piece_l1 = torch.tensor(0.0, device=device)
            g_piece_grad = torch.tensor(0.0, device=device)
            g_piece_perc = torch.tensor(0.0, device=device)
            if D_piece is not None:
                # Sample coords once to align patches across A/B/fake_B
                A_patches_G, B_patches_G, _coords_piece_G = sample_piece_patches_with_coords(
                    A, B, mask, args.piece_crop_size, args.piece_patches_per_image
                )
                fake_B_patches_G = extract_patches(fake_B, _coords_piece_G, args.piece_crop_size)
                pred_fake_piece = D_piece(A_patches_G, fake_B_patches_G)
                if args.gan_loss == "hinge":
                    g_gan_piece = hinge_loss_G(pred_fake_piece)
                else:
                    gan_targets_piece = torch.ones_like(pred_fake_piece)
                    g_gan_piece = adv_crit(pred_fake_piece, gan_targets_piece)
                g_gan = g_gan + args.lambda_piece_gan * g_gan_piece

                # Patch-level reconstruction losses on the same piece crops
                if args.lambda_piece_l1 > 0:
                    g_piece_l1 = l1_crit(fake_B_patches_G, B_patches_G) * args.lambda_piece_l1

                if args.lambda_piece_grad > 0:
                    fb_p = to_grayscale(fake_B_patches_G) if args.grad_gray else fake_B_patches_G
                    bb_p = to_grayscale(B_patches_G) if args.grad_gray else B_patches_G
                    g_piece_grad = l1_crit(sobel(fb_p), sobel(bb_p)) * args.lambda_piece_grad

                if (args.lambda_piece_perceptual > 0) and (vgg_loss is not None):
                    g_piece_perc = vgg_loss(fake_B_patches_G, B_patches_G) * args.lambda_piece_perceptual

            # Weighted L1 loss
            if args.use_piece_mask:
                g_l1 = weighted_l1_loss(fake_B, B, weight_map_norm) * args.lambda_l1
            else:
                g_l1 = l1_crit(fake_B, B) * args.lambda_l1

            # Weighted gradient/Sobel loss
            g_grad = torch.tensor(0.0, device=device)
            if args.lambda_grad > 0:
                fb = to_grayscale(fake_B) if args.grad_gray else fake_B
                bb = to_grayscale(B) if args.grad_gray else B
                g_fake = sobel(fb)
                g_real = sobel(bb)
                if args.use_piece_mask:
                    # For grayscale sobel output [N,1,H,W], weight map is already [N,1,H,W]
                    g_grad = weighted_l1_loss(g_fake, g_real, weight_map_norm) * args.lambda_grad
                else:
                    g_grad = l1_crit(g_fake, g_real) * args.lambda_grad

            # Weighted perceptual loss
            g_perc = torch.tensor(0.0, device=device)
            if vgg_loss is not None:
                if args.use_piece_mask:
                    g_perc = weighted_perceptual_loss(vgg_loss, fake_B, B, weight_map_norm) * args.lambda_perceptual
                else:
                    g_perc = vgg_loss(fake_B, B) * args.lambda_perceptual

            g_loss = g_gan + g_l1 + g_grad + g_perc + g_feat_match + g_piece_l1 + g_piece_grad + g_piece_perc

        if use_amp:
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()
        else:
            g_loss.backward()
            optG.step()

        # ----------------------------
        # Logging / Samples / Val
        # ----------------------------
        if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
            dt = time.time() - t0
            ips = step / max(dt, 1e-9)
            msg = (
                f"[step {step:6d}] "
                f"D={d_loss.item():.4f} (real={d_real.item():.4f} fake={d_fake.item():.4f}) | "
                f"G={g_loss.item():.4f} (gan={g_gan.item():.4f} l1={g_l1.item():.4f}"
            )
            if args.lambda_grad > 0:
                msg += f" grad={g_grad.item():.4f}"
            if args.lambda_perceptual > 0:
                msg += f" perc={g_perc.item():.4f}"
            if args.lambda_feature_match > 0:
                msg += f" feat={g_feat_match.item():.4f}"
            if args.lambda_piece_l1 > 0:
                msg += f" pL1={g_piece_l1.item():.4f}"
            if args.lambda_piece_grad > 0:
                msg += f" pGrad={g_piece_grad.item():.4f}"
            if args.lambda_piece_perceptual > 0:
                msg += f" pPerc={g_piece_perc.item():.4f}"
            msg += f") | {ips:.2f} it/s"
            print(msg)

        if args.sample_every > 0 and step % args.sample_every == 0:
            out = samples_dir / f"step_{step:06d}.png"
            save_samples(G, batch, device, out)
            print(f"[OK] wrote sample {out}")

        if args.val_every > 0 and step % args.val_every == 0:
            mean_l1, metrics = validate_l1(G, dl_val, device, compute_sharpness=args.use_piece_mask)
            msg = f"[VAL] step={step} mean_L1={mean_l1:.4f}"
            if args.use_piece_mask and 'piece_sharpness' in metrics:
                piece_sharp = float(metrics['piece_sharpness'])
                bg_sharp = float(metrics['bg_sharpness'])
                delta = piece_sharp - bg_sharp
                msg += f" piece_sharp={piece_sharp:.6f} bg_sharp={bg_sharp:.6f} delta={delta:.6f}"

                # Show target sharpness too (helps diagnose 'model still blurrier than GT')
                if 'real_piece_sharpness' in metrics and 'real_bg_sharpness' in metrics:
                    rp = float(metrics['real_piece_sharpness'])
                    rb = float(metrics['real_bg_sharpness'])
                    gt_delta = rp - rb
                    msg += f" | GT_piece={rp:.4f} GT_bg={rb:.4f} GT_delta={gt_delta:.4f}"
            print(msg)

            # Always save latest
            save_checkpoint(ckpt_latest, step, best_val, G, D, optG, optD)

            # Save best by val L1
            if mean_l1 < best_val:
                best_val = mean_l1
                save_checkpoint(ckpt_best, step, best_val, G, D, optG, optD)
                print(f"[OK] new best checkpoint: {ckpt_best} (best_val={best_val:.4f})")

            # Save best by delta = piece_sharp - bg_sharp (stable, no epsilon blow-ups)
            if args.use_piece_mask and ('piece_sharpness' in metrics):
                piece_sharp = float(metrics.get('piece_sharpness', 0.0))
                bg_sharp = float(metrics.get('bg_sharpness', 0.0))
                delta = piece_sharp - bg_sharp
                if delta > best_piece_delta:
                    best_piece_delta = delta
                    save_checkpoint(ckpt_best_piece, step, best_val, G, D, optG, optD)
                    print(f"[OK] new best piece checkpoint: {ckpt_best_piece} (best_delta={best_piece_delta:.4f})")

    # Final save
    save_checkpoint(ckpt_latest, step, best_val, G, D, optG, optD)
    print(f"[DONE] saved {ckpt_latest}")


if __name__ == "__main__":
    main()
