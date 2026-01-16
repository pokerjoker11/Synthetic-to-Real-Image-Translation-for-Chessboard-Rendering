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


def load_checkpoint(path: Path, G, D, optG, optD) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    optG.load_state_dict(ckpt["optG"])
    optD.load_state_dict(ckpt["optD"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", math.inf))


@torch.no_grad()
def validate_l1(G: nn.Module, dl: DataLoader, device: torch.device) -> float:
    was_training = G.training
    G.eval()

    losses = []
    for batch in dl:
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        fake_B = G(A)
        losses.append(F.l1_loss(fake_B, B).item())

    if was_training:
        G.train()

    return float(np.mean(losses)) if losses else float("nan")


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

    ap.add_argument("--gan_loss", type=str, default="bce", choices=["bce", "lsgan"])
    ap.add_argument("--no_amp", action="store_true", help="disable AMP even on CUDA")

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
    )
    ds_val = PairedChessDataset(
        args.val_csv,
        repo_root=".",
        image_size=args.image_size,
        train=False,
        seed=args.seed,
        load_size=args.load_size,
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
    D = PatchDiscriminator().to(device)
    init_weights(G)
    init_weights(D)

    # Losses
    if args.gan_loss == "lsgan":
        adv_crit = nn.MSELoss()
    else:
        adv_crit = nn.BCEWithLogitsLoss()

    l1_crit = nn.L1Loss()

    # Opt
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Resume if exists
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_latest = ckpt_dir / "latest.pt"
    ckpt_best = ckpt_dir / "best.pt"

    step = 0
    best_val = float("inf")

    # Determine which checkpoint to resume from
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif ckpt_latest.exists():
        resume_path = ckpt_latest

    if resume_path is not None:
        step, best_val = load_checkpoint(resume_path, G, D, optG, optD)
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

        # ----------------------------
        # Train D
        # ----------------------------
        optD.zero_grad(set_to_none=True)

        with autocast():
            fake_B = G(A).detach()
            pred_real = D(A, B)
            pred_fake = D(A, fake_B)

            real_targets = torch.ones_like(pred_real)
            fake_targets = torch.zeros_like(pred_fake)

            d_real = adv_crit(pred_real, real_targets)
            d_fake = adv_crit(pred_fake, fake_targets)
            d_loss = 0.5 * (d_real + d_fake)

        if use_amp:
            scaler.scale(d_loss).backward()
            scaler.step(optD)
        else:
            d_loss.backward()
            optD.step()

        # ----------------------------
        # Train G
        # ----------------------------
        optG.zero_grad(set_to_none=True)

        with autocast():
            fake_B = G(A)
            pred_fake = D(A, fake_B)

            gan_targets = torch.ones_like(pred_fake)
            g_gan = adv_crit(pred_fake, gan_targets)

            g_l1 = l1_crit(fake_B, B) * args.lambda_l1

            g_grad = torch.tensor(0.0, device=device)
            if args.lambda_grad > 0:
                fb = to_grayscale(fake_B) if args.grad_gray else fake_B
                bb = to_grayscale(B) if args.grad_gray else B
                g_fake = sobel(fb)
                g_real = sobel(bb)
                g_grad = l1_crit(g_fake, g_real) * args.lambda_grad

            g_loss = g_gan + g_l1 + g_grad

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
            msg += f") | {ips:.2f} it/s"
            print(msg)

        if args.sample_every > 0 and step % args.sample_every == 0:
            out = samples_dir / f"step_{step:06d}.png"
            save_samples(G, batch, device, out)
            print(f"[OK] wrote sample {out}")

        if args.val_every > 0 and step % args.val_every == 0:
            mean_l1 = validate_l1(G, dl_val, device)
            print(f"[VAL] step={step} mean_L1={mean_l1:.4f}")

            # Always save latest
            save_checkpoint(ckpt_latest, step, best_val, G, D, optG, optD)

            # Save best by val L1
            if mean_l1 < best_val:
                best_val = mean_l1
                save_checkpoint(ckpt_best, step, best_val, G, D, optG, optD)
                print(f"[OK] new best checkpoint: {ckpt_best} (best_val={best_val:.4f})")

    # Final save
    save_checkpoint(ckpt_latest, step, best_val, G, D, optG, optD)
    print(f"[DONE] saved {ckpt_latest}")


if __name__ == "__main__":
    main()
