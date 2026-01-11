# scripts/train_pix2pix.py
import sys
import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.datasets.pairs_dataset import PairedChessDataset
from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator, init_weights


def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> [0, 1]
    return (x + 1.0) / 2.0


@torch.no_grad()
def save_samples(G, batch, out_path: Path, device: torch.device):
    G.eval()
    A = batch["A"].to(device)
    B = batch["B"].to(device)
    fake = G(A)

    # [A | fake | B]
    grid = torch.cat([denorm(A), denorm(fake), denorm(B)], dim=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path, nrow=1)
    G.train()


@torch.no_grad()
def validate_l1(G, val_loader, device: torch.device, max_batches: int = 25) -> float:
    G.eval()
    l1 = nn.L1Loss(reduction="mean")
    total = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        fake = G(A)
        total += float(l1(fake, B))
        count += 1
    G.train()
    return total / max(count, 1)


def save_checkpoint(path: Path, G, D, optG, optD, step: int, best_val: float):
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


def load_checkpoint(path: Path, G, D, optG, optD):
    ckpt = torch.load(path, map_location="cpu")
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    optG.load_state_dict(ckpt["optG"])
    optD.load_state_dict(ckpt["optD"])
    step = int(ckpt.get("step", 0))
    best_val = float(ckpt.get("best_val", 1e9))
    return step, best_val


def main():
    ap = argparse.ArgumentParser()

    # Data paths (NEW)
    ap.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    ap.add_argument("--val_csv", type=str, default="data/splits/val.csv")

    # Output paths (NEW)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--samples_dir", type=str, default="results/train_samples")

    # Data params
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--load_size",
        type=int,
        default=0,
        help="Resize before cropping. 0 = dataset default. Set equal to image_size to disable crop jitter.",
    )


    # Optim / loss params
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--lambda_l1", type=float, default=100.0)

    # Train loop
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=400)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", type=str, default="auto")  # auto/cpu/cuda

    args = ap.parse_args()

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # Data
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    load_size = None if args.load_size == 0 else args.load_size

    ds_train = PairedChessDataset(
        train_csv,
        repo_root=".",
        image_size=args.image_size,
        load_size=load_size,
        train=True,
        seed=123,
    )
    ds_val = PairedChessDataset(
        val_csv,
        repo_root=".",
        image_size=args.image_size,
        load_size=load_size,
        train=False,
        seed=123,
    )


    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_iter = iter(dl_train)

    # Models
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    init_weights(G)
    init_weights(D)

    # Losses
    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Optimizers
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # AMP only for CUDA (no deprecation warnings on CPU)
    use_amp = (device.type == "cuda")
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        scaler = None
        autocast = None

    # Checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_latest = ckpt_dir / "latest.pt"
    ckpt_best = ckpt_dir / "best.pt"

    step = 0
    best_val = 1e9

    if args.resume and ckpt_latest.exists():
        step, best_val = load_checkpoint(ckpt_latest, G, D, optG, optD)
        print(f"[INFO] resumed from {ckpt_latest} at step={step}, best_val={best_val:.4f}")

    # Output dirs
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Fixed batch for consistent visualization
    fixed_bs = max(1, min(2, args.batch_size))
    fixed_batch = next(iter(DataLoader(ds_val, batch_size=fixed_bs, shuffle=False, num_workers=0)))

    t0 = time.perf_counter()
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            batch = next(train_iter)

        A = batch["A"].to(device)  # synth
        B = batch["B"].to(device)  # real

        # -------------------------
        # Train Discriminator
        # -------------------------
        optD.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                fake_B = G(A)
                pred_real = D(A, B)
                pred_fake = D(A, fake_B.detach())

                loss_D_real = gan_loss(pred_real, torch.ones_like(pred_real))
                loss_D_fake = gan_loss(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            scaler.scale(loss_D).backward()
            scaler.step(optD)
        else:
            fake_B = G(A)
            pred_real = D(A, B)
            pred_fake = D(A, fake_B.detach())

            loss_D_real = gan_loss(pred_real, torch.ones_like(pred_real))
            loss_D_fake = gan_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optD.step()

        # -------------------------
        # Train Generator
        # -------------------------
        optG.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                pred_fake_for_G = D(A, fake_B)
                loss_G_GAN = gan_loss(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
                loss_G_L1 = l1_loss(fake_B, B) * args.lambda_l1
                loss_G = loss_G_GAN + loss_G_L1

            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()
        else:
            pred_fake_for_G = D(A, fake_B)
            loss_G_GAN = gan_loss(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_L1 = l1_loss(fake_B, B) * args.lambda_l1
            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()
            associated = optG.step()

        step += 1

        # Logging
        if step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            it_s = step / elapsed if elapsed > 0 else 0.0
            print(
                f"[step {step:6d}] "
                f"D={loss_D.item():.4f} (real={loss_D_real.item():.4f} fake={loss_D_fake.item():.4f}) | "
                f"G={loss_G.item():.4f} (gan={loss_G_GAN.item():.4f} l1={loss_G_L1.item():.4f}) | "
                f"{it_s:.2f} it/s"
            )

        # Save samples
        if step % args.sample_every == 0 or step == 1:
            out_path = samples_dir / f"step_{step:06d}.png"
            save_samples(G, fixed_batch, out_path, device)
            print(f"[OK] wrote sample {out_path}")

        # Validation + checkpointing
        if step % args.val_every == 0 or step == args.max_steps:
            val = validate_l1(G, dl_val, device=device, max_batches=25)
            print(f"[VAL] step={step} mean_L1={val:.4f}")

            save_checkpoint(ckpt_latest, G, D, optG, optD, step, best_val)

            if val < best_val:
                best_val = val
                save_checkpoint(ckpt_best, G, D, optG, optD, step, best_val)
                print(f"[OK] new best checkpoint: {ckpt_best} (best_val={best_val:.4f})")

    save_checkpoint(ckpt_latest, G, D, optG, optD, step, best_val)
    print(f"[DONE] saved {ckpt_latest}")


if __name__ == "__main__":
    main()
