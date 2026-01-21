import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from src.datasets.pairs_dataset import PairedChessDataset
from src.models.pix2pix_nets import UNetGenerator  # adjust if your import differs
from scripts.train_pix2pix import save_samples  # uses your function


def load_G(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # warning is fine
    state = ckpt["G"] if "G" in ckpt else ckpt

    # Your project’s generator takes in_ch/out_ch (or defaults)
    G = UNetGenerator()  # simplest: matches train_pix2pix.py (G = UNetGenerator())
    # If you really want explicit:
    # G = UNetGenerator(in_ch=3, out_ch=3)

    G.load_state_dict(state, strict=True)
    G.to(device).eval()
    return G


@torch.no_grad()
def save_4wide(G1, G2, batch, device, out_path: Path):
    from PIL import Image
    from scripts.train_pix2pix import tensor_to_uint8

    A = batch["A"].to(device)
    B = batch["B"].to(device)
    f1 = G1(A)
    f2 = G2(A)

    a = tensor_to_uint8(A[0])
    x1 = tensor_to_uint8(f1[0])
    x2 = tensor_to_uint8(f2[0])
    b = tensor_to_uint8(B[0])

    H, W, _ = a.shape
    canvas = Image.new("RGB", (W * 4, H))
    canvas.paste(Image.fromarray(a), (0, 0))
    canvas.paste(Image.fromarray(x1), (W, 0))
    canvas.paste(Image.fromarray(x2), (W * 2, 0))
    canvas.paste(Image.fromarray(b), (W * 3, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt1", required=True)
    ap.add_argument("--ckpt2", required=True)
    ap.add_argument("--out_dir", default="results/ckpt_compare")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = PairedChessDataset(
        args.csv,
        repo_root=".",
        image_size=args.image_size,
        train=False,
        seed=args.seed,
        load_size=args.image_size,
        piece_mask_dir=None,
        use_piece_mask=False,
    )

    # deterministic subset
    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

    loader = DataLoader([ds[i] for i in idxs], batch_size=1, shuffle=False, num_workers=0)

    G1 = load_G(Path(args.ckpt1), device)
    G2 = load_G(Path(args.ckpt2), device)

    out_dir = Path(args.out_dir)
    tag1 = Path(args.ckpt1).stem
    tag2 = Path(args.ckpt2).stem

    for k, batch in enumerate(loader):
        # 3-wide per checkpoint
        save_samples(G1, batch, device, out_dir / f"{tag1}" / f"{k:03d}.png")
        save_samples(G2, batch, device, out_dir / f"{tag2}" / f"{k:03d}.png")
        # 4-wide direct compare
        save_4wide(G1, G2, batch, device, out_dir / f"{tag1}_vs_{tag2}" / f"{k:03d}.png")

    print("[OK] wrote", out_dir.resolve())


if __name__ == "__main__":
    main()
