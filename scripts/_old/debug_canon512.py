from pathlib import Path
import argparse
import torch
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.pairs_dataset import PairedChessDataset


def t2pil_rgb(t: torch.Tensor) -> Image.Image:
    # t: (3,H,W) in [-1,1]
    t = (t.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    arr = (t.mul(255).byte().permute(1, 2, 0).numpy())
    return Image.fromarray(arr, mode="RGB")


def t2mask_green(mask: torch.Tensor, size: int) -> Image.Image:
    # mask: (1,H,W) float in [0,1]
    m = (mask.detach().cpu()[0] > 0.5).to(torch.uint8).mul(255).numpy()
    mL = Image.fromarray(m, mode="L").resize((size, size), resample=Image.NEAREST)
    green = Image.new("RGB", (size, size), (0, 255, 0))
    return green, mL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/pairs/train.csv")
    ap.add_argument("--mask_dir", default="data/masks_manual")
    ap.add_argument("--out", default="results/debug_canon512")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    use_mask = Path(args.mask_dir).exists()
    ds = PairedChessDataset(
        args.csv,
        image_size=512,
        canonical_size=480,
        pad_to=512,
        synth_crop_border=16,
        train=False,
        piece_mask_dir=args.mask_dir if use_mask else None,
        use_piece_mask=use_mask,
    )

    n = min(args.n, len(ds))
    print(f"[OK] csv={args.csv}  mask_dir={args.mask_dir if use_mask else 'NONE'}  n={len(ds)}  dump={n}")

    for i in range(n):
        s = ds[i]
        A = t2pil_rgb(s["A"])
        B = t2pil_rgb(s["B"])
        blend = Image.blend(A, B, 0.5)

        if use_mask and (s.get("mask", None) is not None):
            green, mL = t2mask_green(s["mask"], 512)
            # translucent green only where mask==1
            blend_g = Image.blend(blend, green, 0.35)
            blend = Image.composite(blend_g, blend, mL)

        canvas = Image.new("RGB", (512 * 3, 512))
        canvas.paste(A, (0, 0))
        canvas.paste(B, (512, 0))
        canvas.paste(blend, (1024, 0))
        fp = out / f"debug_{i:05d}.png"
        canvas.save(fp)
        print("wrote", fp)


if __name__ == "__main__":
    main()
