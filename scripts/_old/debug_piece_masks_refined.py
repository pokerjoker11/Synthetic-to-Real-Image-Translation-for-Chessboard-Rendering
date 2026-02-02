# scripts/debug_piece_masks_refined.py
# Visualize coarse vs refined masks on top of REAL images (and optionally synth).

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch


def overlay_red(img_rgb: Image.Image, mask_hw: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Overlay a [0,1] mask as red on an RGB image."""
    img = np.asarray(img_rgb).astype(np.float32)
    m = np.clip(mask_hw, 0.0, 1.0)[..., None].astype(np.float32)
    out = img.copy()
    out[..., 0] = np.clip(out[..., 0] * (1.0 - alpha * m[..., 0]) + 255.0 * alpha * m[..., 0], 0.0, 255.0)
    return Image.fromarray(out.astype(np.uint8))


def load_coarse_mask(mask_dir: Path, real_path: Path, synth_path: Path, size: int) -> np.ndarray:
    candidates = [
        mask_dir / f"{real_path.stem}.png",
        mask_dir / f"{synth_path.stem}.png",
        mask_dir / real_path.name,
    ]
    mp = None
    for c in candidates:
        if c.exists():
            mp = c
            break
    if mp is None:
        raise FileNotFoundError(f"No mask found for {real_path.name} / {synth_path.name} in {mask_dir}")

    m = Image.open(mp).convert("L").resize((size, size), resample=Image.NEAREST)
    arr = (np.asarray(m) > 127).astype(np.float32)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/pairs/train.csv")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--out_dir", default="results/mask_debug_refined")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--load_size", type=int, default=256)

    # Option 1 refinement args (must match PairedChessDataset)
    ap.add_argument("--refine_real_mask", action="store_true")
    ap.add_argument("--refine_quantile", type=float, default=0.85)
    ap.add_argument("--refine_sigma", type=float, default=8.0)
    ap.add_argument("--refine_border", type=int, default=2)
    ap.add_argument("--refine_strength", type=float, default=1.0)
    ap.add_argument("--refine_occ_thr", type=float, default=0.08)
    ap.add_argument("--refine_spill_px", type=int, default=8)

    ap.add_argument("--also_synth", action="store_true", help="Also save synth overlays")
    args = ap.parse_args()

    # Late imports so PYTHONPATH works nicely
    import sys
    sys.path.insert(0, ".")
    from src.datasets.pairs_dataset import PairedChessDataset

    csv_path = Path(args.csv)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PairedChessDataset(
        csv_path,
        repo_root=".",
        image_size=args.image_size,
        load_size=args.load_size,
        train=False,
        seed=args.seed,
        piece_mask_dir=mask_dir,
        use_piece_mask=True,
        refine_real_mask=args.refine_real_mask,
        refine_quantile=args.refine_quantile,
        refine_sigma=args.refine_sigma,
        refine_border=args.refine_border,
        refine_strength=args.refine_strength,
        refine_occ_thr=args.refine_occ_thr,
        refine_spill_px=args.refine_spill_px,
    )

    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(ds), size=min(args.num, len(ds)), replace=False)

    for j, idx in enumerate(idxs):
        it = ds[int(idx)]
        real = ((it["B"] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
        synth = ((it["A"] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
        real_u8 = (real * 255).astype(np.uint8)
        synth_u8 = (synth * 255).astype(np.uint8)

        real_path = Path(it["real_path"])
        synth_path = Path(it["synth_path"])

        coarse = load_coarse_mask(mask_dir, real_path, synth_path, size=args.image_size)
        refined = it["mask"].clamp(0, 1)[0].numpy().astype(np.float32)

        Image.fromarray(real_u8).save(out_dir / f"real_{j}.png")
        Image.fromarray((coarse * 255).astype(np.uint8)).save(out_dir / f"mask_coarse_{j}.png")
        Image.fromarray((refined * 255).astype(np.uint8)).save(out_dir / f"mask_refined_{j}.png")
        overlay_red(Image.fromarray(real_u8), coarse).save(out_dir / f"overlay_real_coarse_{j}.png")
        overlay_red(Image.fromarray(real_u8), refined).save(out_dir / f"overlay_real_refined_{j}.png")

        if args.also_synth:
            Image.fromarray(synth_u8).save(out_dir / f"synth_{j}.png")
            overlay_red(Image.fromarray(synth_u8), coarse).save(out_dir / f"overlay_synth_coarse_{j}.png")

    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
