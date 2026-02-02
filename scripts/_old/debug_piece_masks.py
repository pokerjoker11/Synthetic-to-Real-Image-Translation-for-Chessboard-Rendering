# scripts/debug_piece_masks.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def overlay_red(img_rgb: Image.Image, mask_l: Image.Image, alpha=0.5) -> Image.Image:
    img = np.array(img_rgb).astype(np.float32)
    m = (np.array(mask_l) > 127).astype(np.float32)[..., None]
    out = img.copy()
    out[..., 0] = np.clip(out[..., 0] * (1 - alpha * m[..., 0]) + 255 * alpha * m[..., 0], 0, 255)
    return Image.fromarray(out.astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits_rect/train_clean.csv")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--out_dir", default="results/mask_debug")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "synth" not in df.columns or "real" not in df.columns:
        raise ValueError("CSV must contain 'synth' and 'real' columns")

    sample = df.sample(min(args.num, len(df)), random_state=args.seed)

    missing = 0
    for i, r in enumerate(sample.to_dict("records")):
        real_path = Path(r["real"])
        synth_path = Path(r["synth"])
        mask_path = mask_dir / (synth_path.stem + ".png")

        if not real_path.exists() or not synth_path.exists():
            print(f"[SKIP] missing image: real={real_path.exists()} synth={synth_path.exists()}")
            continue
        if not mask_path.exists():
            print(f"[MISS] mask: {mask_path}")
            missing += 1
            continue

        real = Image.open(real_path).convert("RGB").resize((256, 256))
        synth = Image.open(synth_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256))

        overlay_real = overlay_red(real, mask, alpha=0.5)
        overlay_synth = overlay_red(synth, mask, alpha=0.5)

        real.save(out_dir / f"real_{i}.png")
        synth.save(out_dir / f"synth_{i}.png")
        mask.save(out_dir / f"mask_{i}.png")
        overlay_real.save(out_dir / f"overlay_real_{i}.png")
        overlay_synth.save(out_dir / f"overlay_synth_{i}.png")

    print(f"[OK] wrote {out_dir} | missing_masks={missing}/{len(sample)}")

if __name__ == "__main__":
    main()
