import argparse
from pathlib import Path
import numpy as np
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Train CSV used by PairedChessDataset")
    ap.add_argument("--mask_dir", type=str, required=True, help="Folder with manual masks (png)")
    ap.add_argument("--out", type=str, required=True, help="Output .npz path (dx,dy samples)")
    ap.add_argument("--out_json", type=str, default="", help="Optional output .json path for Blender jitter (dx,dy lists)")
    ap.add_argument("--canonical_size", type=int, default=480)
    ap.add_argument("--pad_to", type=int, default=512)
    ap.add_argument("--synth_crop_border", type=int, default=16)

    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument("--min_pixels", type=int, default=80, help="Min mask pixels in a square to count it")
    ap.add_argument("--clamp", type=float, default=0.45, help="Clamp |dx|,|dy| in square-normalized units")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Import here so your repo import path resolves in your environment
    from src.datasets.pairs_dataset import PairedChessDataset

    ds = PairedChessDataset(
        args.csv,
        image_size=args.pad_to,
        canonical_size=args.canonical_size,
        pad_to=args.pad_to,
        synth_crop_border=args.synth_crop_border,
        train=False,
        piece_mask_dir=args.mask_dir,
        use_piece_mask=True,
    )

    final_size = int(args.pad_to)
    board_size = 448  # your canonical 8x8 board pixels (square=56 at 512)
    margin = (final_size - board_size) // 2
    step = board_size // 8  # 56
    start = margin

    rng = np.random.default_rng(args.seed)

    dxs = []
    dys = []
    per_image_counts = []

    n = len(ds)
    idxs = np.arange(n)
    if args.max_items and args.max_items > 0:
        idxs = rng.choice(idxs, size=min(args.max_items, n), replace=False)

    def mask_to_binary(m):
        # m is torch tensor [1,H,W] or [H,W]; convert robustly to {0,1}
        import torch
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().float()
            if m.ndim == 3:
                m = m[0]
            m = m.numpy()
        m = m.astype(np.float32)
        # if in [-1,1], map to [0,1]
        if m.min() < 0.0:
            m = (m + 1.0) * 0.5
        return (m > 0.5).astype(np.uint8)

    for ii, idx in enumerate(idxs):
        sample = ds[int(idx)]
        real_stem = Path(sample["real_path"]).stem
        if not (Path(args.mask_dir) / f"{real_stem}.png").exists():
            continue

        m = sample.get("mask", None)
        mb = mask_to_binary(m)  # HxW uint8
        H, W = mb.shape
        assert H == final_size and W == final_size, (H, W, final_size)

        cnt_this = 0

        for r in range(8):
            y0 = start + r * step
            y1 = y0 + step
            for c in range(8):
                x0 = start + c * step
                x1 = x0 + step
                patch = mb[y0:y1, x0:x1]
                s = int(patch.sum())
                if s < int(args.min_pixels):
                    continue

                ys, xs = np.nonzero(patch)
                # centroid in image coords (use +0.5 to treat pixels as squares)
                cy = float(ys.mean() + y0 + 0.5)
                cx = float(xs.mean() + x0 + 0.5)

                center_y = float(y0 + step / 2.0)
                center_x = float(x0 + step / 2.0)

                dx = (cx - center_x) / float(step)
                dy = (cy - center_y) / float(step)

                # clamp outliers (spills / noise)
                dx = float(np.clip(dx, -args.clamp, args.clamp))
                dy = float(np.clip(dy, -args.clamp, args.clamp))

                dxs.append(dx)
                dys.append(dy)
                cnt_this += 1

        per_image_counts.append(cnt_this)

        if (ii + 1) % 500 == 0:
            print(f"[{ii+1}/{len(idxs)}] collected={len(dxs)} avg_per_img={np.mean(per_image_counts):.2f}")

    dxs = np.array(dxs, dtype=np.float32)
    dys = np.array(dys, dtype=np.float32)
    per_image_counts = np.array(per_image_counts, dtype=np.int32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        dx=dxs,
        dy=dys,
        final_size=final_size,
        board_size=board_size,
        margin=margin,
        step=step,
        min_pixels=int(args.min_pixels),
        clamp=float(args.clamp),
        n_images=int(len(idxs)),
        per_image_counts=per_image_counts,
    )

    # Also write Blender-compatible jitter JSON if requested
    if args.out_json:
        if dxs.size == 0 or dys.size == 0 or dxs.size != dys.size:
            raise RuntimeError("Cannot write jitter JSON: empty or mismatched dx/dy samples")
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dx": dxs.astype(float).tolist(),
            "dy": dys.astype(float).tolist(),
            # metadata (nice for debugging/repro)
            "final_size": int(final_size),
            "board_size": int(board_size),
            "margin": int(margin),
            "step": int(step),
            "min_pixels": int(args.min_pixels),
            "clamp": float(args.clamp),
            "n_images": int(len(idxs)),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("[OK] wrote jitter JSON:", out_json)

    def stats(a):
        if a.size == 0:
            return "EMPTY"
        q = np.quantile(a, [0.01, 0.05, 0.5, 0.95, 0.99])
        return f"mean={a.mean():+.4f} std={a.std():.4f} q01={q[0]:+.4f} q05={q[1]:+.4f} med={q[2]:+.4f} q95={q[3]:+.4f} q99={q[4]:+.4f}"

    print("\n==== Offset distribution saved ====")
    print("out:", out_path)
    print("samples:", dxs.size)
    print("dx:", stats(dxs))
    print("dy:", stats(dys))
    if per_image_counts.size:
        print("per-image piece-squares:", f"mean={per_image_counts.mean():.2f} med={np.median(per_image_counts):.0f} max={per_image_counts.max()}")

if __name__ == "__main__":
    main()
