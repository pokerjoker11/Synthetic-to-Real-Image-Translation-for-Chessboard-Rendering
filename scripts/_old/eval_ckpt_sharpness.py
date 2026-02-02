# scripts/eval_ckpt_sharpness.py
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.pairs_dataset import PairedChessDataset
from src.models.pix2pix_nets import UNetGenerator
from scripts.train_pix2pix import validate_l1


def _resolve_ckpt(p: str) -> Path:
    """Allow passing either a checkpoint file or a directory that contains one."""
    path = Path(p)
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("best_piece.pt", "best.pt", "latest.pt"):
            cand = path / name
            if cand.exists():
                return cand
    raise FileNotFoundError(f"Could not find a checkpoint at: {p}")


def _load_generator(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    if "G" not in ck:
        raise KeyError(f"{ckpt_path} does not contain key 'G'. Keys: {list(ck.keys())}")

    G = UNetGenerator().to(device)
    missing, unexpected = G.load_state_dict(ck["G"], strict=False)

    if missing:
        print(f"[WARN] Missing keys while loading G (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading G (showing first 10): {unexpected[:10]}")

    G.eval()
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, default="data/splits_rect/val_final.csv")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="Checkpoint file OR dir (dir will try best_piece.pt/best.pt/latest.pt). Repeat --ckpt for multiple.")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--limit", type=int, default=200, help="How many val samples to evaluate (0 = all).")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)

    # Match your 512 pipeline defaults
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--load_size", type=int, default=512)
    ap.add_argument("--canonical_size", type=int, default=480)
    ap.add_argument("--pad_to", type=int, default=512)
    ap.add_argument("--synth_crop_border", type=int, default=16)

    # Masks (needed for sharpness metrics)
    ap.add_argument("--use_piece_mask", action="store_true")
    ap.add_argument("--piece_mask_dir", type=str, default="data/masks_manual")
    ap.add_argument("--refine_real_mask", action="store_true")

    ap.add_argument("--out_json", type=str, default="", help="Optional: write results to this JSON file.")

    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ds = PairedChessDataset(
        args.val_csv,
        repo_root=".",
        image_size=args.image_size,
        load_size=args.load_size,
        canonical_size=args.canonical_size,
        pad_to=args.pad_to,
        synth_crop_border=args.synth_crop_border,
        train=False,
        seed=0,
        use_piece_mask=args.use_piece_mask,
        piece_mask_dir=args.piece_mask_dir,
        refine_real_mask=args.refine_real_mask,
    )

    if args.limit and args.limit > 0:
        ds = Subset(ds, range(min(args.limit, len(ds))))

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = []
    for ckpt_arg in args.ckpt:
        ckpt_path = _resolve_ckpt(ckpt_arg)
        print(f"\n=== Evaluating: {ckpt_path} ===")
        G = _load_generator(ckpt_path, device)

        mean_l1, met = validate_l1(G, dl, device, compute_sharpness=True)

        ps = float(met.get("piece_sharpness", float("nan")))
        bs = float(met.get("bg_sharpness", float("nan")))
        rp = float(met.get("real_piece_sharpness", float("nan")))
        rb = float(met.get("real_bg_sharpness", float("nan")))

        out = {
            "ckpt": str(ckpt_path),
            "mean_L1": float(mean_l1),
            "piece_sharp": ps,
            "bg_sharp": bs,
            "delta": ps - bs,
            "GT_piece": rp,
            "GT_bg": rb,
            "GT_delta": rp - rb,
            "piece_over_bg_ratio": (ps / (bs + 1e-12)) if (ps == ps and bs == bs) else None,
        }
        results.append(out)

        print(f"mean_L1={out['mean_L1']:.4f}")
        print(f"piece_sharp={out['piece_sharp']:.6f}  bg_sharp={out['bg_sharp']:.6f}  delta={out['delta']:.6f}")
        print(f"GT_piece={out['GT_piece']:.6f}  GT_bg={out['GT_bg']:.6f}  GT_delta={out['GT_delta']:.6f}")

        if out["piece_sharp"] != out["piece_sharp"]:
            print("[WARN] Sharpness metrics are NaN. This usually means the val loader isn't providing 'mask'.")
            print("       Try adding --use_piece_mask and ensure --piece_mask_dir is correct + masks exist.")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
