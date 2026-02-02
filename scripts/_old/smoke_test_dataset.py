# scripts/smoke_test_dataset.py
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so `import src...` works reliably on Windows
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.datasets.pairs_dataset import PairedChessDataset


def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x + 1.0) / 2.0


def main():
    out_dir = Path("results/dataset_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path("data/splits/train.csv")
    val_csv = Path("data/splits/val.csv")

    ds_train = PairedChessDataset(train_csv, repo_root=".", image_size=256, train=True, seed=123)
    ds_val = PairedChessDataset(val_csv, repo_root=".", image_size=256, train=False, seed=123)

    dl = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    A = batch["A"]  # synth
    B = batch["B"]  # real

    save_image(denorm(A), out_dir / "train_A_synth.png", nrow=4)
    save_image(denorm(B), out_dir / "train_B_real.png", nrow=4)

    # side-by-side: concat width-wise
    side = torch.cat([denorm(A), denorm(B)], dim=3)
    save_image(side, out_dir / "train_side_by_side.png", nrow=1)

    # val
    dlv = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0)
    vb = next(iter(dlv))
    Av = vb["A"]
    Bv = vb["B"]
    sidev = torch.cat([denorm(Av), denorm(Bv)], dim=3)
    save_image(sidev, out_dir / "val_side_by_side.png", nrow=1)

    print("[OK] Wrote:")
    print(" - results/dataset_smoke/train_A_synth.png")
    print(" - results/dataset_smoke/train_B_real.png")
    print(" - results/dataset_smoke/train_side_by_side.png")
    print(" - results/dataset_smoke/val_side_by_side.png")
    print(f"[INFO] train size={len(ds_train)} val size={len(ds_val)}")


if __name__ == "__main__":
    main()
