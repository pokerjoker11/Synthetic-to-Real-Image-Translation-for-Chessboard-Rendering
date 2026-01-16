# scripts/smoke_test_model_forward.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.datasets.pairs_dataset import PairedChessDataset
from src.models.pix2pix_nets import UNetGenerator, PatchDiscriminator, init_weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    ds = PairedChessDataset("data/splits_rect/train.csv", repo_root=".", image_size=256, train=True, seed=123)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    A = batch["A"].to(device)  # synth
    B = batch["B"].to(device)  # real

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    init_weights(G)
    init_weights(D)

    with torch.no_grad():
        fake_B = G(A)
        pred_real = D(A, B)
        pred_fake = D(A, fake_B)

    print(f"[OK] A shape       : {tuple(A.shape)}")
    print(f"[OK] B shape       : {tuple(B.shape)}")
    print(f"[OK] G(A) shape    : {tuple(fake_B.shape)}")
    print(f"[OK] D(A,B) shape  : {tuple(pred_real.shape)}")
    print(f"[OK] D(A,G(A))     : {tuple(pred_fake.shape)}")

    # sanity ranges
    print(f"[INFO] G(A) min/max: {float(fake_B.min()):.3f} / {float(fake_B.max()):.3f}")


if __name__ == "__main__":
    main()
