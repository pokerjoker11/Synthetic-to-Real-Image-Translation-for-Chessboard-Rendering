# src/datasets/pairs_dataset.py
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


@dataclass
class PairRow:
    real: str
    synth: str
    fen: str
    viewpoint: str
    game: str
    frame: int


class PairedChessDataset(Dataset):
    """
    Paired dataset for Pix2Pix-style training.
    A = synth (input), B = real (target)

    Key property: all random transforms are applied identically to both images.
    """
    def __init__(
        self,
        csv_path: str | Path,
        repo_root: str | Path = ".",
        image_size: int = 256,
        load_size: Optional[int] = None,
        train: bool = True,
        hflip_prob: float = 0.5,
        seed: int = 0,
        mask_dir: str | Path = "data/masks_manual",
        require_masks: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.repo_root = Path(repo_root)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.hflip_prob = float(hflip_prob)
        self.mask_dir = self._resolve(mask_dir)
        self.require_masks = require_masks

        # Common Pix2Pix trick: resize a bit larger then random-crop.
        # For 256 -> 286; for 512 -> 572.
        if load_size is None:
            self.load_size = self.image_size + (30 if self.image_size <= 512 else 60)
        else:
            self.load_size = int(load_size)

        # Deterministic randomness per-worker: base seed + index
        self.base_seed = int(seed)

        self.rows = self._read_rows(self.csv_path)
    
    def _open_mask(self, p: Path) -> Image.Image:
        # grayscale mask (0..255)
        return Image.open(p).convert("L")

    def _read_rows(self, p: Path) -> list[PairRow]:
        if not p.exists():
            raise FileNotFoundError(f"Missing pairs csv: {p}")

        rows: list[PairRow] = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(PairRow(
                    real=r["real"],
                    synth=r["synth"],
                    fen=r.get("fen", ""),
                    viewpoint=r.get("viewpoint", "white"),
                    game=r.get("game", ""),
                    frame=int(r.get("frame", "0") or 0),
                ))
        if not rows:
            raise ValueError(f"pairs csv has no rows: {p}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _open_rgb(self, path: Path) -> Image.Image:
        img = Image.open(path)
        return img.convert("RGB")

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (self.repo_root / p).resolve()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        real_path = self._resolve(row.real)
        synth_path = self._resolve(row.synth)

        if not real_path.exists():
            raise FileNotFoundError(f"Missing real image: {real_path}")
        if not synth_path.exists():
            raise FileNotFoundError(f"Missing synth image: {synth_path}")

        # Load
        real_img = self._open_rgb(real_path)
        synth_img = self._open_rgb(synth_path)

        mask_path = self.mask_dir / (real_path.stem + ".png")
        if not mask_path.exists():
            if self.require_masks:
                raise FileNotFoundError(f"Missing mask: {mask_path} (for real: {real_path.name})")
            mask_img = Image.new("L", real_img.size, color=0)
        else:
            mask_img = self._open_mask(mask_path)

        # Resize to a common "load_size"
        real_img = TF.resize(real_img, [self.load_size, self.load_size], interpolation=Image.BICUBIC)
        synth_img = TF.resize(synth_img, [self.load_size, self.load_size], interpolation=Image.BICUBIC)
        mask_img = TF.resize(mask_img, [self.load_size, self.load_size], interpolation=Image.NEAREST)


        # Randomness: deterministic per index (so multi-worker doesn't go chaotic)
        rng = random.Random(self.base_seed + idx)

        if self.train:
            # Paired random crop
            if self.load_size < self.image_size:
                raise ValueError("load_size must be >= image_size")

            max_xy = self.load_size - self.image_size
            top = rng.randint(0, max_xy)
            left = rng.randint(0, max_xy)

            real_img = TF.crop(real_img, top, left, self.image_size, self.image_size)
            synth_img = TF.crop(synth_img, top, left, self.image_size, self.image_size)
            mask_img = TF.crop(mask_img, top, left, self.image_size, self.image_size)

            # Paired horizontal flip
            if rng.random() < self.hflip_prob:
                real_img = TF.hflip(real_img)
                synth_img = TF.hflip(synth_img)
                mask_img = TF.hflip(mask_img)
        else:
            # Center crop for val
            real_img = TF.center_crop(real_img, [self.image_size, self.image_size])
            synth_img = TF.center_crop(synth_img, [self.image_size, self.image_size])
            mask_img = TF.center_crop(mask_img, [self.image_size, self.image_size])

        # To tensor in [0,1]
        real_t = TF.to_tensor(real_img)
        synth_t = TF.to_tensor(synth_img)

        # Normalize to [-1, 1] (matches tanh generator output convention)
        real_t = real_t * 2.0 - 1.0
        synth_t = synth_t * 2.0 - 1.0

        mask_t = TF.to_tensor(mask_img)          # (1,H,W) in [0,1]
        mask_t = (mask_t > 0.5).float()          # binarize
        
        return {
            "A": synth_t,  # input
            "B": real_t,   # target
            "fen": row.fen,
            "viewpoint": row.viewpoint,
            "game": row.game,
            "frame": row.frame,
            "real_path": str(real_path),
            "synth_path": str(synth_path),
        }
