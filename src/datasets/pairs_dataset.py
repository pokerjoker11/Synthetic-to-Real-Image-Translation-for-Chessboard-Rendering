# src/datasets/pairs_dataset.py
from __future__ import annotations

import csv
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
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
        # Enhanced augmentation parameters
        color_jitter_prob: float = 0.5,
        color_jitter_strength: float = 0.15,  # Increased from 0.1 for more realistic lighting variation
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
        perspective_prob: float = 0.5,
        perspective_max_tilt: float = 0.05,  # Max perspective distortion (0.05 = ~5% shift at corners)
        # Piece mask parameters
        piece_mask_dir: Optional[str | Path] = None,
        use_piece_mask: bool = False,
        # Option 1: refine coarse square masks using edges in the *real* image
        refine_real_mask: bool = False,
        refine_quantile: float = 0.85,
        refine_sigma: float = 8.0,
        refine_border: int = 2,
        refine_strength: float = 1.0,
        refine_occ_thr: float = 0.08,
        refine_spill_px: int = 8,

    ):
        self.csv_path = Path(csv_path)
        self.repo_root = Path(repo_root)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.hflip_prob = float(hflip_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter_strength = float(color_jitter_strength)
        self.noise_prob = float(noise_prob)
        self.noise_std = float(noise_std)
        self.perspective_prob = float(perspective_prob)
        self.perspective_max_tilt = float(perspective_max_tilt)

        # Common Pix2Pix trick: resize a bit larger then random-crop.
        # For 256 -> 286; for 512 -> 572.
        if load_size is None:
            self.load_size = self.image_size + (30 if self.image_size <= 512 else 60)
        else:
            self.load_size = int(load_size)

        # Deterministic randomness per-worker: base seed + index
        self.base_seed = int(seed)
        
        # Piece mask support
        self.piece_mask_dir = Path(piece_mask_dir).resolve() if piece_mask_dir else None
        self.use_piece_mask = bool(use_piece_mask) and (self.piece_mask_dir is not None)

        # Option 1 (mask refinement): shift attention inside each occupied square toward
        # the actual piece pixels, using real-image edges.
        self.refine_real_mask = bool(refine_real_mask)
        self.refine_quantile = float(refine_quantile)
        self.refine_sigma = float(refine_sigma)
        self.refine_border = int(refine_border)
        self.refine_strength = float(refine_strength)
        self.refine_occ_thr = float(refine_occ_thr)
        self.refine_spill_px = int(refine_spill_px)

        # Sobel kernels (dataset runs on CPU)
        self._sobel_kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self._sobel_ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.rows = self._read_rows(self.csv_path)

    def _estimate_perspective_from_image(self, img: Image.Image) -> Optional[Tuple[float, float, float]]:
        """
        Estimate perspective tilt from real image by analyzing board geometry.
        Returns (pitch_shift, roll_shift, side_offset) or None if no significant tilt detected.
        
        Heuristic: detect if board squares appear trapezoidal (perspective) vs square (overhead).
        """
        # Convert to grayscale numpy array
        gray = np.array(img.convert('L'))
        h, w = gray.shape
        
        # Sample edge regions to detect perspective distortion
        # If perfectly overhead, opposite edges should be similar
        # If tilted, one edge will appear different (smaller/larger, different pattern)
        
        margin = int(min(w, h) * 0.15)  # Sample from board area, avoid extreme edges
        
        # Sample top and bottom edges (detect vertical perspective/pitch)
        top_region = gray[margin:margin+10, margin:w-margin]
        bottom_region = gray[h-margin-10:h-margin, margin:w-margin]
        
        # Sample left and right edges (detect horizontal perspective/roll + side offset)
        left_region = gray[margin:h-margin, margin:margin+10]
        right_region = gray[margin:h-margin, w-margin-10:w-margin]
        
        # Calculate statistics to detect perspective
        top_mean = np.mean(top_region) if top_region.size > 0 else 0
        bottom_mean = np.mean(bottom_region) if bottom_region.size > 0 else 0
        left_mean = np.mean(left_region) if left_region.size > 0 else 0
        right_mean = np.mean(right_region) if right_region.size > 0 else 0
        
        # Detect perspective tilt: if edges differ significantly, there's perspective
        vertical_diff = abs(top_mean - bottom_mean)
        horizontal_diff = abs(left_mean - right_mean)
        
        # Threshold: if differences are small, image is likely overhead
        # Use variance as additional signal - perspective creates asymmetry
        top_var = np.var(top_region) if top_region.size > 0 else 0
        bottom_var = np.var(bottom_region) if bottom_region.size > 0 else 0
        left_var = np.var(left_region) if left_region.size > 0 else 0
        right_var = np.var(right_region) if right_region.size > 0 else 0
        
        # Lowered thresholds to catch more subtle perspective (was 5, 5, 200)
        # Most real chess images have at least slight perspective
        has_perspective = (vertical_diff > 3) or (horizontal_diff > 3) or (abs(top_var - bottom_var) > 150) or (abs(left_var - right_var) > 150)
        
        # Fallback: Most real chess images have slight perspective even if detection is uncertain
        # Apply small default tilt if detection is borderline
        is_borderline = (vertical_diff > 2) or (horizontal_diff > 2) or (abs(top_var - bottom_var) > 100)
        
        if not has_perspective:
            # If borderline, apply small default tilt (most real images have slight tilt)
            if is_borderline:
                max_shift = int(min(w, h) * self.perspective_max_tilt)
                pitch_shift = max_shift * 0.25  # Small default forward tilt
                roll_shift = 0
                side_offset_strength = 0.02
                side_offset = side_offset_strength * 0.4 * min(w, h)
                return (float(pitch_shift), float(roll_shift), float(side_offset))
            return None  # Image appears truly overhead, no tilt needed
        
        # Estimate tilt parameters based on detected differences
        max_shift = int(min(w, h) * self.perspective_max_tilt)
        
        # Pitch: forward tilt if bottom appears larger/different (typical camera position)
        if bottom_mean > top_mean or bottom_var > top_var:
            pitch_shift = max_shift * 0.4  # Forward tilt
        else:
            pitch_shift = max_shift * 0.2  # Slight tilt
        
        # Roll: minimal for chess images
        roll_shift = 0
        
        # Side offset: detect if one side shows more detail (camera offset)
        side_offset_strength = 0.03  # 3% for subtle offset
        right_var = np.var(right_region) if right_region.size > 0 else 0
        left_var = np.var(left_region) if left_region.size > 0 else 0
        if right_var > left_var or right_mean < left_mean:
            # Right side shows more detail (camera to left)
            side_offset = side_offset_strength * 0.6 * min(w, h)
        else:
            side_offset = side_offset_strength * 0.3 * min(w, h)
        
        return (float(pitch_shift), float(roll_shift), float(side_offset))

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
    
    def _load_piece_mask(self, real_path: Path, synth_path: Path) -> Optional[Image.Image]:
        """Load piece mask if available.
        Supports either:
        - synth-stem masks: {synth_stem}.png
        - real-stem masks:  {real_stem}.png   (recommended, stable)
        """
        if (not self.use_piece_mask) or (self.piece_mask_dir is None):
            return None

        candidates = [
            self.piece_mask_dir / f"{real_path.stem}.png",
            self.piece_mask_dir / f"{synth_path.stem}.png",
            self.piece_mask_dir / f"{real_path.name}",  # if someone saved masks as .jpg (rare)
        ]

        for mp in candidates:
            if mp.exists():
                return Image.open(mp).convert("L")

        return None




    def _sobel_mag(self, gray01: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude.

        Args:
            gray01: (H,W) float in [0,1]

        Returns:
            (H,W) gradient magnitude
        """
        x = gray01.unsqueeze(0).unsqueeze(0)
        gx = F.conv2d(x, self._sobel_kx, padding=1)
        gy = F.conv2d(x, self._sobel_ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)[0, 0]
    def _refine_mask_edges(self, real_t: torch.Tensor, synth_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        """Refine a coarse square-occupancy mask using real-image edges, with spill support.

        The coarse mask tells you which squares contain pieces (from FEN). In real images, due to
        camera angle and piece height, visible piece pixels can spill into neighboring squares.

        This refinement:
        1) Builds a clean 8x8 occupancy grid from the coarse mask using per-square *mean* coverage
           (prevents a single leaked pixel from marking an empty square as occupied).
        2) Rasterizes occupied squares into an "allowed" pixel region and dilates it by
           refine_spill_px pixels to include projected spillover.
        3) For each occupied square, estimates a center-of-mass of strong edge pixels and places
           a soft Gaussian around it *over the expanded ROI*, then clips to the allowed region.

        Args:
            real_t:  (3,H,W) in [-1,1]
            synth_t: (3,H,W) in [-1,1] (not used currently, kept for future Option 2 alignment)
            mask_t:  (1,H,W) binary coarse mask

        Returns:
            refined mask: (1,H,W) in [0,1]
        """
        H, W = int(mask_t.shape[-2]), int(mask_t.shape[-1])
        if H < 8 or W < 8:
            return mask_t

        # grayscale in [0,1]
        real01 = (real_t + 1.0) * 0.5
        if real01.shape[0] == 3:
            gray = 0.299 * real01[0] + 0.587 * real01[1] + 0.114 * real01[2]
        else:
            gray = real01.mean(dim=0)
        gray = gray.clamp(0.0, 1.0)

        grad = self._sobel_mag(gray)

        # 8x8 grid params
        sq_h = max(1, H // 8)
        sq_w = max(1, W // 8)
        border = max(0, min(self.refine_border, min(sq_h, sq_w) // 4))
        sigma = max(1.0, float(self.refine_sigma))
        q = float(self.refine_quantile)
        q = min(max(q, 0.5), 0.99)

        spill = max(0, int(getattr(self, 'refine_spill_px', 0)))
        occ_thr = float(getattr(self, 'refine_occ_thr', 0.08))

        m = mask_t[0].float()

        # --- per-square occupancy grid using MEAN (robust to tiny leaks) ---
        occ = torch.zeros((8, 8), dtype=torch.float32)
        for r in range(8):
            y0 = r * sq_h
            y1 = (r + 1) * sq_h if r < 7 else H
            for c in range(8):
                x0 = c * sq_w
                x1 = (c + 1) * sq_w if c < 7 else W
                sq = m[y0:y1, x0:x1]
                if sq.numel() == 0:
                    continue
                if float(sq.mean().item()) >= occ_thr:
                    occ[r, c] = 1.0

        # --- allowed region: exact occupied squares, then pixel-dilate by spill ---
        allowed = torch.zeros((H, W), dtype=torch.float32)
        for r in range(8):
            y0 = r * sq_h
            y1 = (r + 1) * sq_h if r < 7 else H
            for c in range(8):
                if float(occ[r, c].item()) < 0.5:
                    continue
                x0 = c * sq_w
                x1 = (c + 1) * sq_w if c < 7 else W
                allowed[y0:y1, x0:x1] = 1.0

        if spill > 0:
            k = 2 * spill + 1
            allowed = F.max_pool2d(allowed.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=spill)[0, 0]

        refined = torch.zeros_like(mask_t)

        # --- refine each occupied square, but paste over expanded ROI ---
        for r in range(8):
            y0 = r * sq_h
            y1 = (r + 1) * sq_h if r < 7 else H
            for c in range(8):
                if float(occ[r, c].item()) < 0.5:
                    continue

                x0 = c * sq_w
                x1 = (c + 1) * sq_w if c < 7 else W

                # expanded ROI for spill
                ry0 = max(0, y0 - spill)
                ry1 = min(H, y1 + spill)
                rx0 = max(0, x0 - spill)
                rx1 = min(W, x1 + spill)

                # inner region inside the ORIGINAL square (avoid board lines)
                y0i = max(ry0, y0 + border)
                y1i = min(ry1, y1 - border)
                x0i = max(rx0, x0 + border)
                x1i = min(rx1, x1 - border)

                # fallback: square center
                cy = (y0 + y1 - 1) * 0.5
                cx = (x0 + x1 - 1) * 0.5

                if (y1i > y0i) and (x1i > x0i):
                    g = grad[y0i:y1i, x0i:x1i].clamp_min(0.0)
                    flat = g.flatten()
                    if flat.numel() > 0 and float(flat.sum().item()) > 1e-8:
                        k_top = int(max(1, round(flat.numel() * (1.0 - q))))
                        vals, _ = torch.topk(flat, k_top, largest=True)
                        thr = vals.min()
                        w = g * (g >= thr).float()
                        sw = w.sum()
                        if float(sw.item()) > 1e-8:
                            yy = torch.arange(y0i, y1i, dtype=torch.float32)
                            xx = torch.arange(x0i, x1i, dtype=torch.float32)
                            try:
                                YY, XX = torch.meshgrid(yy, xx, indexing='ij')
                            except TypeError:
                                YY, XX = torch.meshgrid(yy, xx)
                            cy = float((YY * w).sum().item() / sw.item())
                            cx = float((XX * w).sum().item() / sw.item())

                # Gaussian over expanded ROI
                ys = torch.arange(ry0, ry1, dtype=torch.float32)
                xs = torch.arange(rx0, rx1, dtype=torch.float32)
                try:
                    YY2, XX2 = torch.meshgrid(ys, xs, indexing='ij')
                except TypeError:
                    YY2, XX2 = torch.meshgrid(ys, xs)

                gauss = torch.exp(-((YY2 - cy) ** 2 + (XX2 - cx) ** 2) / (2.0 * sigma * sigma))
                gauss = gauss / gauss.max().clamp_min(1e-6)

                refined[0, ry0:ry1, rx0:rx1] = torch.maximum(refined[0, ry0:ry1, rx0:rx1], gauss)

        # clip to allowed region (prevents far-away empty squares)
        refined = refined * allowed.unsqueeze(0)

        strength = float(self.refine_strength)
        strength = min(max(strength, 0.0), 1.0)
        if strength >= 0.999:
            out = refined
        elif strength <= 1e-6:
            out = mask_t
        else:
            out = (1.0 - strength) * mask_t + strength * refined

        return out.clamp(0.0, 1.0)


    def _resolve(self, rel_or_abs: str) -> Path:
        # CSVs may contain Windows-style backslashes; normalize for cross-platform use
        rel_or_abs = (rel_or_abs or '').replace('\\', '/')
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
        
        # Load piece mask (optional)
        mask_img = self._load_piece_mask(real_path, synth_path)

        # Resize to a common "load_size"
        real_img = TF.resize(real_img, [self.load_size, self.load_size], interpolation=Image.BICUBIC)
        synth_img = TF.resize(synth_img, [self.load_size, self.load_size], interpolation=Image.BICUBIC)
        # Mask uses nearest interpolation (preserves binary values)
        if mask_img is not None:
            mask_img = TF.resize(mask_img, [self.load_size, self.load_size], interpolation=Image.NEAREST)

        # Randomness: deterministic per index (so multi-worker doesn't go chaotic)
        rng = random.Random(self.base_seed + idx)

        if self.train:
            # No random cropping - preserve full board
            # Real images show the whole board with minimal cropping
            # When load_size == image_size, crop_range = 0, so we just center crop
            crop_range = self.load_size - self.image_size
            if crop_range > 0:
                # Small random crop only if load_size > image_size (shouldn't happen now)
                top = rng.randint(0, crop_range) if crop_range > 0 else 0
                left = rng.randint(0, crop_range) if crop_range > 0 else 0
                real_img = TF.crop(real_img, top, left, self.image_size, self.image_size)
                synth_img = TF.crop(synth_img, top, left, self.image_size, self.image_size)
                if mask_img is not None:
                    mask_img = TF.crop(mask_img, top, left, self.image_size, self.image_size)
            else:
                # No cropping - images are already at correct size
                # Just ensure they're exactly image_size
                if real_img.size[0] != self.image_size or real_img.size[1] != self.image_size:
                    real_img = TF.resize(real_img, [self.image_size, self.image_size], interpolation=Image.BICUBIC)
                if synth_img.size[0] != self.image_size or synth_img.size[1] != self.image_size:
                    synth_img = TF.resize(synth_img, [self.image_size, self.image_size], interpolation=Image.BICUBIC)
                if mask_img is not None:
                    if mask_img.size[0] != self.image_size or mask_img.size[1] != self.image_size:
                        mask_img = TF.resize(mask_img, [self.image_size, self.image_size], interpolation=Image.NEAREST)

            # Paired horizontal flip
            if rng.random() < self.hflip_prob:
                real_img = TF.hflip(real_img)
                synth_img = TF.hflip(synth_img)
                if mask_img is not None:
                    mask_img = TF.hflip(mask_img)
            

        real_t = TF.to_tensor(real_img)
        synth_t = TF.to_tensor(synth_img)
        
        # Add Gaussian noise to synthetic (only during training, after tensor conversion)
        if self.train and rng.random() < self.noise_prob:
            noise = torch.randn_like(synth_t) * self.noise_std
            synth_t = torch.clamp(synth_t + noise, 0.0, 1.0)

        # Normalize to [-1, 1] (matches tanh generator output convention)
        real_t = real_t * 2.0 - 1.0
        synth_t = synth_t * 2.0 - 1.0
        
        # Convert mask to tensor [1, H, W] in {0, 1}
        if mask_img is not None:
            # Mask is uint8 {0, 255} -> convert to float {0, 1}
            mask_t = TF.to_tensor(mask_img)  # [1, H, W] in [0, 1]
            mask_t = (mask_t > 0.5).float()  # Binarize: > 128 -> 1.0, else 0.0
        else:
            # Default to all ones (no weighting)
            mask_t = torch.ones((1, self.image_size, self.image_size), dtype=torch.float32)

        # Option 1: refine coarse square mask using real-image edges
        if self.use_piece_mask and self.refine_real_mask and (mask_img is not None):
            mask_t = self._refine_mask_edges(real_t, synth_t, mask_t)

        return {
            "A": synth_t,  # input
            "B": real_t,   # target
            "mask": mask_t,  # piece weight mask [1, H, W] in [0, 1]
            "fen": row.fen,
            "viewpoint": row.viewpoint,
            "game": row.game,
            "frame": row.frame,
            "real_path": str(real_path),
            "synth_path": str(synth_path),
        }
