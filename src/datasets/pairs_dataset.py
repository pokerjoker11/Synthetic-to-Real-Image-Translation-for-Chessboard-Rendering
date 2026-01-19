# src/datasets/pairs_dataset.py
from __future__ import annotations

import csv
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Tuple

import torch
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
    
    def _load_piece_mask(self, synth_path: Path) -> Optional[Image.Image]:
        """Load piece mask if available. Mask file has same stem as synthetic image."""
        if not self.use_piece_mask or self.piece_mask_dir is None:
            return None
        
        # Get mask path: same stem as synth image in piece_mask_dir
        mask_stem = synth_path.stem
        mask_path = self.piece_mask_dir / f"{mask_stem}.png"
        
        if not mask_path.exists():
            return None
        
        # Load as grayscale (single channel)
        mask_img = Image.open(mask_path).convert("L")
        return mask_img

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
        mask_img = self._load_piece_mask(synth_path)

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
            
            # Enhanced augmentation (applied after crop to save computation)
            # Perspective transform: detect tilt in real image and match it in synthetic
            # Note: Horizontal flip (above) already provides black/white perspective variation
            # We match the detected perspective for consistency, but can add small random variation
            # Detect tilt from real image and apply matching transform to synthetic
            real_perspective = self._estimate_perspective_from_image(real_img)
            
            # Apply perspective if detected
            # Option 1: Always match exactly (most consistent)
            # Option 2: Match with small random variation (more diversity, still aligned)
            # Currently using Option 1 - always match exactly for best alignment
            if real_perspective is not None:
                # Real image has perspective tilt - apply matching tilt to synthetic
                pitch_shift, roll_shift, side_offset = real_perspective
                
                # Optional: Add small random variation (±10%) to detected perspective for diversity
                # This maintains alignment while adding slight variation
                # Uncomment to enable:
                # variation_factor = rng.uniform(0.9, 1.1)  # ±10% variation
                # pitch_shift *= variation_factor
                # side_offset *= variation_factor
                
                # Apply perspective transform to synthetic image only
                # Real images already have natural camera tilt, so we don't transform them
                w, h = synth_img.size
                # Enlarge by ~25% to accommodate tilt and ensure full board fits without borders
                scale_factor = 1.25
                enlarged_w = int(w * scale_factor)
                enlarged_h = int(h * scale_factor)
                
                # Resize synthetic to larger size (bicubic for quality)
                # Real image stays unchanged
                synth_enlarged = TF.resize(synth_img, [enlarged_h, enlarged_w], interpolation=Image.BICUBIC)
                
                # Original corners (source points)
                orig_corners = [
                    [0, 0],  # top-left
                    [enlarged_w, 0],  # top-right
                    [enlarged_w, enlarged_h],  # bottom-right
                    [0, enlarged_h],  # bottom-left
                ]
                
                # New corners with perspective distortion (destination points)
                # Combine pitch/roll tilt + horizontal offset
                # Horizontal offset creates asymmetric perspective: one side shifts more than opposite side
                new_corners = [
                    # Top-left: combine pitch, roll, and side offset
                    [orig_corners[0][0] - roll_shift - side_offset, orig_corners[0][1] - pitch_shift],
                    # Top-right: combine pitch, roll, and side offset (opposite direction)
                    [orig_corners[1][0] + roll_shift - side_offset, orig_corners[1][1] - pitch_shift],
                    # Bottom-right: same pattern, pitch moves down
                    [orig_corners[2][0] + roll_shift + side_offset, orig_corners[2][1] + pitch_shift],
                    # Bottom-left: same pattern
                    [orig_corners[3][0] - roll_shift + side_offset, orig_corners[3][1] + pitch_shift],
                ]
                
                # Compute perspective transform coefficients from 4 point pairs
                # PIL PERSPECTIVE mode: maps (x,y) -> ((ax+by+c)/(gx+hy+1), (dx+ey+f)/(gx+hy+1))
                try:
                    from numpy.linalg import solve
                    
                    A = np.zeros((8, 8))
                    b = np.zeros(8)
                    
                    for i, ((x, y), (xp, yp)) in enumerate(zip(orig_corners, new_corners)):
                        # First equation: xp = (ax + by + c) / (gx + hy + 1) => xp*(gx + hy + 1) = ax + by + c
                        # Rearranged: ax + by + c - gx*xp - hy*xp = xp
                        A[i*2] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp]
                        b[i*2] = xp
                        # Second equation: yp = (dx + ey + f) / (gx + hy + 1) => yp*(gx + hy + 1) = dx + ey + f
                        # Rearranged: dx + ey + f - gx*yp - hy*yp = yp
                        A[i*2+1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp]
                        b[i*2+1] = yp
                    
                    try:
                        coeffs = solve(A, b)
                        perspective_coeffs = tuple(coeffs)
                        
                        # Apply perspective transform
                        # Use brown fill color to match synthetic chess set's brown outer frame
                        # Apply perspective transform ONLY to synthetic image
                        # Real image already has correct perspective - keep it unchanged
                        brown_color = (101, 67, 33)  # Brown that matches chess set outer frame
                        
                        # Transform only synthetic image with brown fill
                        synth_tilted = synth_enlarged.transform(
                            synth_enlarged.size, Image.PERSPECTIVE, perspective_coeffs,
                            Image.BICUBIC, fillcolor=brown_color
                        )
                        
                        # Apply same perspective transform to mask if present (use black fill = 0 for background)
                        if mask_img is not None:
                            mask_enlarged = TF.resize(mask_img, [enlarged_h, enlarged_w], interpolation=Image.NEAREST)
                            mask_tilted = mask_enlarged.transform(
                                mask_enlarged.size, Image.PERSPECTIVE, perspective_coeffs,
                                Image.NEAREST, fillcolor=0  # Black = background in mask
                            )
                        
                        # Crop/Resize strategy: ensure board area fits in synthetic, brown border can be cropped if needed
                        # Real image stays unchanged - it already has the correct perspective
                        
                        # Crop synthetic image from center (can crop into brown border, but board must fit)
                        tilted_w, tilted_h = synth_tilted.size
                        
                        # Calculate crop bounds: center crop with margin for board
                        target_crop_size = max(w, h)  # At least target size
                        crop_w = min(tilted_w, target_crop_size + int(w * 0.1))  # 10% extra margin
                        crop_h = min(tilted_h, target_crop_size + int(h * 0.1))
                        
                        # Center crop (we can crop into brown border, but board should still fit)
                        crop_x = (tilted_w - crop_w) // 2
                        crop_y = (tilted_h - crop_h) // 2
                        crop_x = max(0, min(crop_x, tilted_w - crop_w))
                        crop_y = max(0, min(crop_y, tilted_h - crop_h))
                        
                        # Crop synthetic to board area (may crop some brown border)
                        synth_cropped = TF.crop(synth_tilted, crop_y, crop_x, crop_h, crop_w)
                        if mask_img is not None:
                            mask_cropped = TF.crop(mask_tilted, crop_y, crop_x, crop_h, crop_w)
                        
                        # Resize synthetic to exact target size
                        # Real image is already at target size, no changes needed
                        synth_img = TF.resize(synth_cropped, [h, w], interpolation=Image.BICUBIC)
                        if mask_img is not None:
                            mask_img = TF.resize(mask_cropped, [h, w], interpolation=Image.NEAREST)
                        # real_img stays unchanged
                    except (np.linalg.LinAlgError, ValueError):
                        # If solve fails or transform invalid, skip perspective (keep original)
                        pass
                except ImportError:
                    # Fallback: skip perspective if numpy not available
                    pass
            
            # Color jitter on synthetic only (simulate lighting variations)
            # This helps model learn to handle different lighting conditions
            if rng.random() < self.color_jitter_prob:
                # Apply subtle color variations (brightness, contrast, saturation)
                jitter = rng.uniform(1.0 - self.color_jitter_strength, 1.0 + self.color_jitter_strength)
                synth_img = ImageEnhance.Brightness(synth_img).enhance(jitter)
                jitter = rng.uniform(1.0 - self.color_jitter_strength * 0.5, 1.0 + self.color_jitter_strength * 0.5)
                synth_img = ImageEnhance.Contrast(synth_img).enhance(jitter)
                # Add saturation variation (simulates different lighting color temperatures)
                jitter = rng.uniform(1.0 - self.color_jitter_strength * 0.3, 1.0 + self.color_jitter_strength * 0.3)
                synth_img = ImageEnhance.Color(synth_img).enhance(jitter)
        else:
            # Center crop for val
            real_img = TF.center_crop(real_img, [self.image_size, self.image_size])
            synth_img = TF.center_crop(synth_img, [self.image_size, self.image_size])
            if mask_img is not None:
                mask_img = TF.center_crop(mask_img, [self.image_size, self.image_size])

        # To tensor in [0,1]
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

        return {
            "A": synth_t,  # input
            "B": real_t,   # target
            "mask": mask_t,  # piece mask [1, H, W] in {0, 1}
            "fen": row.fen,
            "viewpoint": row.viewpoint,
            "game": row.game,
            "frame": row.frame,
            "real_path": str(real_path),
            "synth_path": str(synth_path),
        }
