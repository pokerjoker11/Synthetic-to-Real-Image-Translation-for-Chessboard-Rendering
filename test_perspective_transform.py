#!/usr/bin/env python3
"""Visualize perspective transform applied to synthetic images next to real images."""

import sys
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).parent

# Import the dataset to use the same transform logic
sys.path.insert(0, str(REPO_ROOT))
from src.datasets.pairs_dataset import PairedChessDataset

def apply_perspective_transform(img: Image.Image, perspective_max_tilt: float = 0.05, 
                                 pitch_amount: float = None, side_offset_amount: float = None, 
                                 seed: int = 42) -> Image.Image:
    """Apply the same perspective transform as used in training.
    
    Args:
        pitch_amount: Vertical tilt (pitch) - normalized 0-1. If None, random.
        side_offset_amount: Horizontal offset (camera to side) - normalized 0-1. If None, random.
    """
    rng = random.Random(seed)
    w, h = img.size
    
    # Enlarge by ~15% to accommodate tilt
    scale_factor = 1.15
    enlarged_w = int(w * scale_factor)
    enlarged_h = int(h * scale_factor)
    
    # Resize to larger size
    enlarged = TF.resize(img, [enlarged_h, enlarged_w], interpolation=Image.BICUBIC)
    
    # Compute perspective distortion (camera tilt in X-Z axes + horizontal offset)
    max_perspective_shift = int(min(w, h) * perspective_max_tilt)
    
    # Component 1: Camera pitch (tilt forward/backward) - vertical perspective
    if pitch_amount is not None:
        # Normalized 0-1 -> 0 to +max for forward tilt (bottom edge larger)
        # Use more of the max shift for visibility
        pitch_shift = pitch_amount * max_perspective_shift * 1.5  # Amplify for visibility
    else:
        pitch_shift = rng.uniform(-max_perspective_shift, max_perspective_shift)
    
    # Component 2: Camera roll (tilt left/right) - horizontal perspective
    roll_shift = 0  # Keep roll minimal for clearer visualization
    
    # Component 3: Camera horizontal offset (camera positioned to side of board)
    # This creates asymmetric perspective where one side shows more "side view" of pieces
    side_offset_strength = 0.12  # Max horizontal offset (12% of image size - increased for visibility)
    if side_offset_amount is not None:
        # Normalized 0-1 -> 0 to +max
        # Make positive offset (camera to left, reveals right sides)
        side_offset = side_offset_amount * side_offset_strength * min(w, h)
    else:
        side_offset = rng.uniform(-side_offset_strength, side_offset_strength) * min(w, h)
    
    # Original corners
    orig_corners = [
        [0, 0],  # top-left
        [enlarged_w, 0],  # top-right
        [enlarged_w, enlarged_h],  # bottom-right
        [0, enlarged_h],  # bottom-left
    ]
    
    # New corners with perspective distortion (combine pitch/roll + side offset)
    new_corners = [
        [orig_corners[0][0] - roll_shift - side_offset, orig_corners[0][1] - pitch_shift],  # top-left
        [orig_corners[1][0] + roll_shift - side_offset, orig_corners[1][1] - pitch_shift],  # top-right
        [orig_corners[2][0] + roll_shift + side_offset, orig_corners[2][1] + pitch_shift],  # bottom-right
        [orig_corners[3][0] - roll_shift + side_offset, orig_corners[3][1] + pitch_shift],  # bottom-left
    ]
    
    # Compute perspective transform coefficients
    try:
        from numpy.linalg import solve
        
        A = np.zeros((8, 8))
        b = np.zeros(8)
        
        for i, ((x, y), (xp, yp)) in enumerate(zip(orig_corners, new_corners)):
            A[i*2] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp]
            b[i*2] = xp
            A[i*2+1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp]
            b[i*2+1] = yp
        
        coeffs = solve(A, b)
        perspective_coeffs = tuple(coeffs)
        
        # Apply perspective transform
        tilted = enlarged.transform(
            enlarged.size, Image.PERSPECTIVE, perspective_coeffs,
            Image.BICUBIC, fillcolor=(0, 0, 0)
        )
        
        # Crop back to original size (centered)
        crop_w = min(w, tilted.size[0])
        crop_h = min(h, tilted.size[1])
        crop_x = (tilted.size[0] - crop_w) // 2
        crop_y = (tilted.size[1] - crop_h) // 2
        
        result = TF.crop(tilted, crop_y, crop_x, h, w)
        return result
    except Exception as e:
        print(f"Error applying perspective: {e}")
        return img


def main():
    # Load dataset
    csv_path = REPO_ROOT / "data" / "splits_rect" / "train_clean.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1
    
    # Load dataset with augmentation (so synthetic gets tilted)
    dataset = PairedChessDataset(
        csv_path=csv_path,
        repo_root=REPO_ROOT,
        image_size=256,
        load_size=256,
        train=True,  # Training mode - applies augmentation
        perspective_prob=1.0,  # Always apply perspective transform
        perspective_max_tilt=0.05,
    )
    
    print(f"Loaded dataset with {len(dataset)} samples")
    print("Creating comparison images...\n")
    
    # Create output directory
    output_dir = REPO_ROOT / "results" / "perspective_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate multiple examples: Tilted Synth | Real
    num_examples = 15
    # Sample diverse indices from across the dataset
    rng = random.Random(42)
    max_idx = min(len(dataset), 1000)  # Sample from first 1000 samples
    indices = sorted(rng.sample(range(max_idx), num_examples))
    
    for example_idx, i in enumerate(indices[:num_examples]):
        if i >= len(dataset):
            continue
        
        # Load synthetic with tilt applied and real image (both from dataset with augmentation)
        sample = dataset[i]
        # Convert from [-1, 1] tensor to [0, 255] uint8 image
        # Dataset normalizes to [-1, 1], so we need to convert back properly
        def tensor_to_image(tensor):
            arr = tensor.numpy().transpose(1, 2, 0)  # (H, W, C)
            arr = (arr + 1.0) * 0.5  # [-1, 1] -> [0, 1]
            arr = np.clip(arr, 0, 1)  # Clamp to valid range
            arr = (arr * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
            return Image.fromarray(arr)
        
        synth_tilted = tensor_to_image(sample['A'])
        real_img = tensor_to_image(sample['B'])
        
        # Create 2-image comparison: Tilted Synth | Real
        w, h = synth_tilted.size
        spacing = 10
        canvas_w = w * 2 + spacing
        canvas_h = h
        
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        
        # Left: Synthetic with tilt applied
        canvas.paste(synth_tilted, (0, 0))
        
        # Right: Real image
        canvas.paste(real_img, (w + spacing, 0))
        
        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        labels = [
            ("Tilted Synth", 0),
            ("Real Image", w + spacing)
        ]
        
        for label, x_offset in labels:
            draw.text((x_offset + 5, 5), label, fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        
        output_path = output_dir / f"example_{example_idx:03d}_tilted_synth_vs_real.png"
        canvas.save(output_path)
        print(f"  [{example_idx+1}] Saved: {output_path.name} (sample {i})")
    
    print(f"\n{'='*60}")
    print(f"Comparison images saved to: {output_dir}")
    print(f"\nThe images show:")
    print(f"  - Left: Original synthetic (perfect overhead)")
    print(f"  - Middle: Synthetic with perspective transform (3D tilt)")
    print(f"  - Right: Real image (natural camera tilt)")
    print(f"\nCheck if the tilted synthetic images show trapezoidal distortion")
    print(f"similar to the real images (squares becoming trapezoids).")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
