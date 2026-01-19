#!/usr/bin/env python3
"""
Generate piece masks from synthetic chess images.

Creates binary masks where white (255) = piece regions, black (0) = background.
Mask files have the same stem as synthetic images.

Method: Color-based segmentation
- White pieces: very light pixels (brightness > threshold)
- Black pieces: very dark pixels (brightness < threshold)
- Board/background: medium brightness (excluded)

Usage:
    python scripts/generate_piece_masks.py --train_csv data/splits_rect/train_clean.csv
    python scripts/generate_piece_masks.py --train_csv data/splits_rect/train_clean.csv --val_csv data/splits_rect/val_final.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def generate_mask_from_image(synth_img: Image.Image, 
                             white_threshold: float = 200.0,
                             black_threshold: float = 60.0) -> Image.Image:
    """
    Generate piece mask from synthetic image using color thresholding.
    
    Args:
        synth_img: PIL RGB image
        white_threshold: Brightness threshold for white pieces (0-255, default: 200)
        black_threshold: Brightness threshold for black pieces (0-255, default: 60)
    
    Returns:
        Binary mask PIL Image (L mode): 255 = piece, 0 = background
    """
    # Convert to numpy array
    img_array = np.array(synth_img.convert("RGB"))
    h, w, c = img_array.shape
    
    # Convert to grayscale (luminance)
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    
    # Piece mask: white pieces (very bright) OR black pieces (very dark)
    # White pieces: brightness > white_threshold
    # Black pieces: brightness < black_threshold
    white_pieces = gray > white_threshold
    black_pieces = gray < black_threshold
    piece_mask = white_pieces | black_pieces
    
    # Convert to uint8 {0, 255}
    mask_array = (piece_mask.astype(np.uint8)) * 255
    
    # Create PIL Image
    mask_img = Image.fromarray(mask_array, mode="L")
    
    return mask_img


def process_csv(csv_path: Path, mask_dir: Path, white_threshold: float, black_threshold: float, 
                dry_run: bool = False) -> tuple[int, int]:
    """
    Process a CSV file and generate masks for all synthetic images.
    
    Returns:
        (generated_count, skipped_count)
    """
    if not csv_path.exists():
        print(f"[SKIP] CSV not found: {csv_path}")
        return 0, 0
    
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    skipped = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in rows:
        synth_path_str = row.get("synth", "")
        if not synth_path_str:
            continue
        
        # Resolve synthetic image path
        synth_path = REPO_ROOT / synth_path_str
        if not synth_path.exists():
            print(f"[WARN] Synthetic image not found: {synth_path}")
            continue
        
        # Mask path: same stem as synthetic image
        mask_stem = synth_path.stem
        mask_path = mask_dir / f"{mask_stem}.png"
        
        if mask_path.exists():
            skipped += 1
            continue
        
        if dry_run:
            print(f"[DRY-RUN] Would generate: {mask_path}")
            generated += 1
            continue
        
        try:
            # Load synthetic image
            synth_img = Image.open(synth_path).convert("RGB")
            
            # Generate mask
            mask_img = generate_mask_from_image(synth_img, white_threshold, black_threshold)
            
            # Save mask
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_img.save(mask_path, format="PNG")
            
            generated += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to generate mask for {synth_path}: {e}")
            continue
    
    return generated, skipped


def main():
    parser = argparse.ArgumentParser(description="Generate piece masks from synthetic images")
    parser.add_argument("--train_csv", type=str, default="data/splits_rect/train_clean.csv",
                       help="Training CSV with synth paths")
    parser.add_argument("--val_csv", type=str, default=None,
                       help="Validation CSV with synth paths (optional)")
    parser.add_argument("--mask_dir", type=str, default="data/masks",
                       help="Output directory for masks (default: data/masks)")
    parser.add_argument("--white_threshold", type=float, default=200.0,
                       help="Brightness threshold for white pieces (0-255, default: 200)")
    parser.add_argument("--black_threshold", type=float, default=60.0,
                       help="Brightness threshold for black pieces (0-255, default: 60)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Preview without generating masks")
    
    args = parser.parse_args()
    
    mask_dir = REPO_ROOT / args.mask_dir
    
    print(f"Generating piece masks from synthetic images...")
    print(f"Mask directory: {mask_dir}")
    print(f"White threshold: {args.white_threshold}")
    print(f"Black threshold: {args.black_threshold}")
    if args.dry_run:
        print("[DRY-RUN MODE] No files will be written")
    print()
    
    # Process training CSV
    train_csv = REPO_ROOT / args.train_csv
    train_gen, train_skip = process_csv(train_csv, mask_dir, args.white_threshold, 
                                       args.black_threshold, args.dry_run)
    
    print(f"Train CSV: generated={train_gen}, skipped={train_skip}")
    
    # Process validation CSV if provided
    val_gen, val_skip = 0, 0
    if args.val_csv:
        val_csv = REPO_ROOT / args.val_csv
        val_gen, val_skip = process_csv(val_csv, mask_dir, args.white_threshold,
                                       args.black_threshold, args.dry_run)
        print(f"Val CSV: generated={val_gen}, skipped={val_skip}")
    
    total_gen = train_gen + val_gen
    total_skip = train_skip + val_skip
    
    print()
    print("==== Summary ====")
    print(f"Total masks generated: {total_gen}")
    print(f"Total masks skipped (already exist): {total_skip}")
    print(f"Mask directory: {mask_dir}")
    
    if not args.dry_run and total_gen > 0:
        print(f"\n[OK] Masks generated. Use --piece_mask_dir {args.mask_dir} in training.")


if __name__ == "__main__":
    main()
