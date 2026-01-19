#!/usr/bin/env python3
"""
Pre-crop synthetic images to remove borders and match real image framing.

Detects the chessboard playing area and crops to just the 8x8 squares.

Usage:
    python scripts/crop_synth_images.py
    python scripts/crop_synth_images.py --preview 10  # Preview first 10 without saving
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = REPO_ROOT / "data" / "synth_v3" / "images"
OUTPUT_DIR = REPO_ROOT / "data" / "synth_v3_cropped" / "images"


def detect_board_bounds(img: Image.Image, margin_pct: float = 0.02) -> tuple:
    """
    Detect the chessboard bounds by finding high-variance regions (checkerboard pattern).
    
    Returns: (left, top, right, bottom) crop box
    """
    arr = np.array(img.convert('L'), dtype=np.float32)
    h, w = arr.shape
    
    # Calculate local variance using sliding window
    window = 8
    
    # Row variance profile (sum variance across each row)
    row_var = []
    for y in range(0, h - window, window // 2):
        row_slice = arr[y:y+window, :]
        var = np.var(row_slice, axis=0).mean()
        row_var.append((y + window // 2, var))
    
    # Column variance profile
    col_var = []
    for x in range(0, w - window, window // 2):
        col_slice = arr[:, x:x+window]
        var = np.var(col_slice, axis=1).mean()
        col_var.append((x + window // 2, var))
    
    # Find high-variance region (the checkerboard)
    if row_var:
        row_vars = [v for _, v in row_var]
        threshold = np.percentile(row_vars, 50)  # Above median
        
        high_rows = [pos for pos, v in row_var if v > threshold]
        if high_rows:
            top = max(0, min(high_rows) - window)
            bottom = min(h, max(high_rows) + window)
        else:
            top, bottom = 0, h
    else:
        top, bottom = 0, h
    
    if col_var:
        col_vars = [v for _, v in col_var]
        threshold = np.percentile(col_vars, 50)
        
        high_cols = [pos for pos, v in col_var if v > threshold]
        if high_cols:
            left = max(0, min(high_cols) - window)
            right = min(w, max(high_cols) + window)
        else:
            left, right = 0, w
    else:
        left, right = 0, w
    
    # Make square (use the larger dimension)
    crop_h = bottom - top
    crop_w = right - left
    
    if crop_h > crop_w:
        # Expand width
        diff = crop_h - crop_w
        left = max(0, left - diff // 2)
        right = min(w, right + diff - diff // 2)
    elif crop_w > crop_h:
        # Expand height
        diff = crop_w - crop_h
        top = max(0, top - diff // 2)
        bottom = min(h, bottom + diff - diff // 2)
    
    # Add small margin
    margin = int(min(crop_h, crop_w) * margin_pct)
    left = max(0, left + margin)
    top = max(0, top + margin)
    right = min(w, right - margin)
    bottom = min(h, bottom - margin)
    
    return (left, top, right, bottom)


def crop_to_board_simple(img: Image.Image, border_pct: float = 0.03) -> Image.Image:
    """
    Simple approach: crop a fixed percentage from all edges.
    The Blender renders have consistent framing, so this works well.
    """
    w, h = img.size
    border_x = int(w * border_pct)
    border_y = int(h * border_pct)
    
    left = border_x
    top = border_y
    right = w - border_x
    bottom = h - border_y
    
    cropped = img.crop((left, top, right, bottom))
    return cropped


def crop_to_board_adaptive(img: Image.Image) -> Image.Image:
    """
    Adaptive approach: detect the actual board bounds.
    """
    bounds = detect_board_bounds(img)
    cropped = img.crop(bounds)
    return cropped


def main():
    parser = argparse.ArgumentParser(description="Crop synthetic images to remove borders")
    parser.add_argument('--preview', type=int, default=0, help="Preview N images without saving")
    parser.add_argument('--border-pct', type=float, default=0.02, help="Border percentage to crop (default: 2%%)")
    parser.add_argument('--method', choices=['simple', 'adaptive'], default='simple',
                        help="Cropping method: simple (fixed %), adaptive (detect bounds)")
    parser.add_argument('--input-dir', type=str, default=str(SYNTH_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1
    
    images = list(input_dir.glob("*.png"))
    print(f"Found {len(images)} images in {input_dir}")
    
    if args.preview > 0:
        print(f"\n=== PREVIEW MODE (first {args.preview} images) ===")
        preview_dir = REPO_ROOT / "results" / "crop_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_path in enumerate(images[:args.preview]):
            img = Image.open(img_path).convert('RGB')
            
            if args.method == 'simple':
                cropped = crop_to_board_simple(img, args.border_pct)
            else:
                cropped = crop_to_board_adaptive(img)
            
            # Create side-by-side comparison
            w, h = img.size
            cw, ch = cropped.size
            
            # Resize cropped to match original height for comparison
            cropped_display = cropped.resize((h, h), Image.BICUBIC)
            
            canvas = Image.new('RGB', (w + h + 10, h), (255, 255, 255))
            canvas.paste(img, (0, 0))
            canvas.paste(cropped_display, (w + 10, 0))
            
            out_path = preview_dir / f"preview_{i:03d}.png"
            canvas.save(out_path)
            print(f"  [{i+1}] {img_path.name} -> {out_path}")
        
        print(f"\nPreview images saved to: {preview_dir}")
        print("Check the previews, then run without --preview to process all images.")
        return 0
    
    # Full processing
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(images)} images...")
    print(f"  Method: {args.method}")
    print(f"  Border %: {args.border_pct * 100:.1f}%")
    print(f"  Output: {output_dir}")
    
    success = 0
    failed = 0
    
    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert('RGB')
            
            if args.method == 'simple':
                cropped = crop_to_board_simple(img, args.border_pct)
            else:
                cropped = crop_to_board_adaptive(img)
            
            # Save with same filename
            out_path = output_dir / img_path.name
            cropped.save(out_path, quality=95)
            success += 1
            
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(images)}] processed...")
                
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {success} images, {failed} failed")
    print(f"Output: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Update CSVs to point to synth_v3_cropped instead of synth_v3")
    print(f"  2. Or run: python scripts/update_csv_synth_path.py")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
