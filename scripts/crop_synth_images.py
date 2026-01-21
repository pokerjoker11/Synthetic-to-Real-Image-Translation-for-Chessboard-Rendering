#!/usr/bin/env python3
"""
Crop synthetic images using a manual bbox (x0,y0,x1,y1) so outputs contain only the 8x8 grid area.

This removes the Blender rim/frame in a deterministic way and (optionally) resizes to a canonical size.

Usage:
  # Preview N images (writes side-by-side previews to results/crop_preview/)
  python scripts/crop_synth_images.py --input-dir data/synth_v3_cropped/images --bbox 26,23,475,476 --preview 10

  # Process all images -> new folder, cropped + resized to 256x256
  python scripts/crop_synth_images.py --input-dir data/synth_v3_cropped/images --output-dir data/synth_v4_rect256/images --bbox 26,23,475,476 --resize 256
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_bbox(bbox_str: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--bbox must satisfy x1>x0 and y1>y0")
    return x0, y0, x1, y1


def _clip_bbox_to_image(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox invalid after clipping to image size {w}x{h}: {x0},{y0},{x1},{y1}")
    return x0, y0, x1, y1


def main():
    parser = argparse.ArgumentParser(description="Crop synthetic images with a manual bbox")
    parser.add_argument("--preview", type=int, default=0, help="Preview N images without saving cropped outputs")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of input .png images")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for cropped outputs (required if not preview)")
    parser.add_argument("--bbox", type=str, required=True, help="Manual crop bbox: x0,y0,x1,y1 (pixel coords, 0,0 is top-left)")
    parser.add_argument("--resize", type=int, default=256, help="Resize cropped output to NxN (use 0 to disable)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    bbox_raw = _parse_bbox(args.bbox)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    images = sorted(input_dir.glob("*.png"))
    print(f"Found {len(images)} images in {input_dir}")
    print(f"Using bbox: {bbox_raw} | resize: {args.resize}")

    # Preview mode
    if args.preview > 0:
        print(f"\n=== PREVIEW MODE (first {args.preview} images) ===")
        preview_dir = REPO_ROOT / "results" / "crop_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images[: args.preview]):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            bbox = _clip_bbox_to_image(bbox_raw, w, h)
            cropped = img.crop(bbox)

            if args.resize and args.resize > 0:
                cropped = cropped.resize((args.resize, args.resize), Image.BICUBIC)

            # Side-by-side: original (left) + cropped resized to original height (right)
            cropped_display = cropped.resize((h, h), Image.BICUBIC)
            canvas = Image.new("RGB", (w + h + 10, h), (255, 255, 255))
            canvas.paste(img, (0, 0))
            canvas.paste(cropped_display, (w + 10, 0))

            out_path = preview_dir / f"preview_{i:03d}.png"
            canvas.save(out_path)
            print(f"  [{i+1}] {img_path.name} -> {out_path}")

        print(f"\nPreview images saved to: {preview_dir}")
        print("If the bbox looks good, run again without --preview to process all images.")
        return 0

    # Full processing mode
    if output_dir is None:
        print("[ERROR] --output-dir is required when not using --preview")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(images)} images...")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  BBox  : {bbox_raw}")
    print(f"  Resize: {args.resize}")

    success = 0
    failed = 0

    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            bbox = _clip_bbox_to_image(bbox_raw, w, h)
            cropped = img.crop(bbox)

            if args.resize and args.resize > 0:
                cropped = cropped.resize((args.resize, args.resize), Image.BICUBIC)

            out_path = output_dir / img_path.name
            cropped.save(out_path, format="PNG", optimize=True)
            success += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(images)}] processed...")

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Done! Processed {success} images, {failed} failed")
    print(f"Output: {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
