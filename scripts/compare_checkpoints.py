#!/usr/bin/env python3
"""
Compare outputs from different checkpoints side-by-side.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    test_dir = REPO_ROOT / "results" / "test_random"
    
    # Find all test results for the same FEN
    # We'll look for test_01_* files (same FEN, different checkpoints)
    
    # Checkpoint outputs to compare
    checkpoints = [
        ("43k", "test_01_best_step40000"),
        ("45k", "test_01_latest_step44500"),
        ("62k (best)", "test_01_best"),
        ("116k (latest)", "test_01_latest"),
    ]
    
    # Find existing files
    existing = []
    for label, prefix in checkpoints:
        realistic = test_dir / f"{prefix}_realistic.png"
        side_by_side = test_dir / f"{prefix}_side_by_side.png"
        
        if realistic.exists() or side_by_side.exists():
            existing.append((label, prefix, realistic, side_by_side))
    
    if len(existing) == 0:
        print(f"[ERROR] No test results found in {test_dir}")
        print("Run test_random_positions.py first with different checkpoints")
        return 1
    
    print(f"Found {len(existing)} checkpoint outputs to compare\n")
    
    # Load realistic outputs (generator outputs only)
    images = []
    labels = []
    
    for label, prefix, realistic_path, side_by_side_path in existing:
        # Prefer realistic.png (just output), fallback to side_by_side
        if realistic_path.exists():
            img = Image.open(realistic_path).convert("RGB")
            images.append(img)
            labels.append(label)
            print(f"[OK] Loaded {label}: {realistic_path.name}")
        elif side_by_side_path.exists():
            # Extract right half (realistic output)
            img = Image.open(side_by_side_path).convert("RGB")
            width, height = img.size
            # Side-by-side is typically 2 images, extract right half
            output = img.crop((width // 2, 0, width, height))
            images.append(output)
            labels.append(label)
            print(f"[OK] Loaded {label}: {side_by_side_path.name} (extracted right half)")
        else:
            print(f"[WARN] No image found for {label}")
    
    if len(images) == 0:
        print("[ERROR] No images loaded")
        return 1
    
    # Create side-by-side comparison
    img_width = images[0].width
    img_height = images[0].height
    
    # Resize all to same size if needed
    for i in range(len(images)):
        if images[i].size != (img_width, img_height):
            images[i] = images[i].resize((img_width, img_height), Image.BICUBIC)
    
    # Horizontal layout
    canvas_width = img_width * len(images)
    canvas_height = img_height + 40  # Extra space for labels
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    
    x_offset = 0
    for img, label in zip(images, labels):
        canvas.paste(img, (x_offset, 40))  # Leave top 40px for label
        
        # Add label
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = f"Step {label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = 5
        
        # Center the label
        label_x = x_offset + (img_width - text_width) // 2 - padding
        label_y = 5
        
        # Draw background
        draw.rectangle(
            [label_x, label_y, label_x + text_width + 2*padding, label_y + text_height + 2*padding],
            fill=(0, 0, 0, 200)
        )
        draw.text((label_x + padding, label_y + padding), text, fill=(255, 255, 255), font=font)
        
        x_offset += img_width
    
    # Save comparison
    output_path = REPO_ROOT / "results" / "checkpoint_comparison.png"
    canvas.save(output_path)
    print(f"\n[OK] Saved comparison: {output_path}")
    print(f"  Size: {canvas_width}x{canvas_height}")
    print(f"  Shows {len(images)} checkpoint outputs side-by-side")
    print(f"  Left to right: {', '.join(labels)}")
    
    # Also create a focused 43k vs 45k comparison if both exist
    if len(images) >= 2:
        # Find 43k and 45k indices
        idx_43k = None
        idx_45k = None
        
        for i, label in enumerate(labels):
            if "43k" in label:
                idx_43k = i
            if "45k" in label:
                idx_45k = i
        
        if idx_43k is not None and idx_45k is not None:
            focused_canvas = Image.new("RGB", (img_width * 2, img_height + 40), color=(255, 255, 255))
            
            # Add 43k
            focused_canvas.paste(images[idx_43k], (0, 40))
            draw = ImageDraw.Draw(focused_canvas)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            text = f"Step {labels[idx_43k]}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            padding = 5
            label_x = (img_width - text_width) // 2 - padding
            draw.rectangle(
                [label_x, 5, label_x + text_width + 2*padding, 5 + bbox[3] - bbox[1] + 2*padding],
                fill=(0, 0, 0, 200)
            )
            draw.text((label_x + padding, 5 + padding), text, fill=(255, 255, 255), font=font)
            
            # Add 45k
            focused_canvas.paste(images[idx_45k], (img_width, 40))
            text = f"Step {labels[idx_45k]}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            label_x = img_width + (img_width - text_width) // 2 - padding
            draw.rectangle(
                [label_x, 5, label_x + text_width + 2*padding, 5 + bbox[3] - bbox[1] + 2*padding],
                fill=(0, 0, 0, 200)
            )
            draw.text((label_x + padding, 5 + padding), text, fill=(255, 255, 255), font=font)
            
            focused_path = REPO_ROOT / "results" / "checkpoint_43k_vs_45k.png"
            focused_canvas.save(focused_path)
            print(f"[OK] Saved focused 43k vs 45k comparison: {focused_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
