#!/usr/bin/env python3
"""
Analyze training samples around the 30k-50k range to find where quality plateaued.
"""

import sys
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    samples_dir = REPO_ROOT / "results" / "train_samples_clean"
    
    if not samples_dir.exists():
        print(f"[ERROR] Samples directory not found: {samples_dir}")
        return 1
    
    # Focus on the 30k-50k range where improvement likely stopped
    milestones = [
        ("30000", "step_030000.png"),
        ("35000", "step_035000.png"),
        ("40000", "step_040000.png"),
        ("43000", "step_043000.png"),
        ("45000", "step_045000.png"),
        ("50000", "step_050000.png"),
    ]
    
    existing = []
    for name, filename in milestones:
        path = samples_dir / filename
        if path.exists():
            existing.append((name, path))
        else:
            print(f"[WARN] {filename} not found")
    
    if len(existing) == 0:
        print("[ERROR] No milestone samples found")
        return 1
    
    print(f"Found {len(existing)} samples in 30k-50k range\n")
    
    # Extract generator outputs (middle column)
    outputs = []
    labels = []
    
    for name, path in existing:
        try:
            img = Image.open(path).convert("RGB")
            width, height = img.size
            if width == 768 and height == 256:
                # Extract middle column (generator output)
                output = img.crop((256, 0, 512, 256))
                outputs.append(output)
                labels.append(name)
                print(f"[OK] Extracted step {name}")
        except Exception as e:
            print(f"[ERROR] Failed to process {path.name}: {e}")
    
    if len(outputs) == 0:
        print("[ERROR] No outputs extracted")
        return 1
    
    # Create side-by-side comparison
    img_width = outputs[0].width
    img_height = outputs[0].height
    
    canvas_width = img_width * len(outputs)
    canvas_height = img_height + 40
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    x_offset = 0
    for img, label in zip(outputs, labels):
        canvas.paste(img, (x_offset, 40))
        
        draw = ImageDraw.Draw(canvas)
        text = f"Step {label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        padding = 5
        label_x = x_offset + (img_width - text_width) // 2 - padding
        
        draw.rectangle(
            [label_x, 5, label_x + text_width + 2*padding, 5 + bbox[3] - bbox[1] + 2*padding],
            fill=(0, 0, 0, 200)
        )
        draw.text((label_x + padding, 5 + padding), text, fill=(255, 255, 255), font=font)
        
        x_offset += img_width
    
    output_path = REPO_ROOT / "results" / "plateau_analysis_30k_50k.png"
    canvas.save(output_path)
    print(f"\n[OK] Saved analysis: {output_path}")
    print(f"  Shows generator outputs from steps: {', '.join(labels)}")
    print(f"  Look for where quality stops improving (plateau point)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
