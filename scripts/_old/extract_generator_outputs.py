#!/usr/bin/env python3
"""
Extract just the generator output (middle column) from training samples
to compare quality progression.
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
    
    # Find all step samples
    step_files = sorted(samples_dir.glob("step_*.png"))
    
    if len(step_files) == 0:
        print(f"[ERROR] No step samples found in {samples_dir}")
        return 1
    
    print(f"Found {len(step_files)} training samples")
    
    # Select key milestones
    milestones = [
        ("000500", "step_000500.png"),
        ("010000", "step_010000.png"),
        ("050000", "step_050000.png"),
        ("100000", "step_100000.png"),
        ("117000", "step_117000.png"),
    ]
    
    # Filter to existing files
    existing = []
    for name, filename in milestones:
        path = samples_dir / filename
        if path.exists():
            existing.append((name, path))
        else:
            print(f"[WARN] {filename} not found, skipping")
    
    if len(existing) == 0:
        print("[ERROR] No milestone samples found")
        return 1
    
    print(f"\nExtracting generator outputs from {len(existing)} milestones:")
    
    # Extract middle column (generator output) from each
    outputs = []
    labels = []
    
    for name, path in existing:
        try:
            img = Image.open(path).convert("RGB")
            # Training samples are 768x256 = 3 columns of 256x256
            # Column 0: input (synth)
            # Column 1: output (generated)
            # Column 2: target (real)
            
            width, height = img.size
            if width == 768 and height == 256:
                # Extract middle column (256-512)
                output = img.crop((256, 0, 512, 256))
                outputs.append(output)
                labels.append(name)
                print(f"[OK] Extracted {name} ({output.size[0]}x{output.size[1]})")
            else:
                print(f"[WARN] Unexpected size for {path.name}: {width}x{height}")
        except Exception as e:
            print(f"[ERROR] Failed to process {path.name}: {e}")
    
    if len(outputs) == 0:
        print("[ERROR] No outputs extracted")
        return 1
    
    # Create side-by-side comparison of generator outputs only
    img_width = outputs[0].width
    img_height = outputs[0].height
    
    # Horizontal layout
    canvas_width = img_width * len(outputs)
    canvas_height = img_height
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    
    x_offset = 0
    for img, label in zip(outputs, labels):
        canvas.paste(img, (x_offset, 0))
        
        # Add label
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text = f"Step {label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = 5
        
        # Draw background
        draw.rectangle(
            [x_offset + 10, 10, x_offset + 10 + text_width + 2*padding, 10 + text_height + 2*padding],
            fill=(0, 0, 0, 200)
        )
        draw.text((x_offset + 10 + padding, 10 + padding), text, fill=(255, 255, 255), font=font)
        
        x_offset += img_width
    
    # Save comparison
    output_path = REPO_ROOT / "results" / "generator_outputs_progression.png"
    canvas.save(output_path)
    print(f"\n[OK] Saved generator outputs comparison: {output_path}")
    print(f"  Size: {canvas_width}x{canvas_height}")
    print(f"  Shows {len(outputs)} generator outputs side-by-side")
    print(f"  Left to right: Early -> Latest")
    
    # Also save individual outputs for closer inspection
    output_dir = REPO_ROOT / "results" / "generator_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img, label in zip(outputs, labels):
        individual_path = output_dir / f"step_{label}_output.png"
        img.save(individual_path)
        print(f"  Saved: {individual_path.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
