#!/usr/bin/env python3
"""
Compare training samples from different steps to see quality progression.
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
    milestones = []
    
    # First sample
    milestones.append(("Early", step_files[0]))
    
    # Middle samples (every ~20k steps)
    if len(step_files) > 2:
        mid_idx = len(step_files) // 2
        milestones.append(("Mid", step_files[mid_idx]))
    
    # Last sample
    milestones.append(("Latest", step_files[-1]))
    
    # Also check specific steps if they exist
    for step_name in ["step_010000.png", "step_050000.png", "step_100000.png"]:
        step_path = samples_dir / step_name
        if step_path.exists():
            milestones.append((step_name.replace("step_", "").replace(".png", ""), step_path))
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_milestones = []
    for name, path in milestones:
        if path not in seen:
            seen.add(path)
            unique_milestones.append((name, path))
    
    print(f"\nComparing {len(unique_milestones)} milestones:")
    for name, path in unique_milestones:
        print(f"  {name}: {path.name}")
    
    # Load and create comparison
    images = []
    labels = []
    
    for name, path in unique_milestones:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            labels.append(name)
            print(f"[OK] Loaded {path.name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"[ERROR] Failed to load {path.name}: {e}")
    
    if len(images) == 0:
        print("[ERROR] No images loaded")
        return 1
    
    # Create side-by-side comparison
    # Training samples are typically 3 images side-by-side: input | output | target
    # We'll stack them vertically to show progression
    
    # Get dimensions
    sample_width = images[0].width
    sample_height = images[0].height
    
    # Create canvas: one row per milestone
    canvas_width = sample_width
    canvas_height = sample_height * len(images)
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    
    y_offset = 0
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (0, y_offset))
        
        # Add label (simple text overlay using PIL's basic text)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            # Try to use a larger font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Draw semi-transparent background for text
        text = f"Step {label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background rectangle
        padding = 5
        draw.rectangle(
            [10, y_offset + 10, 10 + text_width + 2*padding, y_offset + 10 + text_height + 2*padding],
            fill=(0, 0, 0, 200)
        )
        
        # Draw text
        draw.text((10 + padding, y_offset + 10 + padding), text, fill=(255, 255, 255), font=font)
        
        y_offset += sample_height
    
    # Save comparison
    output_path = REPO_ROOT / "results" / "training_progression.png"
    canvas.save(output_path)
    print(f"\n[OK] Saved progression comparison: {output_path}")
    print(f"  Size: {canvas_width}x{canvas_height}")
    print(f"  Shows {len(images)} milestones stacked vertically")
    
    # Also create a grid if we have multiple samples
    if len(images) >= 2:
        # Create a 2-column grid
        cols = 2
        rows = (len(images) + 1) // 2
        
        grid_width = sample_width * cols
        grid_height = sample_height * rows
        grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
        
        for i, (img, label) in enumerate(zip(images, labels)):
            col = i % cols
            row = i // cols
            x = col * sample_width
            y = row * sample_height
            grid.paste(img, (x, y))
            
            # Add label
            draw = ImageDraw.Draw(grid)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            text = f"Step {label}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            padding = 5
            
            draw.rectangle(
                [x + 10, y + 10, x + 10 + text_width + 2*padding, y + 10 + text_height + 2*padding],
                fill=(0, 0, 0, 200)
            )
            draw.text((x + 10 + padding, y + 10 + padding), text, fill=(255, 255, 255), font=font)
        
        grid_path = REPO_ROOT / "results" / "training_progression_grid.png"
        grid.save(grid_path)
        print(f"[OK] Saved grid comparison: {grid_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
