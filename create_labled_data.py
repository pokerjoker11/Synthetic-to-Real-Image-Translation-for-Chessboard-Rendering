import argparse
import os
import shutil
import zipfile
import re
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# Folder containing your game1_per_frame.zip, game2_per_frame.zip, etc.
RAW_ZIPS_DIR = Path("data/raw_zips")  

# Output root folder
OUTPUT_ROOT = Path("LabeledData")

# Blender paths
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE = Path("chess-set.blend")
BLENDER_SCRIPT = Path("chess_position_api_v2.py")

# Regular expression to find tagged images in the zip
FRAME_RE = re.compile(r"(^|/)tagged_images/frame_(\d+)\.(jpg|png)$", re.IGNORECASE)


def normalize_fen(fen: str) -> str:
    """
    Converts a FEN string into a safe folder name.
    Example: '3r2k1/pp4p1/...' -> '3r2k1_pp4p1_...'
    """
    # Take only the board configuration (first part of FEN)
    board_part = fen.split()[0]
    # Replace slashes with underscores for filesystem safety
    return board_part.replace("/", "_")


def _index_tagged_images(z: zipfile.ZipFile):
    """
    Builds a dictionary mapping: frame_index -> zip_member_filename
    """
    mapping = {}
    for name in z.namelist():
        m = FRAME_RE.search(name)
        if not m:
            continue
        frame = int(m.group(2))
        mapping.setdefault(frame, name)
    return mapping


def run_blender_render(fen: str, output_dir: Path):
    """
    Runs Blender to generate the synthetic image.
    FIXED: Uses 'utf-8' encoding to prevent Windows Hebrew/cp1255 crash.
    """
    abs_output_dir = output_dir.resolve()
    abs_output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        str(BLENDER_EXE),
        str(BLEND_FILE),
        "--background",
        "--python", str(BLENDER_SCRIPT),
        "--",
        "--fen", fen,
        "--view", "white",
        "--out_dir", str(abs_output_dir)
    ]

    try:
        # --- THE FIX IS HERE ---
        # We added encoding="utf-8" so it reads Blender logs correctly.
        # We added errors="replace" so if a weird character appears, it won't crash.
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",    # <--- CRITICAL FIX
            errors="replace"     # <--- SAFETY NET
        )
        
        # Validation
        generated_files = list(abs_output_dir.glob("*.png"))
        
        if not generated_files:
            print(f"\n[ERROR] Blender finished, but no PNG found in: {abs_output_dir}")
            print("-" * 20)
            print("BLENDER OUTPUT LOG:")
            print("\n".join(result.stdout.splitlines()[-20:]))
            print("-" * 20)
            return

        # Rename/Clean up
        latest_file = max(generated_files, key=os.path.getmtime)
        target_file = abs_output_dir / "synthetic.png"
        
        if latest_file.name != "synthetic.png":
            if target_file.exists():
                target_file.unlink()
            shutil.move(str(latest_file), str(target_file))

    except subprocess.CalledProcessError as e:
        print(f"\n[CRASH] Blender failed for FEN {fen}.")
        # Use 'replace' here too just in case the error message has weird text
        err_msg = e.stderr.decode("utf-8", errors="replace") if hasattr(e.stderr, "decode") else str(e.stderr)
        print(f"Error Output: {err_msg}")


def process_game_zip(zip_path: Path, limit: int = None, current_count: int = 0):
    """
    Reads a zip file, extracts real images, and triggers Blender for synthetic ones.
    Returns the updated count of processed items.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find the CSV file inside the zip
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            print(f"[WARN] No CSV found in {zip_path.name}")
            return current_count
        
        # Load CSV
        df = pd.read_csv(z.open(csvs[0]))
        
        # Map frame numbers to image file paths inside zip
        frame_map = _index_tagged_images(z)
        
        # Iterate through every labeled position
        for _, row in df.iterrows():
            if limit is not None and current_count >= limit:
                return current_count

            frame = int(row["from_frame"])
            fen = str(row["fen"]).strip()
            
            # Check if we have the real image for this frame
            real_img_member = frame_map.get(frame)
            if not real_img_member:
                continue 

            # Define normalized folder name
            fen_folder_name = normalize_fen(fen)
            
            # Define target directories
            real_target_dir = OUTPUT_ROOT / "real" / fen_folder_name
            syn_target_dir = OUTPUT_ROOT / "synthetic" / fen_folder_name
            
            # --- 1. HANDLE REAL IMAGE ---
            real_target_dir.mkdir(parents=True, exist_ok=True)
            real_out_path = real_target_dir / "real.png"
            
            if not real_out_path.exists():
                with z.open(real_img_member) as src, open(real_out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            # --- 2. HANDLE SYNTHETIC IMAGE ---
            syn_target_dir.mkdir(parents=True, exist_ok=True)
            syn_out_path = syn_target_dir / "synthetic.png"
            
            if not syn_out_path.exists():
                # print(f"Rendering: {fen_folder_name}...") # Uncomment for verbose
                run_blender_render(fen, syn_target_dir)

            current_count += 1
            
    return current_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to process (for testing)")
    args = parser.parse_args()

    if not RAW_ZIPS_DIR.exists():
        print(f"Error: Directory {RAW_ZIPS_DIR} does not exist.")
        return

    zips = sorted(list(RAW_ZIPS_DIR.glob("game*_per_frame.zip")))
    if not zips:
        print(f"No zip files found in {RAW_ZIPS_DIR}")
        return

    print(f"Found {len(zips)} zip files. Outputting to: {OUTPUT_ROOT}")
    
    total_processed = 0
    
    # Progress bar for files
    pbar = tqdm(zips, desc="Processing Games")
    for zpath in pbar:
        if args.limit and total_processed >= args.limit:
            break
            
        total_processed = process_game_zip(zpath, limit=args.limit, current_count=total_processed)
        pbar.set_postfix({"Images": total_processed})

    print(f"\n==== COMPLETE ====")
    print(f"Processed {total_processed} pairs.")
    print(f"Check results in: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()