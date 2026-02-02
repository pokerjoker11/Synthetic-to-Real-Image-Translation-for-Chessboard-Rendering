#!/usr/bin/env python3
"""
Find top POV images that are missing from pairs_all.csv or have incorrect paths.

Checks:
1. All images in real_top_pov/ folder
2. Whether they exist in pairs_all.csv with correct real_path
3. Outputs missing/incorrect entries to a new CSV

Usage:
    python scripts/find_missing_top_pov.py
    python scripts/find_missing_top_pov.py --csv data/chessred2k_rect_pov/pairs_all.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_top_pov_images() -> dict[str, Path]:
    """Get all images in real_top_pov/ folder. Returns dict mapping filename to full path."""
    top_pov_dir = REPO_ROOT / "data" / "chessred2k_rect_pov" / "real_top_pov"
    
    if not top_pov_dir.exists():
        print(f"[WARN] Top POV directory not found: {top_pov_dir}")
        return {}
    
    images = {}
    for img_path in top_pov_dir.glob("*.png"):
        images[img_path.name] = img_path
    
    return images


def load_csv_rows(csv_path: Path) -> tuple[list[dict], dict[str, dict]]:
    """Load CSV and create lookup by image filename."""
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return [], {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # Create lookup: filename -> row
    lookup = {}
    for row in rows:
        real_path = row.get("real_path", "")
        if real_path:
            # Extract filename from path
            filename = Path(real_path).name
            lookup[filename] = row
    
    return rows, lookup, fieldnames


def find_missing_top_pov(csv_path: Path, output_csv: Path = None):
    """Find top POV images missing from CSV or with incorrect paths."""
    # Get all top POV images
    top_pov_images = get_top_pov_images()
    print(f"Found {len(top_pov_images)} images in real_top_pov/ folder")
    
    if not top_pov_images:
        print("[WARN] No top POV images found")
        return
    
    # Load CSV
    rows, lookup, fieldnames = load_csv_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    
    # Find missing/incorrect entries
    missing_rows = []
    incorrect_path_rows = []
    found_correct = []
    
    for filename, img_path in top_pov_images.items():
        expected_path = f"data/chessred2k_rect_pov/real_top_pov/{filename}"
        
        if filename not in lookup:
            # Image not in CSV at all
            # Try to find it in pairs_ok.csv or other CSVs to get metadata
            missing_rows.append({
                "id": filename.replace(".png", ""),
                "real_path": expected_path,
                "status": "missing_from_csv",
                "filename": filename,
            })
        else:
            row = lookup[filename]
            actual_path = row.get("real_path", "")
            
            if "real_top_pov" in actual_path:
                # Correct path
                found_correct.append(row)
            else:
                # Wrong path (e.g., still pointing to real_ok/)
                incorrect_path_rows.append({
                    **row,  # Keep all original fields
                    "expected_path": expected_path,
                    "actual_path": actual_path,
                    "status": "incorrect_path",
                })
    
    # Print summary
    print(f"\n==== Summary ====")
    print(f"Top POV images found in folder: {len(top_pov_images)}")
    print(f"  - Correctly in CSV with correct path: {len(found_correct)}")
    print(f"  - In CSV but wrong path: {len(incorrect_path_rows)}")
    print(f"  - Missing from CSV: {len(missing_rows)}")
    
    # Output results
    if output_csv is None:
        output_csv = csv_path.parent / "missing_top_pov.csv"
    
    all_issues = []
    
    # Add missing rows (try to get metadata from pairs_ok.csv if available)
    pairs_ok_csv = csv_path.parent / "pairs_ok.csv"
    ok_lookup = {}
    if pairs_ok_csv.exists():
        with open(pairs_ok_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                real_path = row.get("real_path", "")
                if real_path:
                    filename = Path(real_path).name
                    ok_lookup[filename] = row
    
    for missing in missing_rows:
        filename = missing["filename"]
        # Try to get metadata from pairs_ok.csv
        if filename in ok_lookup:
            row = ok_lookup[filename].copy()
            row["real_path"] = missing["real_path"]  # Update to top_pov path
            row["status"] = "missing_from_all_csv"
            all_issues.append(row)
        else:
            # Create minimal row
            all_issues.append(missing)
    
    # Add incorrect path rows
    for row in incorrect_path_rows:
        # Update real_path to correct one
        row["real_path"] = row["expected_path"]
        all_issues.append(row)
    
    if all_issues:
        # Write output CSV
        if fieldnames and all_issues and "id" in all_issues[0]:
            # Use original fieldnames if available
            output_fieldnames = list(fieldnames)
            if "status" not in output_fieldnames:
                output_fieldnames.append("status")
            if "expected_path" not in output_fieldnames and any("expected_path" in r for r in all_issues):
                output_fieldnames.append("expected_path")
            if "actual_path" not in output_fieldnames and any("actual_path" in r for r in all_issues):
                output_fieldnames.append("actual_path")
        else:
            # Use fieldnames from first row
            output_fieldnames = list(all_issues[0].keys())
        
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(all_issues)
        
        print(f"\n[OK] Wrote {len(all_issues)} missing/incorrect entries to: {output_csv}")
    else:
        print("\n[OK] All top POV images are correctly in CSV with correct paths!")
    
    return all_issues


def main():
    parser = argparse.ArgumentParser(description="Find missing top POV images in CSV")
    parser.add_argument("--csv", type=str, default="data/chessred2k_rect_pov/pairs_all.csv",
                       help="Input CSV file to check")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file (default: missing_top_pov.csv in same directory)")
    
    args = parser.parse_args()
    
    csv_path = REPO_ROOT / args.csv
    output_csv = REPO_ROOT / args.output if args.output else None
    
    find_missing_top_pov(csv_path, output_csv)


if __name__ == "__main__":
    main()
