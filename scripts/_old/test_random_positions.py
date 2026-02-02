#!/usr/bin/env python3
"""
Test the trained model on random chess positions.

Generates random valid chess positions, renders them with Blender,
and runs them through the model to see results.

Usage:
    python scripts/test_random_positions.py
    python scripts/test_random_positions.py --num 5 --viewpoint white
"""

import argparse
import os
import sys
from pathlib import Path
import random

try:
    import chess
except ImportError:
    print("[ERROR] chess library not found. Install with: pip install chess")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_api import generate_chessboard_image


def _get_checkpoint_step(ckpt_path: Path) -> int:
    """Get step number from checkpoint."""
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("step", 0)
    except:
        return 0


def find_best_checkpoint():
    """Find the best checkpoint from checkpoints directories."""
    # Try checkpoints_fresh first (current training), then others
    for ckpt_dir_name in ["checkpoints_fresh", "checkpoints_advanced", "checkpoints_clean", "checkpoints_improved"]:
        ckpt_dir = REPO_ROOT / ckpt_dir_name
        if not ckpt_dir.exists():
            continue
            
        best_pt = ckpt_dir / "best.pt"
        latest_pt = ckpt_dir / "latest.pt"
        
        if best_pt.exists():
            return best_pt
        elif latest_pt.exists():
            return latest_pt
        else:
            # Find most recent
            pt_files = list(ckpt_dir.glob("*.pt"))
            if pt_files:
                return max(pt_files, key=lambda p: p.stat().st_mtime)
    return None


def generate_random_position(moves: int = None) -> str:
    """
    Generate a random valid chess position.
    
    Args:
        moves: Number of random moves to make (default: 10-30)
    
    Returns:
        FEN string of the position
    """
    board = chess.Board()
    
    if moves is None:
        moves = random.randint(10, 30)
    
    # Make random legal moves
    for _ in range(moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    
    return board.fen()


def main():
    parser = argparse.ArgumentParser(description="Test model on random chess positions")
    parser.add_argument('--num', type=int, default=3, help="Number of random positions to test")
    parser.add_argument('--viewpoint', choices=['white', 'black'], default='white',
                       help="Viewpoint for rendering")
    parser.add_argument('--output-dir', type=str, default='results/test_random',
                       help="Output directory for test results")
    parser.add_argument('--ckpt', type=str, default=None,
                       help="Specific checkpoint to use (e.g., 'latest.pt' or 'latest_step44500_backup.pt')")
    parser.add_argument('--fen', type=str, nargs='+', default=None,
                       help="Use specific FEN(s) instead of generating random")
    
    args = parser.parse_args()
    
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find checkpoint
    if args.ckpt:
        # Try as absolute path first
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_absolute():
            if not ckpt_path.exists():
                print(f"[ERROR] Checkpoint not found: {ckpt_path}")
                return 1
        else:
            # Try relative paths: direct, checkpoints_clean, checkpoints_improved
            if (REPO_ROOT / args.ckpt).exists():
                ckpt_path = REPO_ROOT / args.ckpt
            elif (REPO_ROOT / "checkpoints_advanced" / args.ckpt).exists():
                ckpt_path = REPO_ROOT / "checkpoints_advanced" / args.ckpt
            elif (REPO_ROOT / "checkpoints_clean" / args.ckpt).exists():
                ckpt_path = REPO_ROOT / "checkpoints_clean" / args.ckpt
            elif (REPO_ROOT / "checkpoints_improved" / args.ckpt).exists():
                ckpt_path = REPO_ROOT / "checkpoints_improved" / args.ckpt
            elif (REPO_ROOT / "checkpoints_fresh" / args.ckpt).exists():
                ckpt_path = REPO_ROOT / "checkpoints_fresh" / args.ckpt
            else:
                print(f"[ERROR] Checkpoint not found: {args.ckpt}")
                print(f"  Tried: {REPO_ROOT / args.ckpt}")
                print(f"  Tried: {REPO_ROOT / 'checkpoints_fresh' / args.ckpt}")
                print(f"  Tried: {REPO_ROOT / 'checkpoints_clean' / args.ckpt}")
                print(f"  Tried: {REPO_ROOT / 'checkpoints_improved' / args.ckpt}")
                return 1
    else:
        ckpt_path = find_best_checkpoint()
        if ckpt_path is None:
            print("[ERROR] No checkpoint found in checkpoints directories")
            return 1
    
    # Set CKPT_PATH to absolute path so eval_api can find it
    os.environ['CKPT_PATH'] = str(ckpt_path.resolve())
    print(f"Using checkpoint: {ckpt_path.name} (step {_get_checkpoint_step(ckpt_path) if _get_checkpoint_step else 'unknown'})")
    
    print(f"\nGenerating {args.num} random chess positions...")
    print(f"Viewpoint: {args.viewpoint}")
    print(f"Output: {output_dir}\n")
    
    results = []
    
    # Get FENs to test
    if args.fen:
        fens = args.fen
    else:
        fens = [generate_random_position() for _ in range(args.num)]
    
    for i, fen in enumerate(fens):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(fens)}")
        print(f"{'='*60}")
        
        print(f"FEN: {fen}")
        
        # Generate images
        try:
            generate_chessboard_image(fen, args.viewpoint)
            
            # Copy results to test directory with unique names
            results_dir = REPO_ROOT / "results"
            ckpt_suffix = ckpt_path.stem.replace('_backup', '')  # Remove _backup suffix for cleaner names
            test_id = f"test_{i+1:02d}_{ckpt_suffix}"
            
            # Copy synthetic
            synth_src = results_dir / "synthetic.png"
            synth_dst = output_dir / f"{test_id}_synthetic.png"
            if synth_src.exists():
                import shutil
                shutil.copy2(synth_src, synth_dst)
                print(f"  Saved: {synth_dst.name}")
            
            # Copy realistic
            real_src = results_dir / "realistic.png"
            real_dst = output_dir / f"{test_id}_realistic.png"
            if real_src.exists():
                shutil.copy2(real_src, real_dst)
                print(f"  Saved: {real_dst.name}")
            
            # Copy side-by-side
            side_src = results_dir / "side_by_side.png"
            side_dst = output_dir / f"{test_id}_side_by_side.png"
            if side_src.exists():
                shutil.copy2(side_src, side_dst)
                print(f"  Saved: {side_dst.name}")
            
            results.append({
                'id': test_id,
                'fen': fen,
                'viewpoint': args.viewpoint
            })
            
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Generated {len(results)} test positions")
    print(f"Results saved to: {output_dir}")
    print(f"\nTest positions:")
    for r in results:
        print(f"  {r['id']}: {r['fen']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
