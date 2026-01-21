#!/usr/bin/env python3
"""
prepare_chessred2k_pairs.py

ChessReD2K (your JSON schema) -> rectified (warped) real board crops + optional synthetic renders + CSV pairs.

Your annotations.json schema:
- images: list of {id, file_name, path, height, width, game_id, move_id, ...}
- annotations:
    - pieces:  list of {image_id, category_id, chessboard_position, bbox, ...}
    - corners: list of {image_id, corners:{top_left,top_right,bottom_right,bottom_left}, ...}
- categories: list of {id, name}  e.g. white-pawn ... black-king, empty
- splits:
    - train/val/test
    - chessred2k: {train/val/test}

What it does:
- Reconstructs FEN piece-placement only per image_id.
- Uses 4 corner supervision per image_id to warp to a square (default 256x256).
- Filters "extreme angle" samples into real_extreme/ (and optional synth_extreme/).
- Saves CSV pairs:
    pairs_ok.csv, pairs_extreme.csv, pairs_all.csv
- Saves debug strips:
    overlay(original+quad) | warped | (optional synth)

Usage (rectify only):
  python scripts/prepare_chessred2k_pairs.py --chessred_root D:/Datasets/ChessReD2K --out_root data/chessred2k_rect

Usage (use nested split):
  python scripts/prepare_chessred2k_pairs.py --chessred_root D:/Datasets/ChessReD2K --out_root data/chessred2k_rect --split_path chessred2k/train

Usage (with synth rendering):
  python scripts/prepare_chessred2k_pairs.py --chessred_root D:/Datasets/ChessReD2K --out_root data/chessred2k_rect \
    --render_synth \
    --blender_cmd_template "\"C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe\" assets\\chess-set.blend --background --python assets\\chess_position_api_v2.py -- --fen \"{fen}\" --view white --resolution {size} --samples 64 --seed {seed} --output \"{out}\""
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import random

# ----------------------------
# Utils
# ----------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _imread_rgb(path: Path) -> Optional[np.ndarray]:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _imwrite_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def _draw_corners_overlay(img_rgb: np.ndarray, corners_xy: np.ndarray) -> np.ndarray:
    """
    corners_xy: (4,2) ordered tl,tr,br,bl
    """
    out = img_rgb.copy()
    pts = corners_xy.astype(int).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
    for i, (x, y) in enumerate(corners_xy.astype(int)):
        cv2.circle(out, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(out, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def _order_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    """
    Robust geometric ordering. Do NOT trust the dataset's corner-name labels.
    Input corners: (4,2) unordered quad points
    Output: (4,2) ordered [tl, tr, br, bl]
    """
    c = corners.astype(np.float32)
    s = c.sum(axis=1)
    diff = c[:, 0] - c[:, 1]

    tl = c[np.argmin(s)]
    br = c[np.argmax(s)]
    tr = c[np.argmax(diff)]
    bl = c[np.argmin(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def _quad_area(corners: np.ndarray) -> float:
    return float(cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32)))


def _side_lengths(corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    return np.array([
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(bl - br),
        np.linalg.norm(tl - bl),
    ], dtype=np.float32)


def _diagonal_lengths(corners: np.ndarray) -> Tuple[float, float]:
    tl, tr, br, bl = corners
    d1 = float(np.linalg.norm(br - tl))
    d2 = float(np.linalg.norm(bl - tr))
    return d1, d2


def _corner_angles_degrees(corners: np.ndarray) -> np.ndarray:
    """
    Interior angles (degrees) at each quad corner.
    corners must be ordered tl,tr,br,bl
    """
    c = corners.astype(np.float32)
    angles = []
    for i in range(4):
        p_prev = c[(i - 1) % 4]
        p = c[i]
        p_next = c[(i + 1) % 4]

        v1 = p_prev - p
        v2 = p_next - p
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cosang = float(np.dot(v1, v2) / denom)
        cosang = max(-1.0, min(1.0, cosang))
        ang = np.degrees(np.arccos(cosang))
        angles.append(ang)
    return np.array(angles, dtype=np.float32)


def _warp_to_square(img_rgb: np.ndarray, corners_tltrbrbl: np.ndarray, out_size: int) -> np.ndarray:
    """
    corners_tltrbrbl must be ordered [tl, tr, br, bl]
    """
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners_tltrbrbl.astype(np.float32), dst)
    warped = cv2.warpPerspective(img_rgb, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
    return warped


# ----------------------------
# FEN reconstruction from pieces
# ----------------------------

def _parse_square(square: str) -> Optional[Tuple[int, int]]:
    """
    "e4" -> (file_idx, rank_idx)
    file_idx: 0..7 for a..h
    rank_idx: 0..7 for 1..8
    """
    if not isinstance(square, str):
        return None
    square = square.strip().lower()
    if len(square) != 2:
        return None
    f, r = square[0], square[1]
    if f < "a" or f > "h":
        return None
    if r < "1" or r > "8":
        return None
    return (ord(f) - ord("a"), int(r) - 1)


def _category_name_to_fen_char(name: str) -> Optional[str]:
    """
    e.g. "white-pawn" -> "P", "black-king" -> "k", "empty" -> None
    """
    if not isinstance(name, str) or not name.strip():
        return None

    n = name.strip().lower().replace("-", "_").replace(" ", "_")

    if n in {"empty", "background"}:
        return None

    # color
    is_white = "white" in n
    is_black = "black" in n
    if is_white and is_black:
        return None
    if not is_white and not is_black:
        return None

    # piece type
    if "pawn" in n:
        p = "p"
    elif "knight" in n:
        p = "n"
    elif "bishop" in n:
        p = "b"
    elif "rook" in n:
        p = "r"
    elif "queen" in n:
        p = "q"
    elif "king" in n:
        p = "k"
    else:
        return None

    return p.upper() if is_white else p.lower()


def pieces_to_fen_placement(pieces: List[Tuple[str, str]]) -> Optional[str]:
    """
    pieces: list of (square, fen_char), e.g. [("e4","P"), ("a8","k")]
    Returns piece placement field:
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    """
    board = [["" for _ in range(8)] for _ in range(8)]  # rank 1 at index 0

    for sq, ch in pieces:
        pos = _parse_square(sq)
        if pos is None:
            continue
        f, r = pos
        if ch is None:
            continue
        board[r][f] = ch

    ranks = []
    for r in range(7, -1, -1):  # 8 -> 1
        empties = 0
        out = []
        for f in range(8):
            cell = board[r][f]
            if cell == "":
                empties += 1
            else:
                if empties > 0:
                    out.append(str(empties))
                    empties = 0
                out.append(cell)
        if empties > 0:
            out.append(str(empties))
        ranks.append("".join(out))

    placement = "/".join(ranks)
    if placement == "8/8/8/8/8/8/8/8":
        return None
    return placement

def fen_piece_map(fen_placement: str) -> Dict[Tuple[int,int], str]:
    """
    Returns {(file, rank): piece_char} where file=0..7 (a..h), rank=0..7 (1..8)
    rank=0 is rank1 (bottom in white POV)
    """
    rows = fen_placement.split("/")
    if len(rows) != 8:
        return {}
    out = {}
    for r_top, row in enumerate(rows):  # r_top=0 is rank8
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                rank = 7 - r_top  # convert to 0=rank1
                out[(file, rank)] = ch
                file += 1
    return out


def rotate_k(img: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    if k == 3:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def score_rotation_by_fen_brightness(warped_rgb: np.ndarray, fen_placement: str) -> Tuple[int, float]:
    """
    Try 4 rotations, score each by (mean_brightness(white-piece-squares) - mean_brightness(black-piece-squares)).
    Pick the best rotation k.
    """
    pm = fen_piece_map(fen_placement)
    if not pm:
        return 0, 0.0

    S = warped_rgb.shape[0]
    # sample tiny patches around square centers
    patch = max(3, S // 64)

    # precompute square centers in image coords (white POV indexing)
    def square_center(file, rank):
        # file 0..7 left->right, rank 0..7 bottom->top
        cx = int((file + 0.5) * S / 8)
        cy = int((7 - rank + 0.5) * S / 8)  # rank7(top) small y
        return cx, cy

    best_k, best_score = 0, -1e9

    for k in range(4):
        imgk = rotate_k(warped_rgb, k)
        gray = cv2.cvtColor(imgk, cv2.COLOR_RGB2GRAY)

        whites, blacks = [], []
        for (f, r), ch in pm.items():
            cx, cy = square_center(f, r)
            x0, x1 = max(0, cx - patch), min(S, cx + patch + 1)
            y0, y1 = max(0, cy - patch), min(S, cy + patch + 1)
            val = float(np.mean(gray[y0:y1, x0:x1]))

            if ch.isupper():
                whites.append(val)
            else:
                blacks.append(val)

        # if one side missing, don’t gamble
        if len(whites) < 2 or len(blacks) < 2:
            score = -1e6
        else:
            score = (sum(whites) / len(whites)) - (sum(blacks) / len(blacks))

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, float(best_score)


# ----------------------------
# Dataset parsing (YOUR schema)
# ----------------------------

@dataclass
class Sample:
    image_id: int
    image_path: Path
    fen_placement: str
    corners_tltrbrbl: np.ndarray  # ordered (4,2)
    game_id: Optional[int] = None
    move_id: Optional[int] = None


def resolve_image_path(chessred_root: Path, image_rec: Dict[str, Any]) -> Optional[Path]:
    """
    Resolves the file using:
      - path
      - file_name
    and fallback basename search.
    """
    candidates = []
    p = image_rec.get("path", "")
    fn = image_rec.get("file_name", "")

    if isinstance(p, str) and p.strip():
        candidates.append(p.strip())
    if isinstance(fn, str) and fn.strip():
        candidates.append(fn.strip())

    for rel in candidates:
        rel = rel.replace("\\", "/")
        # direct
        img_path = chessred_root / rel
        if img_path.exists():
            return img_path
        # under images/
        img_path = chessred_root / "images" / rel
        if img_path.exists():
            return img_path
        # basename search
        matches = list(chessred_root.rglob(Path(rel).name))
        if matches:
            return matches[0]

    return None


def get_allowed_ids_from_split(data: Dict[str, Any], split_path: str) -> Optional[set]:
    """
    split_path examples:
      "all" (meaning no filtering)
      "train"
      "val"
      "test"
      "chessred2k/train"
      "chessred2k/val"
      "chessred2k/test"
    """
    if split_path == "all":
        return None

    splits = data.get("splits", {})
    if not isinstance(splits, dict):
        return None

    # navigate split_path
    node: Any = splits
    for part in split_path.split("/"):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Split path '{split_path}' not found in splits.")
        node = node[part]

    if not isinstance(node, dict) or "image_ids" not in node:
        raise ValueError(f"Split path '{split_path}' does not contain 'image_ids'.")

    ids = node.get("image_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"Split '{split_path}' image_ids is not a list.")
    return set(int(x) for x in ids)


def load_samples(chessred_root: Path, split_path: str = "all") -> List[Sample]:
    ann_path = chessred_root / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Could not find annotations.json at: {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("annotations.json is not a dict")

    allowed_ids = get_allowed_ids_from_split(data, split_path)

    # images
    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError("data['images'] must be a list")


    img_by_id: Dict[int, Dict[str, Any]] = {}
    for im in images:
        if isinstance(im, dict) and "id" in im:
            iid = int(im["id"])
            if allowed_ids is not None and iid not in allowed_ids:
                continue
            img_by_id[iid] = im

    # categories
    categories = data.get("categories", [])
    if not isinstance(categories, list):
        categories = []
    cat_id_to_fen: Dict[int, Optional[str]] = {}
    for c in categories:
        if not isinstance(c, dict) or "id" not in c:
            continue
        cid = int(c["id"])
        name = c.get("name", "")
        cat_id_to_fen[cid] = _category_name_to_fen_char(name)

    # annotations
    ann = data.get("annotations", {})
    if not isinstance(ann, dict):
        raise ValueError("data['annotations'] must be a dict with keys: pieces, corners")

    pieces_list = ann.get("pieces", [])
    corners_list = ann.get("corners", [])
    if not isinstance(pieces_list, list) or not isinstance(corners_list, list):
        raise ValueError("data['annotations']['pieces'] and ['corners'] must be lists")

    # image_id -> corners
    corners_by_image: Dict[int, np.ndarray] = {}
    for c in corners_list:
        if not isinstance(c, dict):
            continue
        image_id = c.get("image_id", None)
        if image_id is None:
            continue
        image_id = int(image_id)
        if image_id not in img_by_id:
            continue

        corners_obj = c.get("corners", None)
        if not isinstance(corners_obj, dict):
            continue

        try:
            # Collect 4 points (ignore their labels because labels can be inconsistent)
            pts = np.array(list(corners_obj.values()), dtype=np.float32)  # (4,2)
            if pts.shape != (4, 2):
                continue
            pts = _order_corners_tl_tr_br_bl(pts)

            if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.float32)):
                continue

            corners_by_image[image_id] = pts
        except Exception:
            continue

    # image_id -> list of (square, fen_char)
    pieces_by_image: Dict[int, List[Tuple[str, str]]] = {}
    for p in pieces_list:
        if not isinstance(p, dict):
            continue
        image_id = p.get("image_id", None)
        category_id = p.get("category_id", None)
        sq = p.get("chessboard_position", None)

        if image_id is None or category_id is None or sq is None:
            continue

        image_id = int(image_id)
        if image_id not in img_by_id:
            continue

        category_id = int(category_id)

        fen_ch = cat_id_to_fen.get(category_id, None)
        if fen_ch is None:
            # 'empty' or unknown category -> ignore
            continue

        pieces_by_image.setdefault(image_id, []).append((str(sq), fen_ch))

    samples: List[Sample] = []
    for image_id, img_rec in img_by_id.items():
        if image_id not in corners_by_image:
            continue
        if image_id not in pieces_by_image:
            continue

        fen_placement = pieces_to_fen_placement(pieces_by_image[image_id])
        if fen_placement is None:
            continue

        img_path = resolve_image_path(chessred_root, img_rec)
        if img_path is None or not img_path.exists():
            continue

        game_id = img_rec.get("game_id", None)
        move_id = img_rec.get("move_id", None)

        samples.append(Sample(
            image_id=image_id,
            image_path=img_path,
            fen_placement=fen_placement,
            corners_tltrbrbl=corners_by_image[image_id],
            game_id=int(game_id) if game_id is not None else None,
            move_id=int(move_id) if move_id is not None else None,
        ))

    return samples


# ----------------------------
# Extreme angle filtering
# ----------------------------

@dataclass
class AngleFilterConfig:
    min_area_ratio: float = 0.18   # board quad area / image area
    max_side_ratio: float = 2.5    # max side length / min side length
    min_angle_deg: float = 25.0    # minimum interior angle allowed
    max_diag_ratio: float = 2.2    # max diagonal ratio allowed


def classify_angle(img_w: int, img_h: int, corners_tltrbrbl: np.ndarray, cfg: AngleFilterConfig) -> Tuple[bool, Dict[str, float]]:
    corners = corners_tltrbrbl.astype(np.float32)

    area = _quad_area(corners)
    area_ratio = area / float(img_w * img_h + 1e-9)

    sides = _side_lengths(corners)
    side_ratio = float(np.max(sides) / (np.min(sides) + 1e-9))

    angles = _corner_angles_degrees(corners)
    min_angle = float(np.min(angles))

    d1, d2 = _diagonal_lengths(corners)
    diag_ratio = float(max(d1, d2) / (min(d1, d2) + 1e-9))

    is_extreme = (
        area_ratio < cfg.min_area_ratio or
        side_ratio > cfg.max_side_ratio or
        min_angle < cfg.min_angle_deg or
        diag_ratio > cfg.max_diag_ratio
    )

    metrics = {
        "area_ratio": area_ratio,
        "side_ratio": side_ratio,
        "min_angle_deg": min_angle,
        "diag_ratio": diag_ratio,
    }
    return is_extreme, metrics


# ----------------------------
# Optional synthetic rendering
# ----------------------------

def render_synth_with_template(
    fen_placement: str,
    out_path: Path,
    cmd_template: str,
    image_size: int,
    seed: int,
    view: str
) -> bool:
    """
    Calls an external command to render a synthetic image.

    cmd_template can contain:
      {fen} {out} {size} {seed} {view}

    If template doesn't use {view}, that's fine (extra kwargs are ignored).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = cmd_template.format(
        fen=fen_placement,
        out=str(out_path),
        size=image_size,
        seed=seed,
        view=view
    )

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return False
    except Exception:
        return False

    return out_path.exists()



def build_pov_map(samples, policy: str, seed: int, black_ratio: float) -> dict:
    """
    Returns: key -> 'white'/'black'
    key is game_id when available, otherwise image_id.
    """
    keys = []
    for s in samples:
        k = s.game_id if s.game_id is not None else s.image_id
        keys.append(k)
    uniq = sorted(set(keys))

    if policy == "keep":
        return {k: "white" for k in uniq}

    if policy == "alternate":
        # stable: even -> white, odd -> black
        return {k: ("black" if (int(k) % 2 == 1) else "white") for k in uniq}

    if policy == "random":
        rng = random.Random(seed)
        return {k: ("black" if rng.random() < black_ratio else "white") for k in uniq}

    raise ValueError(f"Unknown pov_policy: {policy}")


def rotate180(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_rgb, cv2.ROTATE_180)


def template_uses_view(cmd_template: str) -> bool:
    # Simple detection for Python format placeholders
    return ("{view}" in cmd_template) or ("{view:" in cmd_template)

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chessred_root", type=str, required=True,
                    help="ChessReD2K root (contains annotations.json + images/...)")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output folder for warped real + optional synth + csvs")
    ap.add_argument("--split_path", type=str, default="all",
                    help="Split path inside splits. Examples: all, train, val, test, chessred2k/train")

    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--debug_keep", type=int, default=120,
                    help="How many debug images to save (0 disables)")

    # filtering thresholds
    ap.add_argument("--min_area_ratio", type=float, default=0.18)
    ap.add_argument("--max_side_ratio", type=float, default=2.5)
    ap.add_argument("--min_angle_deg", type=float, default=25.0)
    ap.add_argument("--max_diag_ratio", type=float, default=2.2)

    # synth rendering (optional)
    ap.add_argument("--render_synth", action="store_true",
                    help="Also render synth images for each FEN placement")
    ap.add_argument("--blender_cmd_template", type=str, default="",
                    help="Command template with {fen},{out},{size},{seed} and optionally {view}")
    ap.add_argument("--seed", type=int, default=0)

    # game flipping policy (AFTER canonicalization)
    ap.add_argument("--pov_policy", type=str, default="alternate",
                    choices=["keep", "alternate", "random"],
                    help="Flip policy per game after canonical orientation: keep=all white, alternate=odd black, random=per-game random")
    ap.add_argument("--pov_seed", type=int, default=0, help="Seed for random POV assignment")
    ap.add_argument("--pov_black_ratio", type=float, default=0.5,
                    help="For random POV: probability a game is assigned black POV")

    args = ap.parse_args()

    chessred_root = Path(args.chessred_root)
    out_root = Path(args.out_root)

    ok_real_dir = out_root / "real_ok"
    ex_real_dir = out_root / "real_extreme"
    ok_synth_dir = out_root / "synth_ok"
    ex_synth_dir = out_root / "synth_extreme"
    debug_dir = out_root / "debug"

    _safe_mkdir(ok_real_dir)
    _safe_mkdir(ex_real_dir)
    _safe_mkdir(ok_synth_dir)
    _safe_mkdir(ex_synth_dir)
    _safe_mkdir(debug_dir)

    cfg = AngleFilterConfig(
        min_area_ratio=args.min_area_ratio,
        max_side_ratio=args.max_side_ratio,
        min_angle_deg=args.min_angle_deg,
        max_diag_ratio=args.max_diag_ratio,
    )

    samples = load_samples(chessred_root, split_path=args.split_path)
    if not samples:
        raise SystemExit(
            "[ERROR] No usable samples found.\n"
            "Common reasons:\n"
            "  - images[].path/file_name doesn't resolve to real files\n"
            "  - annotations.corners missing for those image_ids\n"
            "  - annotations.pieces missing chessboard_position\n"
            "  - categories names not mapped (but yours are fine)\n"
        )

    # Assign which games get flipped AFTER canonicalization
    pov_map = build_pov_map(samples, args.pov_policy, args.pov_seed, args.pov_black_ratio)

    from collections import Counter
    print("Flip policy (pov_policy):", args.pov_policy)
    print("Assigned POV distribution:", Counter(pov_map.values()))
    print("Example assignments:", list(pov_map.items())[:10])

    uses_view = template_uses_view(args.blender_cmd_template) if args.blender_cmd_template else False

    rows_ok, rows_ex, rows_all = [], [], []
    debug_left = args.debug_keep

    for idx, s in enumerate(samples):
        img_rgb = _imread_rgb(s.image_path)
        if img_rgb is None:
            continue

        h, w = img_rgb.shape[:2]

        # classify angle based on original image + quad
        is_extreme, metrics = classify_angle(w, h, s.corners_tltrbrbl, cfg)

        # 1) rectify
        try:
            warped = _warp_to_square(img_rgb, s.corners_tltrbrbl, args.image_size)
        except Exception:
            continue

        # 2) canonicalize rotation using FEN (choose among 0/90/180/270)
        try:
            k_best, rot_score = score_rotation_by_fen_brightness(warped, s.fen_placement)
            warped = rotate_k(warped, k_best)
        except Exception:
            # if anything goes weird, fallback to no extra rotation
            k_best, rot_score = 0, 0.0

        # 3) AFTER canonicalization: flip some games
        game_key = s.game_id if s.game_id is not None else s.image_id
        desired_pov = pov_map.get(game_key, "white")  # white = keep, black = flip 180
        did_flip180 = (desired_pov == "black")
        if did_flip180:
            warped = rotate180(warped)

        # naming
        sample_id = f"img{int(s.image_id):07d}"
        if s.game_id is not None and s.move_id is not None:
            sample_id = f"g{s.game_id:02d}_m{s.move_id:04d}_{sample_id}"
        sample_id = sample_id + ("_bpov" if did_flip180 else "_wpov")

        if is_extreme:
            real_out = ex_real_dir / f"{sample_id}.png"
            synth_out = ex_synth_dir / f"{sample_id}.png"
            bucket = "extreme"
        else:
            real_out = ok_real_dir / f"{sample_id}.png"
            synth_out = ok_synth_dir / f"{sample_id}.png"
            bucket = "ok"

        _imwrite_rgb(real_out, warped)

        # Synth rendering:
        # - canonical orientation is "white"
        # - if we flipped this game: view should be "black" (or rotate synth if no {view})
        synth_written = False
        synth_view = "black" if did_flip180 else "white"

        if args.render_synth and args.blender_cmd_template.strip():
            synth_written = render_synth_with_template(
                fen_placement=s.fen_placement,
                out_path=synth_out,
                cmd_template=args.blender_cmd_template,
                image_size=args.image_size,
                seed=args.seed + idx,
                view=synth_view,
            )

            # If the template does NOT support {view}, rotate synth only when we flipped the real
            if synth_written and did_flip180 and (not uses_view):
                synth_img = _imread_rgb(synth_out)
                if synth_img is not None:
                    synth_img = rotate180(synth_img)
                    _imwrite_rgb(synth_out, synth_img)

        # debug strip: overlay | warped | (optional synth)
        if debug_left > 0:
            overlay = _draw_corners_overlay(img_rgb, s.corners_tltrbrbl)
            overlay_small = cv2.resize(overlay, (args.image_size, args.image_size),
                                       interpolation=cv2.INTER_AREA)

            parts = [overlay_small, warped]
            if synth_written and synth_out.exists():
                synth_img = _imread_rgb(synth_out)
                if synth_img is not None:
                    parts.append(synth_img)

            debug_strip = np.concatenate(parts, axis=1)
            dbg_name = f"{bucket}_{idx:05d}_{sample_id}.png"
            _imwrite_rgb(debug_dir / dbg_name, debug_strip)
            debug_left -= 1

        row = {
            "id": sample_id,
            "bucket": bucket,
            "image_id": int(s.image_id),
            "game_id": s.game_id if s.game_id is not None else "",
            "move_id": s.move_id if s.move_id is not None else "",
            "fen": s.fen_placement,  # piece-placement field only
            "real_path": str(real_out).replace("\\", "/"),
            "synth_path": str(synth_out).replace("\\", "/") if (args.render_synth and synth_written) else "",
            "orig_image": str(s.image_path).replace("\\", "/"),

            # metrics
            **metrics,

            # rotation info
            "canonical_rot_k": int(k_best),
            "canonical_rot_score": float(rot_score),
            "pov": desired_pov,
            "flip180_applied": bool(did_flip180),
        }

        rows_all.append(row)
        (rows_ex if bucket == "extreme" else rows_ok).append(row)

    df_all = pd.DataFrame(rows_all)
    df_ok = pd.DataFrame(rows_ok)
    df_ex = pd.DataFrame(rows_ex)

    df_all.to_csv(out_root / "pairs_all.csv", index=False)
    df_ok.to_csv(out_root / "pairs_ok.csv", index=False)
    df_ex.to_csv(out_root / "pairs_extreme.csv", index=False)

    print(f"[DONE] total={len(df_all)} ok={len(df_ok)} extreme={len(df_ex)}")
    print(f"[OUT]  {out_root}")
    print("[NOTE] Check debug/ to confirm warps + canonical rotation look correct.")
    print("[NOTE] FEN stored is ONLY piece placement (first field of full FEN).")


if __name__ == "__main__":
    main()
