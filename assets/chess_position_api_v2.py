"""
Chess FEN Parser - Auto-detect Starting Positions

Strategy:
1. Analyze current piece positions to determine which piece is on which square
2. Parse target FEN
3. Move pieces from their detected starting square to target square

Usage:
    blender chess-set.blend --background --python chess_position_api_v2.py -- --fen "r4rk1/1p1bqppp/n1p1pn2/p2pN3/2PP4/P1N3P1/1P1QPPBP/R4RK1" --view black
"""

import bpy
import math
from mathutils import Vector
import sys
import argparse
# Rotate the offset vector
from mathutils import Matrix
import os

# ==========================
# CONFIG
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2
DESIRED_ANGLE_DEGREES = 25
LENS = 26
RES = 1024
SAMPLES = 128
OUT_DIR = "//renders"


def get_board_info():
    """Get board dimensions"""
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")

    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8

    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)

    scale_factor = board_size / REAL_BOARD_SIZE

    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'board_size': board_size,
        'scale_factor': scale_factor,
    }


def position_to_square(pos, board_info):
    """Convert 3D position to chess square (e.g., 'e2')"""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']

    # File (a-h) from X coordinate - scene is flipped
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    # Rank (1-8) from Y coordinate
    # Higher Y = lower rank (reversed)
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1

    return f"{file_letter}{rank_number}"


def detect_starting_positions(board_info):
    """
    Detect which piece is on which square currently
    Returns: {piece_name: {'square': 'e2', 'piece_type': 'P'}}
    """
    print("\n" + "=" * 70)
    print("DETECTING STARTING POSITIONS")
    print("=" * 70)

    pieces = {}

    # Get all chess piece objects
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        name = obj.name

        # Determine piece type from name
        piece_type = None

        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']:
            piece_type = 'P'  # White pawn
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']:
            piece_type = 'p'  # Black pawn
        elif 'rook' in name.lower():
            piece_type = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower():
            piece_type = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower():
            piece_type = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower():
            piece_type = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower():
            piece_type = 'K' if 'white' in name.lower() else 'k'

        if piece_type:
            # Use object origin as position
            pos = obj.matrix_world.translation
            square = position_to_square(pos, board_info)

            pieces[name] = {
                'square': square,
                'piece_type': piece_type,
                'obj': obj,
            }

            print(f"  {name:20s} → {square:4s} ({piece_type})")

    print(f"\n✓ Detected {len(pieces)} pieces")
    return pieces


def parse_fen(fen):
    """
    Parse FEN string into dict of positions
    Returns: { 'e4': 'P', ... }
    """
    board_part = fen.split()[0]
    ranks = board_part.split('/')

    positions = {}
    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        for char in rank_str:
            if char.isdigit():
                file_idx += int(char)
            else:
                file_letter = chr(ord('a') + file_idx)
                rank_number = 8 - rank_idx
                square = f"{file_letter}{rank_number}"
                positions[square] = char
                file_idx += 1

    return positions


def square_to_position(square, board_info):
    """
    Convert square (e.g., 'e2') to world position (Vector)
    """
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']

    file_letter = square[0]
    rank_number = int(square[1])

    file_idx = ord(file_letter) - ord('a')
    # Scene is flipped: file_idx 0 (a) should map to max x, file_idx 7 (h) to min x
    x = plane_min.x + (7 - file_idx + 0.5) * square_size

    rank_idx = rank_number - 1
    y = plane_max.y - (rank_idx + 0.5) * square_size

    # z stays same (piece sits on board)
    z = plane_min.z

    return Vector((x, y, z))


def apply_fen(starting_pieces, fen_positions, board_info):
    """
    Move pieces to match target FEN using detected starting positions.
    """
    print("\n" + "=" * 70)
    print("APPLYING FEN")
    print("=" * 70)
    print(f"FEN: {fen_str}\n")

    # Group pieces by type for matching
    available_by_type = {}
    for piece_name, info in starting_pieces.items():
        ptype = info['piece_type']
        available_by_type.setdefault(ptype, []).append(piece_name)

    pieces_used = set()

    # For each target square in the fen, assign a piece
    for target_square, target_piece_type in fen_positions.items():
        candidates = available_by_type.get(target_piece_type, [])
        if not candidates:
            print(f"  [WARN] No available piece for {target_piece_type} at {target_square}")
            continue

        # Pick a piece that isn't used yet
        piece_name = None
        for c in candidates:
            if c not in pieces_used:
                piece_name = c
                break
        if piece_name is None:
            print(f"  [WARN] Ran out of pieces for {target_piece_type} at {target_square}")
            continue

        piece_obj = starting_pieces[piece_name]['obj']
        from_square = starting_pieces[piece_name]['square']

        # Move to target square
        target_pos = square_to_position(target_square, board_info)
        piece_obj.location.x = target_pos.x
        piece_obj.location.y = target_pos.y
        # keep original z
        pieces_used.add(piece_name)

        if from_square != target_square:
            print(f"  Moved {piece_name:20s} {from_square} → {target_square}")
        else:
            print(f"  Kept {piece_name:20s} at {target_square}")

    # Hide unused pieces (captured)
    for piece_name in starting_pieces.keys():
        if piece_name not in pieces_used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True
                print(f"  Hidden {piece_name}")

    print(f"\n✓ Position set ({len(pieces_used)} pieces visible)")


def render_all_views(board_info, view='black'):
    """Render views from white or black perspective"""
    print("\n" + "=" * 70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("=" * 70)

    center = board_info['center']
    scale_factor = board_info['scale_factor']

    board_size = board_info.get('board_size', REAL_BOARD_SIZE * scale_factor)
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)

    # Clean cameras
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Setup lighting
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light_height = center.z + camera_height * 2
        bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
        bpy.context.active_object.data.energy = 3.0

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.use_denoising = True

    try:
        scene.cycles.device = 'GPU'
    except:
        pass

    # Camera positions
    camera_z = center.z + camera_height

    # Flip camera positions for white view (180 degree rotation)
    if view == 'white':
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x + horizontal_offset, center.y, camera_z), "2_east", False),
            ((center.x - horizontal_offset, center.y, camera_z), "3_west", False),
        ]
        z_rotation_offset = math.radians(180)
    else:  # black view (default)
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x - horizontal_offset, center.y, camera_z), "2_west", False),
            ((center.x + horizontal_offset, center.y, camera_z), "3_east", False),
        ]
        z_rotation_offset = 0

    for location, name, point_at_center in views:
        print(f"\nRendering: {name}")

        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object

        if point_at_center:
            direction = center - cam.location
            cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        else:
            cam.rotation_euler = (0, 0, 0)

        # Apply rotation for white/black view
        cam.rotation_euler.z += z_rotation_offset

        cam.data.lens = LENS

        # Tight overhead framing: the dataset expects the chessboard to fill most of the frame.
        # Use an orthographic overhead camera and fit to the outer frame size.
        if name == "1_overhead":
            cam.data.type = 'ORTHO'
            # ortho_scale is the width of the camera view in world units (square render => width==height)
            cam.data.ortho_scale = board_size * 1.08  # small margin around the frame
            cam.data.clip_start = 0.01
            cam.data.clip_end = 100.0

        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = f"{OUT_DIR}/{name}.png"
        bpy.ops.render.render(write_still=True)

        print(f"  ✓ Saved: {name}.png")

        bpy.data.objects.remove(cam, do_unlink=True)

    print("\n✓ Rendering complete")


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    parser.add_argument('--resolution', type=int, default=800)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'],
                        help='Render from white or black perspective')
    parser.add_argument('--out_dir', type=str, default="./renders")

    args = parser.parse_args(argv)

    global RES, SAMPLES, OUT_DIR, fen_str
    RES = args.resolution
    SAMPLES = args.samples
    OUT_DIR = args.out_dir
    try:
        OUT_DIR = bpy.path.abspath(OUT_DIR)
    except Exception:
        pass
    os.makedirs(OUT_DIR, exist_ok=True)
    fen_str = args.fen

    # Get board info
    board_info = get_board_info()

    # Fix inverted board - rotate checkerboard 90 degrees around board center
    plane = bpy.data.objects.get("Black & white")
    if plane:
        # Get board center first (before rotating)
        frame = bpy.data.objects.get("Outer frame")
        frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
        frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
        frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
        center = (frame_min + frame_max) / 2

        # Store original position
        original_pos = plane.location.copy()

        # Move to center, rotate, move back
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)

        rot_mat = Matrix.Rotation(math.radians(90), 4, 'Z')
        new_offset = rot_mat @ offset

        plane.location = center + new_offset

    # Detect starting positions and apply FEN
    starting_pieces = detect_starting_positions(board_info)
    fen_positions = parse_fen(fen_str)
    apply_fen(starting_pieces, fen_positions, board_info)

    # Render
    render_all_views(board_info, view=args.view)


if __name__ == "__main__":
    main()
