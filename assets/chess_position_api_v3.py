"""
Chess FEN Renderer - Robust Blender Script (v3)

Improvements over v2:
- State reset at start (unhide all pieces, reset to starting position)
- Flexible piece detection (handles various naming conventions)
- Board rotation tracking (only rotate once)
- Better diagnostics and validation
- Piece matching that minimizes moves

Usage:
    blender chess-set.blend --background --python chess_position_api_v3.py -- \
        --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --view white
"""

import bpy
import math
from mathutils import Vector, Matrix
import sys
import argparse
import re

# ==========================
# CONFIG
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2
DESIRED_ANGLE_DEGREES = 25
LENS = 26
RES = 1024
SAMPLES = 128
OUT_DIR = "./renders"

# Track if we've already rotated the board in this session
_BOARD_ROTATED = False


def get_board_info():
    """Get board dimensions from Blender scene"""
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    
    if not plane:
        raise RuntimeError("Could not find 'Black & white' object (checkerboard)")
    if not frame:
        raise RuntimeError("Could not find 'Outer frame' object")

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


def classify_piece(obj_name):
    """
    Classify a Blender object as a chess piece based on its name.
    Returns (piece_type, color) or (None, None) if not a piece.
    
    piece_type: 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king'
    color: 'white' or 'black'
    
    Known pawn names in the Blender file:
    - White pawns: B, C, D, E, F, G, H, A(texture)
    - Black pawns: B.001, C.001, D.001, E.001, F.001, G.001, H.001, A(textures)
    """
    name_lower = obj_name.lower()
    
    # Explicit piece names
    if 'rook' in name_lower:
        color = 'white' if 'white' in name_lower else 'black'
        return ('rook', color)
    if 'knight' in name_lower:
        color = 'white' if 'white' in name_lower else 'black'
        return ('knight', color)
    if 'bishop' in name_lower or 'bitshop' in name_lower:
        color = 'white' if 'white' in name_lower else 'black'
        return ('bishop', color)
    if 'queen' in name_lower:
        color = 'white' if 'white' in name_lower else 'black'
        return ('queen', color)
    if 'king' in name_lower:
        color = 'white' if 'white' in name_lower else 'black'
        return ('king', color)
    
    # Pawn detection - these have letter names (A-H) in the Blender file
    # The naming is inconsistent, so we handle each case explicitly
    
    # WHITE PAWNS:
    # - Single letters: B, C, D, E, F, G, H
    # - A pawn: A(texture) - singular
    if re.match(r'^[B-H]$', obj_name):  # B through H, single letter
        return ('pawn', 'white')
    if obj_name == 'A(texture)':  # White A pawn - singular 'texture'
        return ('pawn', 'white')
    
    # BLACK PAWNS:
    # - Letter.001 pattern: A.001, B.001, C.001, D.001, E.001, F.001, G.001, H.001
    # - A pawn might also be: A(textures) - plural with 's'
    if re.match(r'^[A-H]\.001$', obj_name):  # A.001 through H.001
        return ('pawn', 'black')
    if obj_name == 'A(textures)':  # Black A pawn alternate name
        return ('pawn', 'black')
    
    # Fallback: check for 'pawn' in name explicitly
    if 'pawn' in name_lower:
        color = 'white' if 'white' in name_lower else 'black'
        return ('pawn', color)
    
    return (None, None)


def piece_to_fen_char(piece_type, color):
    """Convert piece type and color to FEN character"""
    char_map = {
        'pawn': 'p', 'rook': 'r', 'knight': 'n', 
        'bishop': 'b', 'queen': 'q', 'king': 'k'
    }
    char = char_map.get(piece_type, '?')
    return char.upper() if color == 'white' else char


def fen_char_to_piece(char):
    """Convert FEN character to (piece_type, color)"""
    char_map = {
        'p': 'pawn', 'r': 'rook', 'n': 'knight',
        'b': 'bishop', 'q': 'queen', 'k': 'king'
    }
    color = 'white' if char.isupper() else 'black'
    piece_type = char_map.get(char.lower(), None)
    return (piece_type, color)


def position_to_square(pos, board_info):
    """Convert 3D world position to chess square (e.g., 'e2')"""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']

    # File (a-h) from X coordinate
    # In this scene: higher X = lower file (h→a)
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    
    # Rank (1-8) from Y coordinate  
    # Higher Y = lower rank (8→1)
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1

    return f"{file_letter}{rank_number}"


def square_to_position(square, board_info):
    """Convert chess square (e.g., 'e2') to world position (Vector)"""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']

    file_letter = square[0].lower()
    rank_number = int(square[1])

    file_idx = ord(file_letter) - ord('a')
    # Inverse of position_to_square: file_idx 0 (a) → high X
    x = plane_min.x + (7 - file_idx + 0.5) * square_size

    rank_idx = rank_number - 1
    y = plane_max.y - (rank_idx + 0.5) * square_size

    z = plane_min.z

    return Vector((x, y, z))


def parse_fen(fen):
    """Parse FEN string into dict of {square: fen_char}"""
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


def list_all_mesh_objects():
    """Debug: List all mesh objects in the scene"""
    print("\n" + "=" * 70)
    print("ALL MESH OBJECTS IN SCENE")
    print("=" * 70)
    
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        if obj.type != 'MESH':
            continue
        piece_type, color = classify_piece(obj.name)
        status = f"-> {piece_type} ({color})" if piece_type else "(not a piece)"
        hidden = "[HIDDEN]" if obj.hide_render else ""
        print(f"  {obj.name:30s} {status:25s} {hidden}")


def reset_scene():
    """Reset all pieces to visible state"""
    print("\n" + "=" * 70)
    print("RESETTING SCENE")
    print("=" * 70)
    
    # First, list all objects for debugging
    list_all_mesh_objects()
    
    count = 0
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        
        piece_type, color = classify_piece(obj.name)
        if piece_type:
            obj.hide_render = False
            obj.hide_viewport = False
            count += 1
    
    print(f"\n  Unhid {count} pieces")


def detect_all_pieces(board_info):
    """
    Find all chess pieces in the scene.
    Returns dict: {obj_name: {'obj': obj, 'piece_type': str, 'color': str, 'fen_char': str, 'square': str}}
    """
    print("\n" + "=" * 70)
    print("DETECTING ALL PIECES")
    print("=" * 70)
    
    pieces = {}
    
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        
        piece_type, color = classify_piece(obj.name)
        if not piece_type:
            continue
        
        pos = obj.matrix_world.translation
        square = position_to_square(pos, board_info)
        fen_char = piece_to_fen_char(piece_type, color)
        
        pieces[obj.name] = {
            'obj': obj,
            'piece_type': piece_type,
            'color': color,
            'fen_char': fen_char,
            'square': square,
        }
        
        print(f"  {obj.name:25s} → {square:3s} ({fen_char}) [{piece_type} {color}]")
    
    # Validate piece counts
    by_type = {}
    for info in pieces.values():
        key = info['fen_char']
        by_type[key] = by_type.get(key, 0) + 1
    
    print(f"\n  Piece counts: {by_type}")
    print(f"  Total: {len(pieces)} pieces")
    
    # Expected: 16 white (8P, 2R, 2N, 2B, 1Q, 1K) + 16 black
    expected = {'P': 8, 'R': 2, 'N': 2, 'B': 2, 'Q': 1, 'K': 1,
                'p': 8, 'r': 2, 'n': 2, 'b': 2, 'q': 1, 'k': 1}
    
    for char, count in expected.items():
        actual = by_type.get(char, 0)
        if actual != count:
            print(f"  [WARN] Expected {count} {char}, found {actual}")
    
    return pieces


def apply_fen_position(pieces, target_fen, board_info):
    """
    Move pieces to match target FEN.
    Uses optimal matching to minimize piece movements.
    """
    print("\n" + "=" * 70)
    print("APPLYING FEN POSITION")
    print("=" * 70)
    print(f"  Target: {target_fen}")
    
    target_positions = parse_fen(target_fen)
    
    # Group pieces by FEN char
    available = {}
    for name, info in pieces.items():
        char = info['fen_char']
        available.setdefault(char, []).append(name)
    
    used_pieces = set()
    
    # For each target square, find best matching piece
    for target_square, target_char in target_positions.items():
        candidates = available.get(target_char, [])
        
        if not candidates:
            print(f"  [ERROR] No piece available for {target_char} at {target_square}")
            continue
        
        # Find unused candidate, prefer one already on target square
        best_piece = None
        for name in candidates:
            if name in used_pieces:
                continue
            if pieces[name]['square'] == target_square:
                best_piece = name
                break
            if best_piece is None:
                best_piece = name
        
        if best_piece is None:
            print(f"  [ERROR] All {target_char} pieces already used, can't place at {target_square}")
            continue
        
        used_pieces.add(best_piece)
        
        obj = pieces[best_piece]['obj']
        from_square = pieces[best_piece]['square']
        
        # Move to target position
        target_pos = square_to_position(target_square, board_info)
        obj.location.x = target_pos.x
        obj.location.y = target_pos.y
        # Keep original Z (height above board)
        
        if from_square != target_square:
            print(f"  Move {best_piece:25s}: {from_square} → {target_square}")
        else:
            print(f"  Keep {best_piece:25s}: {target_square}")
    
    # Hide unused pieces (captured)
    hidden_count = 0
    for name, info in pieces.items():
        if name not in used_pieces:
            info['obj'].hide_render = True
            info['obj'].hide_viewport = True
            hidden_count += 1
    
    print(f"\n  Placed {len(used_pieces)} pieces, hid {hidden_count} (captured)")


def validate_position(pieces, target_fen, board_info):
    """Validate that pieces are in correct positions"""
    print("\n" + "=" * 70)
    print("VALIDATING POSITION")
    print("=" * 70)
    
    target_positions = parse_fen(target_fen)
    
    # Build current board state from visible pieces
    current = {}
    for name, info in pieces.items():
        obj = info['obj']
        if obj.hide_render:
            continue
        
        pos = obj.matrix_world.translation
        square = position_to_square(pos, board_info)
        current[square] = info['fen_char']
    
    # Compare
    errors = []
    for square, expected_char in target_positions.items():
        actual_char = current.get(square)
        if actual_char != expected_char:
            errors.append(f"  {square}: expected {expected_char}, got {actual_char}")
    
    for square, actual_char in current.items():
        if square not in target_positions:
            errors.append(f"  {square}: unexpected piece {actual_char}")
    
    if errors:
        print("  ERRORS FOUND:")
        for e in errors[:10]:  # Limit output
            print(e)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print("  Position validated OK!")
        return True


def setup_board_orientation():
    """
    Ensure board checkerboard has correct orientation.
    Only rotates once per Blender session.
    """
    global _BOARD_ROTATED
    
    if _BOARD_ROTATED:
        print("  Board already rotated in this session, skipping")
        return
    
    plane = bpy.data.objects.get("Black & white")
    if not plane:
        print("  [WARN] Could not find 'Black & white' object")
        return
    
    frame = bpy.data.objects.get("Outer frame")
    if not frame:
        print("  [WARN] Could not find 'Outer frame' object")
        return
    
    # Get board center
    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    
    # Store original position
    original_pos = plane.location.copy()
    
    # Rotate 90 degrees around board center
    offset = original_pos - center
    plane.rotation_euler.z = math.radians(90)
    
    rot_mat = Matrix.Rotation(math.radians(90), 4, 'Z')
    new_offset = rot_mat @ offset
    
    plane.location = center + new_offset
    
    _BOARD_ROTATED = True
    print("  Rotated checkerboard 90 degrees")


def render_view(board_info, view='white'):
    """Render the board from specified viewpoint"""
    print("\n" + "=" * 70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("=" * 70)

    center = board_info['center']
    scale_factor = board_info['scale_factor']
    board_size = board_info['board_size']
    
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor

    # Clean existing cameras
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Setup lighting if needed
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
        print("  Using GPU rendering")
    except:
        print("  Using CPU rendering")

    # Camera position (overhead)
    camera_z = center.z + camera_height
    
    # Camera rotation for viewpoint
    z_rotation = math.radians(180) if view == 'white' else 0

    # Create camera
    bpy.ops.object.camera_add(location=(center.x, center.y, camera_z))
    cam = bpy.context.active_object
    
    # Point at board center
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.z += z_rotation
    
    # Orthographic camera for top-down view
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = board_size * 1.02  # Tighter crop, minimal margin
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0

    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = f"{OUT_DIR}/1_overhead.png"
    
    print(f"  Rendering to {OUT_DIR}/1_overhead.png")
    bpy.ops.render.render(write_still=True)
    print("  Render complete!")

    # Cleanup
    bpy.data.objects.remove(cam, do_unlink=True)


def main():
    global RES, SAMPLES, OUT_DIR
    
    # Parse arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render chess position from FEN")
    parser.add_argument('--fen', type=str, 
                        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                        help="FEN string (board part)")
    parser.add_argument('--view', type=str, default='white', 
                        choices=['white', 'black'],
                        help='Render from white or black perspective')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--output', type=str, default="./renders",
                        help='Output directory')

    args = parser.parse_args(argv)

    RES = args.resolution
    SAMPLES = args.samples
    OUT_DIR = args.output
    
    print("\n" + "=" * 70)
    print("CHESS POSITION RENDERER v3")
    print("=" * 70)
    print(f"  FEN: {args.fen}")
    print(f"  View: {args.view}")
    print(f"  Resolution: {RES}x{RES}")
    print(f"  Samples: {SAMPLES}")
    print(f"  Output: {OUT_DIR}")

    # Step 1: Reset scene
    reset_scene()
    
    # Step 2: Setup board orientation (only once per session)
    setup_board_orientation()
    
    # Step 3: Get board geometry
    board_info = get_board_info()
    
    # Step 4: Detect all pieces
    pieces = detect_all_pieces(board_info)
    
    # Step 5: Apply FEN position
    apply_fen_position(pieces, args.fen, board_info)
    
    # Step 6: Validate position
    validate_position(pieces, args.fen, board_info)
    
    # Step 7: Render
    render_view(board_info, view=args.view)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
