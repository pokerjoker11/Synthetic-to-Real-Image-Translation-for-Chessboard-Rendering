import os
import re
import shutil
import chess
import chess.pgn

def get_fens_from_pgn(pgn_path):
    with open(pgn_path, 'r') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
    board = game.board()
    fens = [board.fen()]
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.fen())
    return fens

def generate_labels_single_pass(pgn_path, frames_dir, state_start_indices):
    fens = get_fens_from_pgn(pgn_path)
    if not fens or not os.path.exists(frames_dir):
        return

    all_files = [f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]
    frame_data = []
    for f in all_files:
        match = re.search(r'frame_(\d+)', f)
        if match:
            frame_data.append((int(match.group(1)), f))
    
    frame_data.sort()

    current_state_idx = 0
    num_fens = len(fens)
    num_indices = len(state_start_indices)

    for frame_num, filename in frame_data:
        while current_state_idx + 1 < num_indices and frame_num >= state_start_indices[current_state_idx + 1]:
            current_state_idx += 1
        
        if current_state_idx < num_fens:
            fen = fens[current_state_idx]
            safe_fen = fen.split()[0].replace("/", "_")
            target_dir = os.path.join(frames_dir, safe_fen)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(frames_dir, filename), os.path.join(target_dir, filename))

def main():
    PGN_FILE = r"game8\game8.pgn"
    FRAMES_DIR = r"game8\images"
    state_start_indices = [0, 224, 248, 292, 320, 376, 420, 472, 504, 708, 756, 872, 980, 1132, 1168, 1236, 1400, 1732, 1884,
    2388, 2852, 3208, 3504, 3604, 4240, 4632, 4856, 5004, 5352, 5552, 5724, 5844, 6116, 6228, 6848, 7028, 7304, 7748, 8280, 8368, 8404,
    8572, 8848, 9068, 9164, 9224, 9368, 9592, 9748, 10636, 10900, 10964, 11144, 11328, 11480, 11540, 11636, 12044,
    12484, 13020, 13664, 13844, 14088, 14364, 14780, 14848, 14988, 15032, 15152, 15268, 15672, 15828, 16772,
    17408, 17560, 17776, 17896, 18580, 19452, 19640, 19852, 20348, 20480, 20524, 21264, 21400, 21784, 22432,
    22640, 22764, 22816, 23004, 23184, 23260, 23704, 23788, 23824, 23900, 23948, 24152, 24404, 24516, 24564,
    24876, 24924, 24968, 25280, 25360, 25468, 25576, 25636, 25860, 25968, 26060, 26096, 26144, 26184, 26364,
    26604, 26656, 26724, 26840, 27532, 27704, 27832, 27940, 28012, 28312, 28372, 28444, 28488]
    generate_labels_single_pass(PGN_FILE, FRAMES_DIR, state_start_indices)

if __name__ == "__main__":
    main()