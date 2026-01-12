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

    # Keep track of how many frames we've moved per FEN folder
    fen_frame_count = {}

    for frame_num, filename in frame_data:
        while current_state_idx + 1 < num_indices and frame_num >= state_start_indices[current_state_idx + 1]:
            current_state_idx += 1

        if current_state_idx < num_fens:
            fen = fens[current_state_idx]
            safe_fen = fen.split()[0].replace("/", "_")
            target_dir = os.path.join(frames_dir, safe_fen)
            os.makedirs(target_dir, exist_ok=True)

            # Initialize counter for this FEN if it doesn't exist
            if safe_fen not in fen_frame_count:
                fen_frame_count[safe_fen] = 0

            # Only move the first 4 frames
            if fen_frame_count[safe_fen] < 4:
                shutil.move(os.path.join(frames_dir, filename), os.path.join(target_dir, filename))
                fen_frame_count[safe_fen] += 1

def main():
    PGN_FILE = r"game10\game10.pgn"
    FRAMES_DIR = r"game10\images"
    # state_start_indices_8 = [0, 224, 248, 292, 320, 376, 420, 472, 504, 708, 756, 872, 980, 1132, 1168, 1236, 1400, 1732, 1884,
    # 2388, 2852, 3208, 3504, 3604, 4240, 4632, 4856, 5004, 5352, 5552, 5724, 5844, 6116, 6228, 6848, 7028, 7304, 7748, 8280, 8368, 8404,
    # 8572, 8848, 9068, 9164, 9224, 9368, 9592, 9748, 10636, 10900, 10964, 11144, 11328, 11480, 11540, 11636, 12044,
    # 12484, 13020, 13664, 13844, 14088, 14364, 14780, 14848, 14988, 15032, 15152, 15268, 15672, 15828, 16772,
    # 17408, 17560, 17776, 17896, 18580, 19452, 19640, 19852, 20348, 20480, 20524, 21264, 21400, 21784, 22432,
    # 22640, 22764, 22816, 23004, 23184, 23260, 23704, 23788, 23824, 23900, 23948, 24152, 24404, 24516, 24564,
    # 24876, 24924, 24968, 25280, 25360, 25468, 25576, 25636, 25860, 25968, 26060, 26096, 26144, 26184, 26364,
    # 26604, 26656, 26724, 26840, 27532, 27704, 27832, 27940, 28012, 28312, 28372, 28444, 28488]
    # state_start_indices_9 = [4, 3524, 3572, 3604, 3692, 3720, 3756, 3816, 3888, 4144, 4260, 4568, 4796, 5380,
    #                        5460, 5836, 6460, 7368, 8052, 9220, 9636, 10104, 10184, 10812, 11972,
    #                        12804, 12864, 12940, 13532, 14024, 14412, 14476, 14660, 15780, 15976,
    #                        17068, 18628, 19264, 19424, 20680, 21176, 21560, 22160, 25236, 25800, 26524,
    #                        26716, 27032, 27616, 27888, 28320, 28632, 28824, 30152, 32576, 35400, 35996,
    #                        38572, 39712, 40460, 41004, 41068, 41392, 41468]
    state_start_indices = [0, 3604, 3644, 3940, 4048, 4176, 4476, 4552, 4756, 4928, 4980, 5064, 5584, 5708, 6200, 6416, 6784, 7636,
                           7740, 7832, 8936, 9728, 11912, 13072, 15284, 15744, 18768, 19412, 20132, 20792, 23040, 23988, 25216, 25512,
                           25912, 26640, 28828, 29160, 29976, 30240, 30456, 30948, 31648, 31988, 32828, 33880, 34836, 35328, 36032, 36384,
                           36800, 36952, 37372, 37476, 39436, 39520]
    generate_labels_single_pass(PGN_FILE, FRAMES_DIR, state_start_indices)

if __name__ == "__main__":
    main()