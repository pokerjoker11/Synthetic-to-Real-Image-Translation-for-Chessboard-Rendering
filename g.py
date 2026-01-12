import os
import shutil
import glob

root = "game10/images"
indices = [0, 3604, 3644, 3940, 4048, 4176, 4476, 4552, 4756, 4928, 4980, 5020, 5064, 5584, 5708, 6200, 6416, 6784, 7636,
                           7740, 7832, 8936, 9728, 11912, 13072, 15284, 15744, 18768, 19412, 20132, 20792, 23040, 23988, 25216, 25512,
                           25912, 26640, 28828, 29160, 29976, 30240, 30456, 30948, 31648, 31988, 32828, 33880, 34836, 35328, 36032, 36384,
                           36800, 37476, 39439, 39520]
output_dir = "selected_frames"  # all copies go here

os.makedirs(output_dir, exist_ok=True)

for x in indices:
    x_str = f"{x:06d}"  # zero-pad to 6 digits
    pattern = os.path.join(root, f"frame_{x_str}.*")
    matches = glob.glob(pattern)

    if not matches:
        print(f"Warning: No frame found for index {x}")
        continue

    for src_file in matches:
        filename = os.path.basename(src_file)
        dst_file = os.path.join(output_dir, filename)
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {filename} -> {output_dir}")