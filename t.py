import os
import shutil
import cv2

root = "game8/images"

for folder_name in os.listdir(root):
    folder_path = os.path.join(root, folder_name)

    if not os.path.isdir(folder_path):
        continue

    white_dir = os.path.join(folder_path, "white")
    black_dir = os.path.join(folder_path, "black")

    os.makedirs(white_dir, exist_ok=True)
    os.makedirs(black_dir, exist_ok=True)

    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))
             and f.lower().endswith((".jpg", ".png", ".jpeg"))]

    files.sort()
    midpoint = len(files) // 2

    for i, fname in enumerate(files):
        src = os.path.join(folder_path, fname)

        if i < midpoint:
            dst = os.path.join(white_dir, fname)
            shutil.move(src, dst)
        else:
            img = cv2.imread(src)
            if img is not None:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
                dst = os.path.join(black_dir, fname)
                cv2.imwrite(dst, rotated)
            os.remove(src)

    for sub_name in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, sub_name)
        if os.path.isdir(sub_path) and not os.listdir(sub_path):
            os.rmdir(sub_path)