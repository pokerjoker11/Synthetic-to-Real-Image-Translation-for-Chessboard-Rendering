import os
import shutil

root = "game8/images"

for subdir in [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]:
    path = os.path.join(root, subdir)
    for f in os.listdir(path):
        shutil.move(os.path.join(path, f), os.path.join(root, f))
    os.rmdir(path)
