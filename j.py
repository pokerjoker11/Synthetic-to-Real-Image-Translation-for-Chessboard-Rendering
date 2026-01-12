import os
import shutil

root = "game10/images"

for item in os.listdir(root):
    path = os.path.join(root, item)
    # If it's a file (not a folder), delete it
    if os.path.isfile(path):
        os.remove(path)
        print(f"Deleted: {path}")
