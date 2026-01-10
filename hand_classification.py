import cv2
import mediapipe as mp
import os
import shutil
import numpy as np

def filter_hybrid(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.0001)
    
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg'))])
    
    base_img = cv2.imread(os.path.join(src_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    shutil.copy(os.path.join(src_dir, files[0]), os.path.join(dst_dir, files[0]))

    for i in range(1, len(files)):
        img_path = os.path.join(src_dir, files[i])
        image = cv2.imread(img_path)
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        
        if not res.multi_hand_landmarks:
            shutil.copy(img_path, os.path.join(dst_dir, files[i]))
            print(f"{files[i]}: Clean")
        else:
            print(f"{files[i]}: Filtered (AI Hand or Motion)")

    detector.close()

def main():
    src_dir = r"game8\images"
    dst_dir = r"game8\clean_training_data"
    
    if os.path.exists(src_dir):
        filter_hybrid(src_dir, dst_dir)
    else:
        print(f"Directory {src_dir} not found.")

if __name__ == "__main__":
    main()