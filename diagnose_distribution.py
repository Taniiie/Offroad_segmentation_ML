import os
import cv2
import numpy as np
from tqdm import tqdm

MASK_DIR = "data/train/Segmentation"
CLASS_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def diagnose_distribution():
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    class_presence = np.zeros(10)
    
    print(f"Analyzing class presence across {len(mask_files)} images...")
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None: continue
        
        uniques = np.unique(mask)
        mapped_uniques = [CLASS_MAP.get(val, 8) for val in uniques]
        for c in mapped_uniques:
            class_presence[c] += 1
            
    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    
    print("\n--- Class Presence (Number of images containing the class) ---")
    for i, name in enumerate(class_names):
        print(f"{name}: {int(class_presence[i])} images ({class_presence[i]/len(mask_files):.2%})")

if __name__ == "__main__":
    diagnose_distribution()
