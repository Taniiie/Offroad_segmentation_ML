import os
import cv2
import numpy as np
from tqdm import tqdm

MASK_DIR = "data/train/Segmentation"
CLASS_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
RARE_CLASSES = [0, 1, 3, 4, 5, 6, 7] 

def diagnose_density():
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    all_densities = []
    
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None: continue
        
        rare_pixels = 0
        for val, mapped_id in CLASS_MAP.items():
            if mapped_id in RARE_CLASSES:
                rare_pixels += (mask == val).sum()
        
        all_densities.append(rare_pixels / mask.size)
            
    all_densities = np.array(all_densities)
    print(f"\n--- Rare Pixel Density Stats ---")
    print(f"Mean: {all_densities.mean():.4f}")
    print(f"Median: {np.median(all_densities):.4f}")
    print(f"Max: {all_densities.max():.4f}")
    print(f"Min: {all_densities.min():.4f}")
    print(f"90th Percentile: {np.percentile(all_densities, 90):.4f}")
    print(f"Number of images with < 1% rare pixels: {np.sum(all_densities < 0.01)}")
    print(f"Number of images with > 10% rare pixels: {np.sum(all_densities > 0.10)}")

if __name__ == "__main__":
    diagnose_density()
