import os
import cv2
import numpy as np
from tqdm import tqdm

INFERENCE_DIR = "inference_results_v2"

def diagnose_predictions():
    results = [f for f in os.listdir(INFERENCE_DIR) if f.startswith('result_')]
    if not results:
        print("No results found.")
        return
        
    class_counts = np.zeros(10)
    
    print(f"Analyzing {len(results)} prediction images...")
    for res_name in tqdm(results):
        res_path = os.path.join(INFERENCE_DIR, res_name)
        img = cv2.imread(res_path)
        if img is None: continue
        
        # The result image is [ Original | Mask | Overlay ]
        # Mask is in the middle (768x768)
        img_size = 768
        mask_region = img[:, img_size:2*img_size, :]
        
        # Find which of our C0-C9 colors are present
        # This is a bit slow, but it's only for a few images
        COLORS = np.array([
            [34, 139, 34],    # 0: Trees
            [0, 255, 0],      # 1: Lush Bushes
            [189, 183, 107],  # 2: Dry Grass
            [160, 82, 45],    # 3: Dry Bushes
            [105, 105, 105],  # 4: Ground Clutter
            [255, 0, 255],    # 5: Flowers
            [139, 69, 19],    # 6: Logs
            [128, 128, 128],  # 7: Rocks
            [210, 180, 140],  # 8: Landscape
            [135, 206, 235],  # 9: Sky
        ])
        
        # Flatten and convert to BGR for matching cv2 image
        COLORS_BGR = COLORS[:, ::-1]
        
        for i, color in enumerate(COLORS_BGR):
            # Check if this color exists in the mask
            match = (mask_region == color).all(axis=2)
            if match.any():
                class_counts[i] += 1
                
    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    
    print("\n--- Predictions Class Presence ---")
    for i, name in enumerate(class_names):
        print(f"{name}: present in {int(class_counts[i])} / {len(results)} images")

if __name__ == "__main__":
    diagnose_predictions()
