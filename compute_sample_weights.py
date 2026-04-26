import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

MASK_DIR = "data/train/Segmentation"
CLASS_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

# Rare classes that we want to emphasize
RARE_CLASSES = [0, 1, 3, 4, 5, 6, 7] 

def compute_sample_weights():
    print(f"Calculating DENSITY-BASED sample weights in {MASK_DIR}...")
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    weights = []
    
    # We want to find images where rare classes are PROMINENT, not just present.
    # We'll calculate the % of pixels for each rare class.
    
    all_densities = []
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            all_densities.append(0.0)
            continue

        h, w = mask.shape
        total_pixels = h * w
        
        # Count pixels for all rare classes combined
        rare_pixels = 0
        for val, mapped_id in CLASS_MAP.items():
            if mapped_id in RARE_CLASSES:
                rare_pixels += (mask == val).sum()
        
        density = rare_pixels / total_pixels
        all_densities.append(density)

    # Convert to numpy for easier math
    all_densities = np.array(all_densities)
    
    # Calculate weights: Use an exponential boost for images with high rare-class density.
    # This will heavily favor images that are NOT just mostly grass/landscape.
    # We use (density + epsilon) and normalize or use a power scale.
    
    # Normalize densities to a reasonable range for sampling weights
    # If density is 0.05 (5%), we want it much higher than 0.001.
    epsilon = 0.001
    weights = np.power(all_densities + epsilon, 0.5) # Square root helps prevent extreme outliers while still boosting
    
    # Scale weights so that the min weight is 1.0 and rare-rich images are higher
    weights = weights / weights.min()
    
    # Cap weights to avoid extreme sampling bias (e.g., max 20x)
    weights = np.clip(weights, 1.0, 20.0)

    # Save weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(torch.tensor(weights, dtype=torch.float32), "checkpoints/sample_weights.pt")
    
    print(f"\n--- Sample Weight Stats ---")
    print(f"Total Images: {len(weights)}")
    print(f"Min Weight: {weights.min():.2f}")
    print(f"Max Weight: {weights.max():.2f}")
    print(f"Mean Weight: {weights.mean():.2f}")
    print(f"Number of images with weight > 5: {np.sum(weights > 5)}")
    print(f"Sample weights saved to checkpoints/sample_weights.pt")

if __name__ == "__main__":
    compute_sample_weights()
