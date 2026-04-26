import os
import torch
import numpy as np
from PIL import Image

MASK_DIR = "data/train/Segmentation"
NUM_CLASSES = 10

def debug_weights():
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    filename = mask_files[0]
    mask_path = os.path.join(MASK_DIR, filename)
    mask = np.array(Image.open(mask_path))
    
    print(f"File: {filename}")
    print(f"Unique values: {np.unique(mask)}")
    
    counts = np.bincount(mask.flatten(), minlength=256)
    class_counts = np.zeros(NUM_CLASSES)
    
    class_counts[0] = counts[0]   # Trees
    class_counts[1] = counts[1]   # Lush Bushes
    class_counts[2] = counts[2]   # Dry Grass
    class_counts[3] = counts[3]   # Dry Bushes
    class_counts[4] = counts[4]   # Ground Clutter
    class_counts[5] = counts[5]   # Flowers
    class_counts[6] = counts[6]   # Logs
    class_counts[7] = counts[7]   # Rocks
    class_counts[8] = counts[8]   # Landscape
    class_counts[9] = counts[255] # Sky
    
    print(f"Class counts: {class_counts}")
    total = class_counts.sum()
    print(f"Total: {total}")
    freq = class_counts / total
    print(f"Freq: {freq}")
    
    weights = 1.0 / np.log(1.02 + freq)
    print(f"Weights: {weights}")

if __name__ == "__main__":
    debug_weights()
