import os
import shutil
import glob

def organize_data(root_dir="."):
    """
    Organizes the Duality AI dataset into a clean structure for training.
    Expected raw structure:
    - training_dataset/
        - train/
            - Color_Images/
            - Segmentation/
        - val/
            - Color_Images/
            - Segmentation/
    - testImages/
        - Color_Images/
        - Segmentation/
    """
    
    # Target structure
    target_base = "data"
    mappings = [
        ("training_dataset/train/Color_Images", "data/train/images"),
        ("training_dataset/train/Segmentation", "data/train/masks"),
        ("training_dataset/val/Color_Images", "data/val/images"),
        ("training_dataset/val/Segmentation", "data/val/masks"),
        ("testImages/Color_Images", "data/testImages/images"),
        ("testImages/Segmentation", "data/testImages/masks"),
    ]

    for src_rel, dst_rel in mappings:
        src = os.path.join(root_dir, src_rel)
        dst = os.path.join(root_dir, dst_rel)
        
        if os.path.exists(src):
            print(f"Moving {src} -> {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                print(f"Destination {dst} already exists. Skipping move.")
            else:
                shutil.move(src, dst)
        else:
            print(f"Source not found: {src}")

    # Remove the now empty training_dataset folder if it exists
    training_dataset_path = os.path.join(root_dir, "training_dataset")
    if os.path.exists(training_dataset_path) and not os.listdir(training_dataset_path):
        os.rmdir(training_dataset_path)

    print("\nData organization complete!")
    print("Your 'data/' folder is ready for training.")

if __name__ == "__main__":
    organize_data()
