import torch
import torch.nn as nn
import torch.optim as optim
from models.deeplabv3plus import get_model
from losses.losses import CombinedLoss
from dataset.dataset import SegDataset
import os
import glob
from tqdm import tqdm

def overfit():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10
    
    # Pick 1 image
    train_img_dir = "data/train/Color_Images"
    train_mask_dir = "data/train/Segmentation"
    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))[:1]
    train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")))[:1]
    
    print(f"Overfitting on: {train_imgs[0]}")
    
    model = get_model(NUM_CLASSES).to(DEVICE)
    # Simple Adam for overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss(num_classes=NUM_CLASSES) # No weights for overfitting
    
    ds = SegDataset(train_imgs, train_masks, augment=False, img_size=512)
    img, mask = ds[0]
    # Create a batch of size 2 to avoid BatchNorm errors
    img = torch.from_numpy(img).unsqueeze(0).repeat(2, 1, 1, 1).to(DEVICE)
    mask = torch.from_numpy(mask).unsqueeze(0).repeat(2, 1, 1).to(DEVICE)
    
    for epoch in range(200): # Increased epochs for overfitting
        model.train()
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            pred = torch.argmax(outputs, dim=1)
            correct = (pred == mask).sum().item()
            total = mask.numel()
            acc = correct / total
            # Check how many unique classes are in prediction
            uniques = torch.unique(pred).cpu().numpy()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Pixel Acc: {acc:.4f} | Predicted Classes: {uniques}")

if __name__ == "__main__":
    overfit()
