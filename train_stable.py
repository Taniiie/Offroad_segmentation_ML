"""
Stable Training Script for Offroad Segmentation
This uses a simpler loss function and moderate class weights to avoid training collapse.
"""
import torch
import os
import glob
from torch.utils.data import DataLoader
from models.deeplabv3plus import get_model
from dataset.dataset import SegDataset
from tqdm import tqdm
import numpy as np
from metrics import dice_score, iou_score
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# GPU optimizations
torch.backends.cudnn.benchmark = True

def train():
    # Stable training parameters
    NUM_CLASSES = 10  
    IMG_SIZE = 512       
    EPOCHS = 50         
    BATCH_SIZE = 8
    LR = 1e-4  # Single learning rate for stability
    WEIGHT_DECAY = 1e-4 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths
    train_img_dir = "data/train/Color_Images"
    train_mask_dir = "data/train/Segmentation"
    val_img_dir = "data/val/Color_Images"
    val_mask_dir = "data/val/Segmentation"

    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")))
    val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))

    if not train_imgs:
        print("Error: No training images found.")
        return

    print(f"Training on {len(train_imgs)} images, validating on {len(val_imgs)} images")
    print(f"Device: {DEVICE}")

    # Create model
    model = get_model(NUM_CLASSES).to(DEVICE)
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    # Resume from checkpoint if exists
    start_epoch = 0
    checkpoint_path = "checkpoints/last_model_stable.pth"
    if os.path.exists(checkpoint_path):
        print(f"📂 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"   Resuming from epoch {start_epoch}")
        else:
            # Old format - just model weights
            model.load_state_dict(checkpoint)
            print("   Loaded model weights (epoch unknown, starting from 0)")
    else:
        print("🆕 Starting fresh training...")

    # Scheduler and Early Stopping
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7)
    early_stopping = EarlyStopping(patience=10, path="checkpoints/best_model_stable.pth")

    # MODERATE class weights based on inverse frequency (not extreme 100x multipliers)
    # Approximate frequencies: Trees 0%, Bushes 0.16%, Grass 35%, DryBushes 5%, Clutter 3.8%
    #                          Flowers 0%, Logs 0%, Rocks 1.9%, Landscape 36.7%, Sky 17%
    # Using sqrt of inverse frequency for stability
    class_weights = torch.tensor([
        10.0,  # Trees (very rare)
        8.0,   # Lush Bushes (rare)
        1.0,   # Dry Grass (common)
        3.0,   # Dry Bushes
        3.0,   # Ground Clutter
        10.0,  # Flowers (very rare)
        10.0,  # Logs (very rare)
        4.0,   # Rocks
        1.0,   # Landscape (common)
        1.5,   # Sky (common)
    ]).to(DEVICE)

    # Datasets
    train_ds = SegDataset(train_imgs, train_masks, augment=True, img_size=IMG_SIZE)
    val_ds = SegDataset(val_imgs, val_masks, augment=False, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)

    print("Starting stable training...")
    best_iou = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        train_dice = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                # Simple CrossEntropy with moderate weights - no complex combination
                loss = F.cross_entropy(outputs, masks, weight=class_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            dice, _ = dice_score(outputs, masks, NUM_CLASSES)
            train_dice += dice.item()
            
            # Show prediction diversity in progress bar
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                n_classes = len(torch.unique(pred))
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}", "classes": n_classes})

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, total_dice, total_iou = 0, 0, 0
        val_classes_seen = set()
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = F.cross_entropy(outputs, masks, weight=class_weights)
                
                val_loss += loss.item()
                d_mean, _ = dice_score(outputs, masks, NUM_CLASSES)
                total_dice += d_mean.item()
                total_iou += iou_score(outputs, masks, NUM_CLASSES).item()
                
                # Track unique predicted classes
                pred = torch.argmax(outputs, dim=1)
                val_classes_seen.update(torch.unique(pred).cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = total_dice / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f} | Classes predicted: {sorted(val_classes_seen)}")
        
        # Alert if model is collapsing
        if len(val_classes_seen) <= 2:
            print(f"⚠️ WARNING: Model only predicting {len(val_classes_seen)} classes - possible collapse!")
        
        scheduler.step(avg_val_dice)
        early_stopping(avg_val_dice, model)

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break

        # Save checkpoint with epoch and optimizer for resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': avg_val_iou,
        }, "checkpoints/last_model_stable.pth")
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            print(f"✅ New best IoU: {best_iou:.4f}")

    print(f"\n🎉 Training complete! Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    train()
