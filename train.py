import torch
import os
import glob
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.deeplabv3plus import get_model
from losses.losses import CombinedLoss
from dataset.dataset import SegDataset
from tqdm import tqdm
import numpy as np
from metrics import dice_score, iou_score
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

# GPU optimizations
torch.backends.cudnn.benchmark = True

def train():
    # Parameters for High-Speed 70% IoU Push
    NUM_CLASSES = 10  
    IMG_SIZE = 512       
    EPOCHS = 100         
    BATCH_SIZE = 8       # Increased for more stable gradients
    LR_DECODER = 5e-4    
    LR_ENCODER = 1e-5    
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

    model = get_model(NUM_CLASSES).to(DEVICE)
    checkpoint_path = "checkpoints/best_model_optimized.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
        {'params': [p for n, p in model.named_parameters() if "encoder" not in n], 'lr': LR_DECODER}
    ], weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda')

    # Scheduler and Early Stopping (Increased for 70% Push)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-7)
    early_stopping = EarlyStopping(patience=15, path="checkpoints/best_model_optimized.pth")

    # Combined Loss (Updated with AGGRESSIVE down-weighting of majority classes)
    # Indices: 2: Dry Grass, 8: Landscape, 9: Sky. We set them to very low values.
    # Rare classes get 50-100x more weight.
    class_weights = torch.tensor([
        100.0, 100.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0
    ]).to(DEVICE)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, weight=class_weights)

    # Datasets
    train_ds = SegDataset(train_imgs, train_masks, augment=True, img_size=IMG_SIZE)
    val_ds = SegDataset(val_imgs, val_masks, augment=False, img_size=IMG_SIZE)
    
    # Weighted Sampling for Imbalance
    sample_weights_path = "checkpoints/sample_weights.pt"
    if os.path.exists(sample_weights_path):
        print("Loading sample weights for WeightedRandomSampler...")
        sample_weights = torch.load(sample_weights_path, weights_only=True)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True, prefetch_factor=2)
    else:
        print("Warning: Sample weights not found. Using standard shuffle.")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, prefetch_factor=2)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        train_dice = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            dice, _ = dice_score(outputs, masks, NUM_CLASSES)
            train_dice += dice.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, total_dice, total_iou = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                d_mean, _ = dice_score(outputs, masks, NUM_CLASSES)
                total_dice += d_mean.item()
                total_iou += iou_score(outputs, masks, NUM_CLASSES).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = total_dice / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f}")
        
        scheduler.step(avg_val_dice)
        early_stopping(avg_val_dice, model)

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break

        torch.save(model.state_dict(), f"checkpoints/last_model_optimized.pth")

if __name__ == "__main__":
    train()
