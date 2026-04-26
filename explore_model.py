import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import numpy as np
from PIL import Image
from models.deeplabv3plus import get_model

MODEL_PATH = 'checkpoints/best_model.pth'
IMG_SIZE = 512
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

img_path = 'data/testImages/Color_Images/0000060.png'
image = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
image_rgb = np.array(image)

input_tensor = image_rgb.astype(np.float32) / 255.0
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)

print(f"Mean Confidence: {conf.mean().item():.4f}")
unique, counts = np.unique(pred.cpu().numpy(), return_counts=True)
print(f"Unique Predicted Classes: {unique}")
print(f"Counts: {counts}")

probs_mean = probs.mean(dim=(0,2,3)).cpu().numpy()
for i, p in enumerate(probs_mean):
    print(f"Class {i}: {p:.4f}")
