import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, num_classes=10):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W)
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        tp = torch.sum(inputs * targets_one_hot, dims)
        fp = torch.sum(inputs * (1 - targets_one_hot), dims)
        fn = torch.sum((1 - inputs) * targets_one_hot, dims)
        
        tversky_index = (tp + 1e-6) / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        return 1 - tversky_index.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class OHEMLoss(nn.Module):
    def __init__(self, top_k=0.7):
        super(OHEMLoss, self).__init__()
        self.top_k = top_k

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Flatten to (N,)
        ce_loss = ce_loss.view(-1)
        
        # Sort and take top k
        num_pixels = ce_loss.numel()
        num_top_k = int(self.top_k * num_pixels)
        
        top_k_losses, _ = torch.topk(ce_loss, num_top_k)
        return top_k_losses.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=10, weight=None):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        # alpha=0.9 focuses heavily on reducing False Negatives (missing rare classes)
        self.tversky = TverskyLoss(alpha=0.9, beta=0.1, num_classes=num_classes)
        self.focal = FocalLoss()
        # top_k=0.1 focuses on the top 10% hardest pixels - extremely aggressive
        self.ohem = OHEMLoss(top_k=0.1)

    def forward(self, inputs, targets):
        # Increased focus on Hard Example Mining (OHEM)
        return self.ce(inputs, targets) + \
               self.tversky(inputs, targets) + \
               self.focal(inputs, targets) + \
               self.ohem(inputs, targets)
