import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        num_classes = logits.size(1)
        
        # 將 targets 轉為 One-hot 格式: [B, C, H, W]
        true_masks = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # 對預測結果取 Softmax 得到機率值
        probs = F.softmax(logits, dim=1)
        
        # 計算交集與聯集
        dims = (0, 2, 3) # 在 Batch, Height, Width 維度上加總
        intersection = torch.sum(probs * true_masks, dims)
        cardinality = torch.sum(probs + true_masks, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 回傳 1 - Dice 作為 Loss
        return 1. - dice_score.mean()
    
# --- 計算 mIoU 的函數 ---
def get_miou(preds, targets, num_classes=4):
    # preds: [B, H, W], targets: [B, H, W]
    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum()
        union = ((preds == cls) | (targets == cls)).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)