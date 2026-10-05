# ==============================================================================
# DINOv3 Semantic Segmentation Training Script
# Backbone: facebook/dinov3-vitb16 (or tiny)
# Head: All-MLP Decoder (SegFormer Style)
# ==============================================================================
# 若MLP Decoder 放大 16 倍的邊緣太模糊（馬賽克感），把 decode_head 換成 DPT 邏輯。
# DPT 拿 多層特徵 進行融合。

import csv
import os
import random
from datetime import date

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel

from Segmentation.data import collect_image_mask_pairs
from Segmentation.metrics import DiceLoss

NUM_CLASSES = 4
MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
CLASS_MAP = {"background": 0, "cherry": 1, "daylily": 2, "hydrangeas": 3}


def set_seed(seed: int) -> None:
    """Set all random sources used by the segmentation training workflow."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def segmentation_confusion_matrix(
    predictions: np.ndarray, targets: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Return a pixel-level confusion matrix for one or more segmentation batches."""
    flattened_targets = targets.reshape(-1)
    flattened_predictions = predictions.reshape(-1)
    indices = flattened_targets * num_classes + flattened_predictions
    return np.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def mean_iou_from_confusion_matrix(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    union = true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    valid_iou = iou[~np.isnan(iou)]
    return float(valid_iou.mean()) if valid_iou.size else float("nan")


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_miou: float,
    seed: int,
    data_loader_rng_state: torch.Tensor,
    training_config: dict,
) -> dict:
    return {
        "format_version": 1,
        "epoch": epoch,
        "best_val_miou": best_val_miou,
        "seed": seed,
        "class_map": CLASS_MAP,
        "training_config": training_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "data_loader_rng_state": data_loader_rng_state,
    }


def resume_training(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Load a complete training checkpoint or a legacy model-only state dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        model.load_state_dict(checkpoint)
        return {
            "start_epoch": 0,
            "best_val_miou": float("-inf"),
            "data_loader_rng_state": None,
        }

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    restore_rng_state(checkpoint["rng_state"])
    return {
        "start_epoch": int(checkpoint["epoch"]),
        "best_val_miou": float(checkpoint["best_val_miou"]),
        "data_loader_rng_state": checkpoint["data_loader_rng_state"],
    }


# --- 1. 定義模型架構 ---
class DINOv3SemanticSeg(nn.Module):
    def __init__(
        self, model_name=MODEL_NAME, num_classes=NUM_CLASSES
    ):
        super().__init__()
        # 加載 DINOv3
        self.backbone = AutoModel.from_pretrained(
            model_name, token=os.environ.get("HF_TOKEN")
        )
        hidden_size = self.backbone.config.hidden_size  # 例如 Vit-S 是 384

        # 簡單但強大的 MLP 解碼器 (語義分割專用)
        self.decode_head = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=16, mode="bilinear", align_corners=False
            ),  # 將 patch 特徵放大回原圖
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        # 提取 Patch Tokens (排除 CLS token 和 Registers)
        # DINOv3 有 4 個 register tokens, 1 個 CLS token，所以從 index 5 開始
        last_hidden_state = outputs.last_hidden_state[:, 5:, :]

            # 將序列轉換回 2D 特徵圖 (假設輸入 224x224, patch 16, 則特徵圖為 14x14)
        b, n, c = last_hidden_state.shape
        h = w = int(np.sqrt(n))
        feat = last_hidden_state.transpose(1, 2).reshape(b, c, h, w)

        logits = self.decode_head(feat)
        return logits


# --- 2. 語義分割數據集類別 ---
class SemanticDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.samples = collect_image_mask_pairs(img_dir, mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(
            Image.open(mask_path).convert("L")
        )  # 讀取為單通道灰階圖
        mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask


def validate(model, val_loader, ce_criterion, dice_criterion, device):
    model.eval()
    total_loss = 0
    global_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss_ce = ce_criterion(outputs, masks)
            loss_dice = dice_criterion(outputs, masks)
            loss = loss_ce + loss_dice

            preds = torch.argmax(outputs, dim=1)
            global_confusion += segmentation_confusion_matrix(
                preds.cpu().numpy(), masks.cpu().numpy()
            )

            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader), mean_iou_from_confusion_matrix(global_confusion)


def save_training_csv(log_file, epoch, train_loss, train_miou, val_loss, val_miou):
    file_exists = os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["epoch", "train_loss", "train_miou", "val_loss", "val_miou"]
            )
        writer.writerow(
            [
                epoch,
                f"{train_loss:.4f}",
                f"{train_miou:.4f}",
                f"{val_loss:.4f}",
                f"{val_miou:.4f}",
            ]
        )


# --- 3. 訓練主程式 ---
def train_semantic(
    input_dir: str,
    output_dir: str,
    resume_from=None,
    val_split=0.2,
    epochs=30,
    seed: int = 42,
):
    # 參數設定
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4 # 若有old_model可以折半成 5e-5
    EPOCHS = epochs
    training_config = {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "val_split": val_split,
        "epochs": EPOCHS,
    }

    set_seed(seed)

    # 初始化模型與優化器
    model = DINOv3SemanticSeg(num_classes=NUM_CLASSES).to(DEVICE)

    weights = torch.tensor([1.0, 1.0, 10.0, 10.0]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ce_criterion = nn.CrossEntropyLoss(weight=weights)
    dice_criterion = DiceLoss().to(DEVICE)

    # 數據增強
    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 建立完整數據集
    train_full_dataset = SemanticDataset(
        img_dir=input_dir, mask_dir=f"{input_dir}/masks", transform=train_transform
    )
    val_full_dataset = SemanticDataset(
        img_dir=input_dir, mask_dir=f"{input_dir}/masks", transform=eval_transform
    )

    # 拆分 train/val
    dataset_size = len(train_full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=seed
    )

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📊 數據拆分: {len(train_dataset)} train / {len(val_dataset)} val")

    # CSV 日誌
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log.csv")

    # 追蹤最佳模型
    start_epoch = 0
    best_val_miou = float("-inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(output_dir, "last_checkpoint.pth")
    best_training_checkpoint_path = os.path.join(output_dir, "best_training_checkpoint.pth")

    if resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from}")
        print(f"♻️ Loading training checkpoint: {resume_from}")
        resume_state = resume_training(resume_from, model, optimizer, DEVICE)
        start_epoch = resume_state["start_epoch"]
        best_val_miou = resume_state["best_val_miou"]
        if resume_state["data_loader_rng_state"] is not None:
            train_generator.set_state(resume_state["data_loader_rng_state"])
        print(f"✅ Resuming from epoch {start_epoch} with best Val mIoU {best_val_miou:.4f}.")

    print("🚀 開始語義分割訓練...")
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        # === Training ===
        model.train()
        total_loss = 0
        train_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss_ce = ce_criterion(outputs, masks)
            loss_dice = dice_criterion(outputs, masks)
            loss = loss_ce + loss_dice

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_confusion += segmentation_confusion_matrix(
                preds.cpu().numpy(), masks.cpu().numpy()
            )

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_miou = mean_iou_from_confusion_matrix(train_confusion)

        # === Validation ===
        val_loss, val_miou = validate(
            model, val_loader, ce_criterion, dice_criterion, DEVICE
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f} | Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}"
        )

        # 寫入 CSV
        save_training_csv(
            log_file, epoch + 1, train_loss, train_miou, val_loss, val_miou
        )

        # 儲存最佳模型
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), best_model_path)
            torch.save(
                training_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    best_val_miou,
                    seed,
                    train_generator.get_state(),
                    training_config,
                ),
                best_training_checkpoint_path,
            )

        torch.save(
            training_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_val_miou,
                seed,
                train_generator.get_state(),
                training_config,
            ),
            last_checkpoint_path,
        )

        # 每 5 個 Epoch 存一次 checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                f"{output_dir}/dinov3_seg_{date.today().strftime('%Y%m%d')}_epoch{epoch + 1}.pth",
            )
    print(f"模型訓練完成，最佳 Val mIoU: {best_val_miou:.4f}")
    # return best_model_path
