# U-Net + ResNet34

此目錄提供常規的 U-Net 對照模型：ImageNet 預訓練 ResNet34 encoder、四層 skip-connected decoder，以及四類 pixel logits。訓練沿用主模型的 224×224、水平翻轉、ImageNet normalization、加權 Cross Entropy + Dice Loss、seed 42 和 validation mIoU 選模規則。

在 repository 根目錄執行（Python 3.13）：

```powershell
uv run --no-sync --python 3.13 python -m unet_resnet34.train --output-dir unet_resnet34/output/new_run
uv run --no-sync --python 3.13 python -m unet_resnet34.evaluate --checkpoint unet_resnet34/output/new_run/best_model.pth --output-dir unet_resnet34/output/new_evaluation
```

模型與報表分別輸出至 `output/training` 和 `output/evaluation`。原始資料集只讀，不會被改寫。

目前 `output/training` 已保存 30 epochs 的完成實驗；再訓練請指定新的 `--output-dir`，程式會拒絕覆蓋已有訓練紀錄。預設為 265 train／67 validation、seed 42、batch 8、AdamW lr 1e-4。全模型微調；decoder 採 bilinear 上採樣、skip concatenation、3×3 Conv＋BN＋ReLU。

評估預設為 37 張 `testdata0929`，輸出 mIoU、foreground mIoU、Pixel Accuracy、Dice／Macro-F1、每類 IoU／Dice／Precision／Recall 和像素 AP/mAP。AP 使用完整 softmax 分數，與 `research_analysis` 定義一致；`pixel_scores.npz` 可供重算，CSV 和 JSON 保存完整數值。

環境實際使用 torch 2.11.0+cu128、torchvision 0.26.0+cu128。使用 `--no-sync` 可保留已驗證的 GPU 套件；直接 `uv sync` 會依目前 lockfile 裝回 CPU 版。現有紀錄為單次實驗，seed 固定了資料切分及 batch 順序，但當時初始化與增強 RNG 未完整固定；程式已修正未來訓練的 seed 時機。本次結果與資料洩漏限制見 [比較報告](output/comparison_20261004.md)。
