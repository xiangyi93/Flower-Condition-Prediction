# DINOv3 + 線性分割頭

此目錄提供標準 linear probe：凍結 `facebook/dinov3-vits16plus-pretrain-lvd1689m` backbone，只訓練單一 `1×1 Conv2d` 將 patch embedding 映射成四類 logits，再以 bilinear interpolation 還原至輸入大小。

在 repository 根目錄執行（Python 3.13）：

```powershell
uv run --no-sync --python 3.13 python -m dinov3_linear_head.train --output-dir dinov3_linear_head/output/new_run
uv run --no-sync --python 3.13 python -m dinov3_linear_head.evaluate --checkpoint dinov3_linear_head/output/new_run/best_model.pth --output-dir dinov3_linear_head/output/new_evaluation
```

訓練資料切分、前處理、損失及評估方式與 U-Net 對照一致。模型與報表分別輸出至 `output/training` 和 `output/evaluation`；原始資料集只讀。

目前 `output/training` 已保存 30 epochs 的完成實驗；再訓練請指定新的 `--output-dir`。預設 265 train／67 validation、batch 8、AdamW lr 1e-3，僅更新 1,540 個線性頭參數，backbone 保持 `eval()`。沒有多尺度融合、額外 decoder 或後處理。

評估使用全測試集混淆矩陣，輸出與原研究相同的 mIoU、foreground mIoU、Pixel Accuracy、Dice／Macro-F1、每類 IoU／Dice／Precision／Recall 及像素 AP/mAP。`pixel_scores.npz` 保存完整 softmax 分數，可重算 AP。評估要求現有 mask，缺少時報錯，不會在原始資料集生成新檔案。

環境使用 torch 2.11.0+cu128、torchvision 0.26.0+cu128，執行時保留 `--no-sync`。現有 seed 固定了切分和 batch 順序，但當時初始化與增強 RNG 未完整固定；程式已修正未來訓練的 seed 時機。此模型凍結 backbone，而原 DINOv3＋MLP 完整微調，兩者差值不能全部歸因於分割頭。資料洩漏限制與比較表見 [比較報告](../unet_resnet34/output/comparison_20261004.md)。
