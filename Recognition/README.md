# YOLO11n-cls 花種與花況分類

此模組接收 DINOv3 分割後的黑底 RGB 花種影像，預測櫻花、金針花、繡球花與 `green`、`half`、`full` 組成的九個聯合類別，例如 `cherry_half` 與 `daylily_full`。

## 共用前處理

`preprocessing.py` 供訓練、驗證與推論共同使用：

1. 將透明影像合成黑底，取得所有非黑色前景的聯集外框。
2. 外框四周保留 5% 邊界，等比例縮放並補黑邊至 224×224。
3. RGB 值除以 255；只有訓練資料以 50% 機率水平翻轉。

流程不使用隨機裁切、遮除或顏色擾動。全黑影像會回報錯誤，不會直接判為 `green`。

## 資料格式

已切分的分類資料可使用以下結構：

```text
Recognition/datasets/
├─ train/<class_name>/...
├─ val/<class_name>/...
└─ test/<class_name>/...
```

九個類別為三個花種與三種花況的笛卡兒組合。當資料準備程式從 `trainset/testset` 建立切分時，會以 SHA-256 將完全相同的影像綁在同一集合；`Recognition.evaluate` 也會稽核已切分資料是否存在跨集合的完全重複。近似照片與同場景連拍仍需依來源資訊檢查。

## 訓練與推論

設定資料、基礎模型與輸出路徑後執行：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.YOLOv11 --config configs/local.toml
```

只執行資料檢查與準備：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.YOLOv11 --config configs/local.toml --prepare-only
```

單獨測試分類模型：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.predict image.png --model path/to/best.pt
```

## 評估

測試集需與訓練、驗證集分離，並明確指定模型及輸出目錄：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.evaluate `
  --model path/to/best.pt `
  --data-dir Recognition/datasets `
  --split test `
  --output-dir output/classification_evaluation `
  --device auto
```

評估程式輸出九類、花種及花況的 Accuracy、Macro-F1、各類別 Precision／Recall／F1、混淆矩陣、資料涵蓋率與逐張預測紀錄。花況預測會先將九類機率依 `green`、`half`、`full` 跨花種加總，再選擇總機率最高的階段，因此不等同於直接取九類 Top-1 標籤的階段後綴。模型分數未經信心校準，不代表實際開花比例。
