# DINOv3 花種分割與 YOLO11n-cls 花況辨識

本專案為花卉觀光資訊系統的電腦視覺模組，辨識櫻花、金針花與繡球花。先以 DINOv3＋MLP 進行語義分割，再將保留花卉前景的 RGB 影像交給 YOLO11n-cls，判斷花種與 `green`、`half`、`full` 三種花況，共九個聯合類別。

## 功能與進度

- 已完成資料處理、模型訓練、評估及單張影像的分割與分類串接。
- 推論輸出包含遮罩、疊圖、花種分割影像、分類前處理影像、JSON／CSV 結果與異常警示。
- 提供 U-Net＋ResNet34 與 DINOv3＋線性分割頭作為對照模型。
- 即時 API、TravelRAG、文案生成與正式部署仍待完成。

## 快速開始

使用 Python 3.13 與 uv；訓練建議使用 NVIDIA GPU。首次載入 DINOv3 預訓練模型需具備 Hugging Face 存取權限與網路連線。

在專案根目錄執行：

```powershell
uv sync --locked --all-groups
Copy-Item configs/default.toml configs/local.toml
```

自行準備資料集與模型權重，在 `configs/local.toml` 設定分割與分類的資料路徑、`checkpoint_path`，以及 `pipeline.output_root`、`pipeline.min_area_ratio`。相對路徑以專案根目錄為基準；目前介面與參數請以 `configs/default.toml`、`Recognition/README.md` 與各比較模型的 README 為準。

單張影像推論：

```powershell
uv run --no-sync --python 3.13 python -m pipeline_predict "path/to/image.jpg" --config configs/local.toml --device auto
```

程式品質檢查：

```powershell
uv run --no-sync --python 3.13 ruff check .
uv run --no-sync --python 3.13 pytest
```

## 目前結果

- 分割：37 張原始測試影像，mIoU 71.27%、平均 Dice 82.69%。
- 分類：105 張人工遮罩測試影像，九類 Accuracy 91.43%、Macro-F1 76.69%。
- 串接：共同 104 組配對樣本中，花種與花況皆正確的比例，人工遮罩輸入為 91.35%，DINO 分割輸入為 85.58%。

上述為固定資料切分的單次實驗；資料規模與類別分布仍有限，不代表跨場域部署成效。分類信心分數不等於實際開花比例。

## 資料與評估原則

保留原始資料與標註，衍生資料另存。訓練、驗證、測試與推論共用分類前處理流程；資料應依來源群組切分並檢查重複影像，避免資料洩漏。測試集僅供最終評估，不用於選擇超參數。資料集、模型權重與實驗輸出不隨專案發布。

## 公開版本範圍

此版本包含核心程式、設定範例、測試與 GitHub Actions，不包含原始資料集、模型權重、逐張預測結果、訓練輸出、個人文件及舊 Git 歷史。建立 repository 前請先完成 `PUBLICATION_CHECKLIST.md`；本專案目前尚未指定開源授權。
