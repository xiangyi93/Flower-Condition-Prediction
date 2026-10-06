# DINOv3 花種分割與 YOLO11n-cls 花況辨識

[![Quality checks](https://github.com/xiangyi93/Flower-Condition-Prediction/actions/workflows/quality.yml/badge.svg)](https://github.com/xiangyi93/Flower-Condition-Prediction/actions/workflows/quality.yml)

本專案辨識臺灣花卉觀光場域中的櫻花、金針花與繡球花，並將每種花卉分為 `green`（未開花）、`half`（半開）與 `full`（盛開）三種花況。系統先用 DINOv3＋MLP 找出花種區域，再由 YOLO11n-cls 判斷花種與花況，共九個聯合類別。

## 影像流程

```mermaid
flowchart LR
    A[原始花卉影像] --> B[DINOv3＋MLP<br/>背景與三種花卉的語義分割]
    B --> C[依遮罩產生三張<br/>黑底 RGB 花種影像]
    C --> D[面積門檻、前景外框、<br/>5% 邊界與 224×224 補邊]
    D --> E[YOLO11n-cls<br/>九類聯合分類]
    E --> F[各花種的花況、機率<br/>與缺漏或種類不一致警示]
```

DINOv3＋MLP 輸出背景、櫻花、金針花及繡球花四類遮罩。流程控制程式把三種花卉分別套回原圖，保留色彩與紋理；YOLO 接收處理後的 RGB 影像，不直接接收遮罩或 DINOv3 特徵向量。

## 目前成果

| 評估範圍 | 測試資料 | 結果 |
|---|---:|---|
| DINOv3＋MLP 語義分割 | 37 張原始影像 | mIoU 71.27%、平均 Dice 82.69% |
| YOLO11n-cls 個別分類 | 105 張人工遮罩影像 | 九類 Accuracy 91.43%、Macro-F1 76.69% |
| YOLO 輸入來源比較 | 104 組同時具備人工遮罩與 DINO 輸入的花況標註區域 | 人工遮罩輸入 Accuracy 為 95／104（91.35%）；DINO 分割輸入的條件式 Accuracy 為 89／104（85.58%），Macro-F1 為 71.45% |
| 完整流程花況辨識 | 105 組有人工花況標註的花種區域 | 104 組通過分割門檻，涵蓋率為 99.05%；將未通過的 1 組列為錯誤後，端到端 Accuracy 為 89／105（84.76%） |

## 專案結構

| 路徑 | 用途 |
|---|---|
| `.github/workflows/` | 在 push 與 pull request 執行 Ruff 與 pytest。 |
| `configs/` | 放置不含資料與權重的 TOML 設定範例。 |
| `Segmentation/` | DINOv3＋MLP 的資料讀取、訓練、遮罩轉換與分割評估。 |
| `Recognition/` | YOLO 資料準備、共用前處理、訓練、推論與分類評估。 |
| `label_region_export/` | 將 Labelme 標註轉成各花種的黑底 RGB 影像，供分類資料整理使用。 |
| `tests/` | 驗證資料切分、前處理、模型輸出、指標與完整流程。 |

### 主要程式

| 程式 | 用途 |
|---|---|
| `Segmentation/data.py`、`getMask.py` | 配對原圖與遮罩，並將 Labelme JSON 轉為四類索引遮罩。 |
| `Segmentation/train.py`、`metrics.py` | 定義 DINOv3＋MLP、訓練函式、Dice loss 與 mIoU 計算。 |
| `Segmentation/evaluate.py` | 輸出 mIoU、Dice、混淆矩陣等分割指標。 |
| `Segmentation/experiment_runner.py` | 提供共用的資料切分、訓練與評估流程。 |
| `Recognition/prepare_data.py` | 建立 train／val／test 結構，並用 SHA-256 檢查完全重複影像。 |
| `Recognition/preprocessing.py`、`classifier.py` | 統一前景裁切、補邊及 YOLO 訓練、驗證與推論介面。 |
| `Recognition/YOLOv11.py` | 準備分類資料並訓練 YOLO11n-cls。 |
| `Recognition/predict.py`、`evaluate.py` | 執行單張分類，並輸出九類、花種與花況指標。 |
| `Recognition/preview_preprocessing.py`、`plot_confusion_matrices.py` | 產生前處理預覽與混淆矩陣圖，供人工檢查。 |
| `label_region_export/export_label_regions.py` | 將 Labelme 標註轉為各花種的黑底 RGB 中間影像。 |
| `pipeline_predict.py` | 執行單張影像的分割、三分支轉換與花況分類。 |
| `pipeline_evaluate.py` | 評估 DINO→YOLO 串接、面積門檻及與人工標註的對應關係。 |
| `project_config.py` | 讀取 TOML，並將相對路徑解析至 repository 根目錄。 |
| `pyproject.toml`、`uv.lock` | 定義 Python 版本、相依套件與品質檢查工具。 |

## 執行方式

環境使用 Python 3.13 與 [uv](https://docs.astral.sh/uv/)。首次載入 DINOv3 預訓練模型時，需要 Hugging Face 存取權限與網路連線。

```powershell
uv sync --locked --all-groups
Copy-Item configs/default.toml configs/local.toml
```

在 `configs/local.toml` 填入自己的資料與 checkpoint 路徑後，可執行單張影像流程：

```powershell
uv run --no-sync --python 3.13 python -m pipeline_predict "path/to/image.jpg" --config configs/local.toml --device auto
```

分類模組的資料格式與訓練、評估方式見 [Recognition 說明](Recognition/README.md)。程式品質檢查如下：

```powershell
uv run --no-sync --python 3.13 ruff check .
uv run --no-sync --python 3.13 pytest
```

## 參考資料與授權

- [DINOv3 論文](https://arxiv.org/abs/2508.10104)與本研究採用的 [DINOv3 ViT-S/16+ 預訓練模型](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
- [Ultralytics YOLO11 官方文件](https://docs.ultralytics.com/models/yolo11/)

本 repository 自行撰寫的程式碼採 [GNU AGPL v3.0](LICENSE.md) 授權。DINOv3 權重仍受 [Meta DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) 約束；使用或再散布第三方程式與模型前，請查閱各上游專案的最新條款。
