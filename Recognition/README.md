# Recognition 資料準備

在專案根目錄使用 Python 3.13 執行：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.YOLOv11 --prepare-only
uv run --no-sync --python 3.13 python -m Recognition.YOLOv11
```

第一行只準備資料；第二行準備／檢查資料後才開始訓練。
設定中的 `classification.train_data` 現在指向已人工清理的 `Recognition/datasets_prepared`。
直接使用現有 train/val/test，不再从原始 datasets 複製回已刪除的圖片。
若要重新從原始資料準備另一份切分，明確指定：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.YOLOv11 --data-dir Recognition/datasets --prepared-dir Recognition/datasets_prepared_v2 --prepare-only
```

- 原始 `trainset/<類別>` 按類別切為約 80% train、20% val，固定 seed=42。
- 原始 `testset/<類別>` 完整複製到 test，不參與切分或訓練中的驗證。
- 預設輸出為同層的 `Recognition/datasets_prepared/{train,val,test}/<類別>`。
- 複製原始影像位元組，不縮放、不更改背景、不修改原始資料。
- 完全相同檔案依 SHA-256 分組，不會分散到 train/val；跨原始訓練／測試集合的重複檔案會報錯。
- 訓練集內跨類別的重複圖片會警告並固定留在 train，清單記錄在 manifest 的 `conflicting_train_hashes`，可用 `images` 查出對應檔名。這只是避免它們流入驗證集，尚未修正標籤問題。測試集的衝突標籤會報錯。
- `split_manifest.json` 記錄来源、目的地、雜湊與切分設定。再次執行會驗證來源及輸出，避免悄悄使用過期資料。
- 若來源、切分設定或輸出改變，請指定新的 `--prepared-dir`，既有輸出不會被覆寫。
- 可設定 `--val-fraction 0.2 --split-seed 42 --prepared-dir Recognition/datasets_prepared_v2`。
- 也接受已整理好的 `train/val/test` 目錄（test 可省略），但必須有獨立 val，且各集合類別一致、非空。

此切分尚無植株／原始照片／連拍群組資訊，只能防止完全相同檔案造成的洩漏，無法排除不同裁切或近似照片。同來源群組資訊齊全後，應改採來源群組切分。

## 植物區域前處理

`preprocessing.py` 共用於訓練、驗證與推論：

1. 讀取 RGB，透明 PNG 則先合成黑底。
2. 用非純黑像素找出所有前景的聯集外框，保留多株植物與不連續區塊，不只取最大區塊。
3. 外框四周各保留 5% 邊界；邊界超過原圖時補黑，避免靠邊植物受到不同處理。
4. 等比例縮放並補黑邊到正方形，預設 224×224，RGB 像素除以 255。
5. 僅訓練時以 50% 機率水平翻轉。停用隨機裁切、隨機遮除與顏色擾動。

這是依黑底裁切空白，不是重新分割；遮罩裡殘留的天空會保留。
分散區塊之間的空間也保留，避免任意拼接改變花況。
全黑圖會報出檔名並停止，不會默默把它當成 green。小區域的警示／拒絕判斷留待正式串接時處理。
對 JPEG 壓縮產生的近黑雜訊或非黑底圖片，應先轉成一致的遮罩黑底資料，或改用原始遮罩定位前景。

```powershell
uv run --no-sync --python 3.13 python -m Recognition.YOLOv11
# 比較較高解析度時，可用 --image-size 320；先固定資料切分再比較。
uv run --no-sync --python 3.13 python -m Recognition.predict image.png --model runs/classify/實際訓練目錄/weights/best.pt
```

訓練結束會印出實際儲存的 best.pt（包含自動遞增的目錄名稱）。
新前處理需要重新訓練，勿將舊的中央裁切模型視為等效模型。
獨立驗證也必須使用 `BloomYOLO`，避免原生驗證器退回中央裁切：

```python
from Recognition.classifier import BloomYOLO

model = BloomYOLO("runs/classify/實際訓練目錄/weights/best.pt")
metrics = model.val(data="Recognition/datasets_prepared", split="val")
```

模型 PT 檔會包含驗證用轉換；部署時需保留 Recognition 模組。改解析度或導出模型後也應使用同一前處理。
`pipeline_predict.py` 已接上新模型、前處理與品質篩選，使用方式如下。

## DINOv3 → YOLO 單張圖片串接

在專案根目錄執行：

```powershell
uv run --no-sync --python 3.13 python -m pipeline_predict "你的圖片.jpg" --device cuda
```

預設權重由 `configs/default.toml` 的 segmentation/classification.checkpoint_path 讀取。
本次 YOLO 預設為 `runs/classify/flower_bloom_model-3-3/weights/best.pt`。
可用 `--seg-model`、`--cls-model`、`--output-dir` 覆蓋。

- `segments/` 保留 background、cherry、daylily、hydrangeas 四張原尺寸黑底圖片，background 不送分類。
- DINO 模型的索引順序保持 0=background、1=cherry、2=daylily、3=hydrangeas，不能按顯示順序任意更換。
- 裁切前檢查非黑色前景：全黑或面積比例低於 `--min-area-ratio` 就跳過。預設 0.001，即原圖的 0.1%，是可調參數，尚非驗證過的最佳門檻。
- 通過檢查才套用 checkpoint 儲存的前處理（全部前景外框、5% 邊界、等比例缩放、補黑邊），`preprocessed/` 保留實際送入 YOLO 的圖片。
- 終端、`results.json`、`summary.csv` 都輸出三種植物的結果。跳過的類別仍保留紀錄，stage/confidence 為 null（CSV 空白），並列出全黑或比例不足原因。
- `area_ratio` 為 DINO 類別遮罩比例；`foreground_ratio` 為該分割圖非黑色像素占原圖的比例，後者用於篩選。
- 花況採用與評估相同的三階段邊際機率：加總各植物同階段機率，再選最高階段。不是強迫 YOLO 預測為 DINO 指定種類。
- 若 YOLO 九類 Top-1 的植物種類與 DINO 不同，另列 warning 供人工確認。
- 信心為模型分數，不是開花百分比，也不是已校準的正確率。小碎片可能通過面積門檻但仍缺少花況資訊。

例如將面積門檻改為 1%：

```powershell
uv run --no-sync --python 3.13 python -m pipeline_predict "你的圖片.jpg" --device cuda --min-area-ratio 0.01
```

## CSV 評估

本次訓練輸出的模型為 `runs/classify/flower_bloom_model-3-3/weights/best.pt`。
評估時明確指定模型，避免使用設定檔中較舊的 checkpoint。預設評估 test，絕不自動改用 val：

```powershell
uv run --no-sync --python 3.13 python -m Recognition.evaluate --model runs/classify/flower_bloom_model-3-3/weights/best.pt --split test --device cuda
```

日常調參使用 `--split val`；test 用於固定模型的最終評估，避免用測試集選超參數。
預設 CSV 寫入 `Recognition/evaluation/<訓練名稱>_<split>_<時間>/`，可指定 `--output-dir`，但不覆寫已有報告。
可用 `--data-dir` 指定同樣含 train/val/test 的資料根目錄。

| CSV | 用途 |
|---|---|
| summary.csv | 九類、植物種類、盛開階段，以及各植物盛開階段的整體指標 |
| per_class_metrics.csv | 各類別樣本數、precision、recall、F1 |
| predictions.csv | 每張圖片的真實標籤、預測、機率、正確與否、讀圖錯誤 |
| errors.csv | 九類或盛開階段判錯，以及無法評估的圖片 |
| confusion_class.csv | 九類混淆矩陣：列是真實類別，欄是預測類別 |
| confusion_stage.csv | green/half/full 混淆矩陣 |
| confusion_species.csv | 植物種類混淆矩陣 |
| metadata.csv | 模型路徑與 SHA256、執行版本、前處理和統計定義 |
| dataset_manifest.csv | 本次各集合圖片的檔案雜湊快照 |
| leakage_check.csv | 跨集合完全相同檔案；無重複時只有表頭 |

全部使用 UTF-8 BOM，方便 Excel 開啟；分數以 0～1 儲存，可套用百分比格式。
stage 的判斷是將同一階段的三種植物機率相加，species 則將同種植物的各階段機率相加。
因此這兩項預測不一定等於九類 Top-1 名稱的前綴／後綴。

建議先看 stage 的 macro-F1、各階段 recall，再看 full 的 precision（避免把半開說成盛開）。
同時保留九類指標，以查出特定植物的弱點。macro 指標等權重納入列出的所有類別；沒有樣本的類別 F1 為 0，應搭配 support 閱讀。
accuracy 僅計算可評估圖片；coverage 與 accuracy_all_inputs 則保留無效圖片的影響。
全黑／損壞圖片會明確記錄，不會當作 green。尚未加入小區域拒絕判斷與信心校準。
檔案雜湊檢查不能排除同場景連拍、不同裁切或重新編碼的重複來源。
