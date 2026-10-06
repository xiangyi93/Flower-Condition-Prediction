# 依 Labelme 標註輸出四類影像

將圖片依照label json檔案切出圖片，用來YOLOv11訓練
程式、說明與測試集中在 `label_region_export/`：

```text
label_region_export/
├─ export_label_regions.py
├─ README.md
└─ test_export_label_regions.py
```

在專案根目錄執行（Python 3.13）：

```powershell
uv run --no-sync --python 3.13 python label_region_export/export_label_regions.py --dry-run
uv run --no-sync --python 3.13 python label_region_export/export_label_regions.py
```

預設讀取 `configs/default.toml` 的 `segmentation.train_data` 與
`segmentation.test_data`。也可以指定來源：

```powershell
uv run --no-sync --python 3.13 python label_region_export/export_label_regions.py --train-dir Segmentation/datasets/train --test-dir Segmentation/datasets/test --output-dir Recognition/label_regions
```

只測試此工具：

```powershell
uv run --no-sync --python 3.13 python -m pytest -p no:cacheprovider label_region_export
```

專案預設的 pytest 搜尋路徑也包含此資料夾。

輸出結構：

```text
Recognition/label_regions/
├─ train/
│  ├─ background/000001.png ...
│  ├─ cherry/000001.png ...
│  ├─ hydrangeas/000001.png ...
│  └─ daylily/000001.png ...
├─ test/
│  ├─ background/000001.png ...
│  ├─ cherry/000001.png ...
│  ├─ hydrangeas/000001.png ...
│  └─ daylily/000001.png ...
└─ manifest.csv
```

- 依 JSON 相對路徑的自然順序編號，例如 image2 先於 image10；train/test 各自從 000001 開始。同一來源的四張圖片共用編號。
- 輸出原圖大小的 RGB PNG，保留目標類別原始顏色，其餘填黑。未出現的類別也固定輸出全黑圖，不做物件裁切或模型推論。
- 未標註的區域為 background。多邊形填色方式與 `getMask.py` 相同：座標轉整數，重疊處以後面的標註為準。
- JSON 與圖片必須同目錄、同主檔名，且唯一配對；不依賴 JSON 內可能過期的 imagePath。不讀取既有 masks，也不修改原始檔。
- 寫入前檢查所有影像尺寸、標籤與形狀。未知標籤、不支援的形狀、缺少 JSON 或圖片會直接報錯，不默默丟棄資料。支援 polygon 與 linestrip；後者按 Labelme 預設的 10 像素線寬繪製未封閉折線，不自動填滿內部。若原意是標出整片花卉，需修正標註。
- 輸出目錄必須不存在，且不可與輸入資料夾互相包含，以免覆寫或混入原始資料。
- 以解碼後的 RGB 像素雜湊檢查 train/test 完全相同的影像，若發現則停止；這不能排除連拍、重新壓縮或不同裁切造成的近似重複，仍需依拍攝來源檢查。
- train/test 使用相同前處理，沒有隨機增強。manifest.csv 記錄來源、編號、雜湊與各類像素數，方便追溯和挑出空白類別。

這是依花種分組的中間資料，後續仍要人工標註 green/half/full，才可作為花況分類資料集；不存在的花種所產生的全黑圖不應直接加入花況訓練。人工標註遮罩的品質不代表實際分割模型品質，部署評估仍應另測模型產生的輸入。
