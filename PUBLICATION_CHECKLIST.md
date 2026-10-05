# Repository 發布前檢查清單

此資料夾以白名單從本機專案建立，未包含原始 `.git` 歷史、資料集、模型權重、訓練／推論輸出、推甄文件、虛擬環境與快取。

## 發布前

- [ ] 選擇並加入適合的 `LICENSE`；目前未授權他人複製、修改或散布。
- [ ] 確認預訓練模型、資料來源及圖片的授權與引用方式。
- [ ] 在乾淨的 Python 3.13 環境執行 `uv sync --locked --all-groups`。
- [ ] 執行 `uv run ruff check .` 與 `uv run pytest`。
- [ ] 檢查 `configs/default.toml`，確認只保留通用範例路徑。
- [ ] 若另行發布權重，使用 Release 或模型平台，並附來源、版本、授權及 SHA-256；不要直接提交到 Git。
- [ ] 初始化 Git 前確認作者信箱；若不希望公開私人信箱，可使用 GitHub noreply 信箱。

## 建立新 repository

將此資料夾移到預定位置後，在資料夾內執行：

```powershell
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```

在 `git add .` 後先執行 `git status`，再次確認沒有資料集、權重、輸出結果或私人文件被加入。
