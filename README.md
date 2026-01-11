# multimodal-web-app（純前端展示）

期末專題展示頁（純前端）：手寫數字辨識（ONNX Runtime Web）+ 語音數字辨識（ONNX + 麥克風）+ 準確度統計（內建 MNIST 測試樣本）+ 專題說明頁。

## 模型設計與架構（摘要）

本專題的訓練/匯出（PyTorch）在另一個資料夾 `multimodal_mobile_project/`；本 repo 的網頁端只負責「前處理 + 推論 + UI 展示」。

**任務**
- 手寫數字辨識：輸入 28×28 灰階影像，輸出 0–9 分類（10 類）
- 語音數字辨識：輸入語音的 MelSpectrogram（128×32），輸出 0–9 分類（10 類）

**模型結構（PyTorch）**
- `ImageEncoder`：2 層 Conv+Pool → FC → 128 維影像特徵
- `SpeechEncoder`：3 層 Conv+BN+ReLU+Pool → AdaptiveAvgPool → FC → 128 維語音特徵
- `MultiModalModel`：
  - 影像-only：`ImageEncoder` → `image_classifier(128→10)`
  - 語音-only：`SpeechEncoder` → `speech_classifier(128→10)`
  - 多模態（示範用）：concat(影像特徵, 語音特徵) → `classifier(256→10)`

**匯出與部署**
- 影像模型 ONNX：`public/image_model.onnx`（由 `multimodal_mobile_project/convert_to_onnx.py` 匯出）
- 語音模型 ONNX：`public/speech_model.onnx`（由 `multimodal_mobile_project/convert_speech_to_onnx.py` 匯出）
- 瀏覽器推論：`onnxruntime-web`（WASM），完全不需要後端

## 使用的資料集與前處理（摘要）

**影像（MNIST）**
- 資料集：MNIST（0–9 手寫數字）
- 前處理：
  - 取灰階像素 →（依筆畫反色/置中縮放）→ 28×28
  - Normalize：`(x - mean) / std`，其中 mean=0.1307、std=0.3081（MNIST 常用設定）
- 網頁端實作：`src/lib/preprocessMnist.ts`（並顯示 28×28 預覽）

**語音（SpeechCommands digits）**
- 資料集：SpeechCommands v0.02（只取 zero~nine 10 類）
- 前處理（訓練端）：MelSpectrogram
  - sample_rate=16000、n_fft=1024、hop_length≈512、n_mels=128
  - 取 1 秒音訊後得到 MelSpectrogram 128×32
- 網頁端實作：`src/lib/melSpectrogram.ts` + `src/lib/audio.ts`
  - 支援麥克風錄音（1 秒）或上傳音檔（取中間 1 秒）再做推論

## 準確度展示

- 頁面：介面上的「準確度」分頁
- 資料：`public/mnist_eval_samples.json`（內建 300 筆 MNIST 測試樣本）
- 輸出：Accuracy（%）+ 混淆矩陣（真值×預測）

## 開發/執行

在 `/home/wushengyong/multimodal-web-app`：

- 安裝依賴：`npm i`
- 開發：`npm run dev`
- 打包：`npm run build`
- 本機預覽 dist：`npm run preview`

注意：
- 麥克風錄音（`getUserMedia`）通常需要 `https://` 或 `http://localhost`，不要用 `file://` 直接打開 `dist/index.html`。

## 檔案輸出

- 成品靜態檔：`dist/`
- 靜態資源（模型/wasm/資料）：`public/`

## 雲端部署（靜態網站）

這個專案可以直接當「靜態網站」部署（不用後端）。

### GitHub Pages（推薦：自動部署）

已內建 GitHub Actions 工作流程：`multimodal-web-app/.github/workflows/deploy-pages.yml`

1. 把整個專案（不含 `node_modules/`、不含 `dist/`）推到 GitHub repo
2. 到 GitHub → Settings → Pages → Source 選 `GitHub Actions`
3. 之後每次 push 到 `main` 都會自動 build 並部署

### 其他平台

- Netlify / Vercel / Cloudflare Pages：Build Command=`npm run build`、Output=`dist`

提醒：
- 若要用語音辨識/麥克風，部署網址必須是 `https://`。

## 像 App 一樣使用

本專案包含 `public/manifest.webmanifest` + `public/sw.js`，在支援的瀏覽器上可用「加入主畫面/安裝」方式以近似 App 的方式使用。
