# CSIE2103 重現：SIFDAL

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

**Boost UAV-based Object Detection via Scale-Invariant Feature Disentanglement and Adversarial Learning**（IEEE TGRS 2025）  
官方：https://github.com/1e12Leon/SIFDAL

---

## 快速開始

### 1. 環境

```bash
git clone https://github.com/Xiang108/csie2103-sifdal-repro.git
cd csie2103-sifdal-repro

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow tqdm tensorboard
```

### 2. 資料集 → VOC

下載 VisDrone2019-DET 至 `datasets/VisDrone-DET/`，執行轉換：

```bash
python3 scripts/visdrone_to_voc.py
```

會產生 `sifdal/SIFAD/VOCdevkit/` 與 `2007_train.txt`、`2007_val.txt`（train 6471 / val 548）。

### 3. 訓練

```bash
python3 scripts/run_train.py
```

| 變數 | 預設 | 說明 |
|------|------|------|
| `SIFDAL_GPU` | 1 | GPU 編號 |
| `SIFDAL_EPOCHS` | 300 | epoch |
| `SIFDAL_FREEZE_ALL` | 1 | 1=全程凍結 backbone |
| `DATA_ROOT` | `./datasets/VisDrone-DET` | 原始資料 |

### 4. 輸出

| 路徑 | 內容 |
|------|------|
| `sifdal/SIFAD/logs/` | loss、mAP 紀錄 |
| `results.json` | 最佳 / 最終 mAP |
| `figures/` | 訓練完成後產生 |
| `outputs/train_300e.log` | 訓練 log |

---

## 目錄結構

```
├── sifdal/SIFAD/      # SIFDAL 官方程式（含 patch）
├── scripts/
│   ├── visdrone_to_voc.py
│   ├── run_train.py
│   └── plot_figures.py
├── device_utils.py
├── results.json
└── outputs/
```

## 重現狀態

300 epoch 重訓進行中（`FREEZE_ALL=1`）。舊 run 解凍後 mAP 崩潰，改以全程凍結 backbone 重訓。
