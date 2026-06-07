# CSIE2103 重現：SIFDAL

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

**SIFDAL**（IEEE TGRS 2025）  
官方：https://github.com/1e12Leon/SIFDAL

> **資料集會自動下載**：首次執行訓練時會下載 VisDrone train+val（壓縮約 **1.5 GB**，解壓後約 **43 GB**），不需手動下載、也不會 push 至 GitHub。詳見 [`datasets/README.md`](datasets/README.md)。


---

## 一鍵訓練

```bash
git clone https://github.com/Xiang108/csie2103-sifdal-repro.git
cd csie2103-sifdal-repro

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow tqdm tensorboard

python3 scripts/run_train.py
```

自動流程：**下載 VisDrone → 轉 VOC → 訓練 300 epoch（FREEZE_ALL=1）**。

| 環境變數 | 預設 |
|----------|------|
| `SIFDAL_GPU` | 1 |
| `SIFDAL_EPOCHS` | 300 |
| `SIFDAL_FREEZE_ALL` | 1 |
