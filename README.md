# CSIE2103 重現紀錄：SIFDAL

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

## 論文

**Boost UAV-based Object Detection via Scale-Invariant Feature Disentanglement and Adversarial Learning**（IEEE TGRS 2025）  
官方程式：https://github.com/1e12Leon/SIFDAL

## 本 repo 內容

本 repo 為**期末作業重現結果紀錄**，包含：

| 檔案 | 說明 |
|------|------|
| [`results.json`](results.json) | 重現數值摘要（訓練完成後更新） |
| [`outputs/train_300e.log`](outputs/train_300e.log) | 訓練 log |

> 模型權重體積過大，未上傳至此 repo。訓練完成後將補上 `figures/`。

## 重現設定

| 項目 | 設定 |
|------|------|
| 框架 | PyTorch YOLOv7 + SIFAD（SIFDAL 官方實作） |
| 資料集 | VisDrone2019-DET（train 6471 / val 548，VOC 格式） |
| 訓練 | 300 epoch，`FREEZE_ALL=1` 全程凍結 backbone |
| 環境 | Python 3.12、PyTorch 2.12+cu128、GPU |

## 重現狀態

| 項目 | 狀態 |
|------|------|
| 訓練 | 🔄 300 epoch 重訓進行中 |
| 舊 run | epoch 50 解凍後 mAP 崩潰，結果不可直接對照論文 |

## 如何重現（參考官方 repo）

```bash
git clone https://github.com/1e12Leon/SIFDAL.git
cd SIFDAL
# 依官方 README 安裝環境、準備 VisDrone VOC 資料並訓練
```

## 差距原因與改善方向

- **可能原因**：舊版解凍策略導致 mAP 崩潰；改以全程凍結 backbone 重訓
- **改善方向**：對齊論文解凍時程與作者預訓練權重；訓練完成後更新本 repo 的 `results.json` 與 figures
