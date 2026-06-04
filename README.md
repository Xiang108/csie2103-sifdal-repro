# CSIE2103 重現：SIFDAL（UAV 物件偵測）

學號：M11417015　姓名：謝宇翔  
課程：CSIE2103 類神經網路（114-2）

## 論文

Boost UAV-based Object Detection via Scale-Invariant Feature Disentanglement and Adversarial Learning（IEEE TGRS 2025）

官方程式：https://github.com/1e12Leon/SIFDAL

## 環境

- Python 3.12
- PyTorch 2.12+cu128（RTX 5090）
- 依 `reproduction/device_utils.py` 自動選 GPU／CPU

## 快速重現

```bash
cd reproduction/sifdal
python3 setup_voc_from_stateair.py
cd SIFAD
python3 voc_annotation.py
python3 train.py
```

訓練完成後權重在 `SIFAD/logs/best_epoch_weights.pth`。

## 資料集

使用官方 repo 內附 **State-Air** 小樣本（7 張，VOC 格式）。  
完整資料集需至作者百度網盤下載；本重現僅驗證流程可跑通。

## 本次結果（課堂重現設定：8 epoch、小樣本）

| 項目 | 論文（完整資料） | 本次重現 |
|------|------------------|----------|
| 資料集 | VisDrone / UAVDT / State-Air | State-Air 子集（7 張） |
| 指標 | mAP（多資料集 SOTA） | val mAP ≈ 0%（樣本少、訓練短） |
| Train Loss | — | ≈ 6.61（epoch 8） |
| Val Loss | — | ≈ 6.54 |

## 差距說明（學生重現）

1. 僅用 repo 內 7 張示範圖，無法對齊論文大規模實驗。  
2. 訓練 epoch 少（8），模型尚未收斂，mAP 接近 0 屬預期。  
3. 未下載作者完整 State-Air／VisDrone 訓練集。

## 改善方向

- 下載完整 State-Air 或 VisDrone，訓練至少 100+ epoch。  
- 使用作者提供的預訓練權重（若有）再 fine-tune。  
- GPU 空閒時延長訓練並調整 batch size。

## 跨資料集（簡易）

可將 `setup_voc_from_stateair.py` 產生的 VOC 結構套用到其他 VOC 格式 UAV 資料；本次未另跑第二資料集。
