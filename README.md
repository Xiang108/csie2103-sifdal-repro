# CSIE2103 重現：SIFDAL

學號：**M11417015**　姓名：**謝宇翔**  
課程：CSIE 2103 類神經網路｜Spring 2026

## 論文

**Boost UAV-based Object Detection via Scale-Invariant Feature Disentanglement and Adversarial Learning**（IEEE TGRS 2025）  
官方程式：https://github.com/1e12Leon/SIFDAL

## 重現狀態

| 項目 | 狀態 |
|------|------|
| 資料集 | VisDrone train 6471 / val 548（轉 VOC） |
| 訓練 | 300 epoch，`FREEZE_ALL=1` 全程凍結 backbone 重訓中 |
| 舊 run | epoch 50 解凍後 mAP 崩潰，結果不可直接對照論文 |

詳見 [`results.json`](results.json)、`outputs/train_300e.log`。

## 如何重現

```bash
cd reproduction
FREEZE_ALL=1 python3 scripts/run_sifdal_full_train.py
python3 scripts/plot_sifdal_figures.py   # 訓練完成後
```

環境：Python 3.12、PyTorch 2.12+cu128、GPU。

## 與論文差距

- 舊版 300e 解凍策略導致 mAP 崩潰
- 改以全程凍結 backbone 對齊可重現訓練規模（重訓進行中）

## 改善方向

- 對齊論文解凍時程與作者預訓練權重
- 訓練完成後更新本 repo 的 `results.json` 與 figures
