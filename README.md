# 聯邦學習停車推薦與位置隱私保護專案

本專案實作基於聯邦學習（Federated Learning）的停車場空位率預測與推薦，並包含位置隱私攻擊模擬分析。

## 專案目的

1. 使用多個停車場客戶端進行分散式 LSTM 訓練，避免集中原始數據傳輸。
2. 比較聯邦平均（FedAvg）與加權聯邦平均（FedWAvg）聚合策略。
3. 用 `attack.py` 模擬攻擊者重建位置往返真實位置，評估隱私保護能力（重疊率 overlap）。

## 檔案說明

- `functions.py`
  - 讀取資料、計算 `vacancy_rate`
  - 建 `torch.utils.data.Dataset` 與 `DataLoader`
  - 建立 LSTM 模型與優化器
  - 指標計算 (MSE, RMSE, MAE, MAPE, R2)

- `LSTM.py`
  - 定義 LSTMModel
  - 本地訓練函式 `train_model`
  - 聯邦平均與加權平均聚合函式
  - 聯邦訓練迴圈 `federated_learning_cycle`

- `fed_avg.py`
  - FedAvg 版本：簡單平均 local 權重

- `fed_wei.py`
  - FedWAvg 版本：依每個 Client 資料量加權平均

- `networks.py`
  - 執行完整訓練/驗證/測試流程
  - 每 10 個 epoch 寫 log，並記錄最好的模型至測試集評估

- `attack.py`
  - 位置隱私攻擊模擬：隨機生成真實 vs 推斷位置圓域
  - 計算幾何重疊率 (intersection/union)

## 執行環境

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- shapely
- tqdm

## 使用步驟

1. 放置資料檔 `occupancy.csv` 與 `inf.csv` 于 `functions.py` 內相同路徑 (可修改 `read_data` 提供的路徑)。
2. 執行聯邦學習流程：
   - `python LSTM.py`（或 `python fed_avg.py` / `python fed_wei.py`，如需要簡化版）
3. 執行測試/實驗結果：
   - `python networks.py`，結果輸出到 `test29` 檔案。
4. 執行隱私攻擊評估：
   - `python attack.py`

## 進階擴充建議

- 加入差分隱私機制 `DP-SGD` 或噪音注入於本地更新。
- 改為安全聚合（Secure Aggregation）避免服務器見到明文權重。
- 將 `attack.py` 模型從隨機位置改為根據 LSTM 預測結果做攻擊分析。
- 擴展客戶端 `ClientSelection`、異質性（non-IID）實驗。

## 結果解讀

- `LSTM.py / fed_avg.py / fed_wei.py` 的訓練 loss 趨近：說明模型學到空位率時間序列。
- `attack.py` 重疊率越低越安全；若越高表示對手能復原真實位置範圍。

---
