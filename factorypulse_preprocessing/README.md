# FactoryPulse 前處理程式碼

資料集：*Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating
Machine Under Varying Load Conditions for Fault Diagnosis*（KAIST, Mendeley DOI
10.17632/ztmf3m7h5x, **CC BY 4.0** — 記得在簡報 / 報告附錄引用這篇）。

## 我幫你檢查過你資料夾裡的實際狀況

- `vibration/`：45 個檔案（3 負載 × 15 種條件），OK。
- `current,temp/`：45 個檔案，OK。
- `acoustic/`：**只有 5 個檔案**（`0Nm_BPFI_03/10`, `0Nm_BPFO_03/10`, `0Nm_Normal`），
  缺了 `0Nm_BPFO_30`、所有 `Misalign`、所有 `Unbalance`，以及全部 `2Nm`/`4Nm`。
  **這個要先回 Mendeley 資料集頁面確認是不是分成多個 part，把剩下的補齊**，
  不然聲音模型只能做「0Nm 下 BPFI vs BPFO vs Normal」三分類，會拖累整個多感測
  融合 demo 的完整度。
- 振動檔名有個原始資料集本身的拼字錯誤：`2Nm` 資料夾裡 `Unbalance` 被打成
  `Unbalalnce`（`current,temp/` 裡對應檔案拼字是對的）。已經在
  `label_utils.py` 和 `build_dataset.py` 的比對邏輯裡處理掉了，不用手動改檔名。
- 檔案都是 MATLAB v5 格式（`scipy.io.loadmat` 讀得動）、TDMS 是 NI FlexLogger
  格式（`nptdms` 讀得動），格式本身沒問題。

## 安裝

沙盒環境沒有對外網路，沒辦法在這裡直接跑，**請在你自己的電腦或 Colab 執行**：

```bash
pip install -r requirements.txt
```

## 執行順序

1. **先跑一次 `inspect_tdms.py`**，核對電流/溫度欄位名稱：

   ```bash
   python inspect_tdms.py "acoustic_temp_vibration/current,temp/0Nm_Normal.tdms"
   ```

   我在沙盒裡只能用 `strings` 粗略看到 header 裡有 `Temperature`、
   `NI_CjcTemperature`、`Current`、`Voltage` 這些關鍵字，沒辦法確認完整的
   group/channel 樹狀結構長怎樣。`io_utils.py` 裡的欄位比對是「保守模糊比對」，
   如果報錯说找不到某個欄位，把這支腳本印出來的實際名稱貼給我，我再幫你調整
   `io_utils.py` 裡的判斷邏輯。

2. 把 `config.py` 裡的 `DATA_ROOT` 改成你資料實際的路徑。

3. 跑主流程：

   ```bash
   python build_dataset.py                    # 處理振動/電流/溫度三個模態（預設不含聲音）
   python build_dataset.py --include-acoustic  # 聲音資料補齊到 45 個檔案後才加這個 flag
   ```

   目前決定先不做聲音模型（見下方「關於拿掉聲音模態」），所以預設行為已經改成
   **不處理聲音**，不用每次都手動加 `--skip-acoustic`。

4. 輸出會在 `processed/` 資料夾：

   - `manifest.csv`：每個檔案的標籤（load / condition / severity / severity_level）
   - `vibration_windows_{train,val,test}.npz`：1D CNN 用的原始波形
   - `vibration_features_{split}.csv`：手工特徵（RMS/峰值因子/頻帶能量...），
     可以先拿去跑 RandomForest/XGBoost 當 baseline，比等 CNN 訓練好快很多
   - `current_features_{split}.csv`：三相電流 MCSA 特徵（含三相不平衡度）
   - `temperature_features_{split}.csv`：溫度趨勢特徵（已經用同負載下的 Normal
     檔案算出 baseline 溫度）
   - `acoustic_specs_{split}.npz`：mel spectrogram，給 2D CNN（若聲音檔齊全）

## 為什麼 train/val/test 是「同一檔案內部按時間切」，不是「整個檔案分組」

這組資料每個 (load, condition, severity) 只有一個檔案，如果直接把整個檔案分去
train 或 test，某些故障類別會完全不在訓練集裡，或完全不在測試集裡。所以
`build_dataset.py` 對每個檔案的訊號依時間順序切成前 70%(train) / 中間
15%(val) / 後 15%(test)，切窗完全在各自的時間區段內進行，不會有同一段訊號的
窗同時出現在不同 split。

如果之後想做「跨負載泛化測試」（例如 0Nm/2Nm 訓練、4Nm 測試，驗證模型不是
背答案），可以另外寫一個 `split_by_load()`（我可以幫你加），這個對評審來說是
很有說服力的實驗設計，回應「你們是不是資料洩漏」的質疑。

## 關於拿掉聲音模態

聲音資料只有 5/45 個檔案（0Nm 下 BPFI/BPFO/Normal），資料量太少，不足以支撐一個
獨立訓練/驗證的模型，決定先不做。對應的調整：

- `build_dataset.py` 預設不處理聲音（`--include-acoustic` 才會打開），
  `features_acoustic.py` 保留著沒刪，之後資料補齊隨時可以接回來。
- 系統從「四模態融合」改成「三模態融合：振動 + 電流 + 溫度」，GPT 原稿裡聲音
  20~30% 的權重要重新分配（建議：振動 55~65% / 電流 20~25% / 溫度 10~15%，
  維持振動仍是主要診斷來源的邏輯不變）。
- 規則引擎裡跟聲音有關的規則（例如「振動與聲音同時異常」「只有聲音異常→環境
  噪音」）要拿掉或改寫成只用振動+電流+溫度的組合。
- 簡報/提案文件裡如果還留著「聲音模型」的描述，建議改成「聲音偵測列為未來擴充
  方向，目前驗證資料量不足」，這樣講反而顯得工程判斷紮實，比硬做一個資料量
  不夠、demo 時容易被問倒的模型更安全。

## 已知限制 / 下一步

- 這邊的 sandbox 沒有對外網路，`scipy`/`nptdms`/`librosa` 都裝不了，所以
  `io_utils.load_vibration_mat` 等實際讀檔的函式**沒有在真實資料上跑過**，
  只驗證過：檔名解析（`label_utils.py`）、切窗與時間切分（`windowing.py`）、
  以及振動/電流/溫度的特徵計算邏輯（用合成訊號驗證數值方向正確，例如
  「有週期性衝擊的假訊號」峰值因子與高頻能量確實比正常訊號高）。
- 第一次在你自己的環境跑完，如果哪一步報錯，把錯誤訊息貼給我，我可以直接對症
  下藥修 `io_utils.py`（最可能出錯的就是 tdms 的欄位比對，因為我看不到完整
  channel 清單）。
