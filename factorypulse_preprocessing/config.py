"""
FactoryPulse 前處理設定檔
==========================
對應資料集：Vibration, Acoustic, Temperature, and Motor Current Dataset of
Rotating Machine Under Varying Load Conditions for Fault Diagnosis (KAIST,
Mendeley DOI 10.17632/ztmf3m7h5x, CC BY 4.0)

實測到的資料夾結構（使用者環境）：
  acoustic_temp_vibration/
    vibration/        *.mat   (MATLAB v5, 4 通道: x/y * housing A/B, 單位 g)
    acoustic/          *.mat   (MATLAB v5, 1 通道麥克風, 單位 Pa) -- 目前只有 5/45 個檔案！
    current,temp/      *.tdms  (NI TDMS, 2 溫度 + 3 相電流(+電壓), 單位 degC / A)

檔名規則： "{load}Nm_{condition}_{severity}.{ext}"
  load      : 0 / 2 / 4   (Nm 扭矩負載)
  condition : Normal | BPFI | BPFO | Misalign | Unbalance
              (原始資料夾裡 2Nm 的振動檔名有拼字錯誤 "Unbalalnce"，已在
               label_utils.py 內做正規化，不用手動改檔名)
  severity  : Normal 無 severity
              BPFI / BPFO   -> "03" / "10" / "30"   (等級 1/2/3，數值越大越嚴重)
              Misalign      -> "01" / "03" / "05"   (等級 1/2/3)
              Unbalance     -> "0583mg" ... "3318mg" (等級 1~5，依質量由小到大)
"""

from pathlib import Path

# ---- 路徑設定：請依實際存放位置修改 ----
# 使用相對於這個 config.py 檔案本身的路徑，確保不論從哪個資料夾執行都能正確定位
DATA_ROOT = Path(__file__).parent.parent / "acoustic_temp_vibration"
VIBRATION_DIR = DATA_ROOT / "vibration"
ACOUSTIC_DIR = DATA_ROOT / "acoustic"
CURRENT_TEMP_DIR = DATA_ROOT / "current,temp"

OUTPUT_DIR = Path("./processed")

# ---- 取樣率（依論文 / Mendeley 說明）----
VIBRATION_FS = 25600      # Hz
ACOUSTIC_FS = 25600       # Hz  (與振動同一台 Siemens DAQ)
CURRENT_TEMP_FS = 50000   # Hz  (NI DAQ 原始取樣率；溫度變化緩慢，會另外降採樣)

# ---- 切窗設定 ----
WINDOW_SEC = 1.0          # 每個訓練樣本的時間長度（秒）
WINDOW_OVERLAP = 0.5      # 重疊比例 (0~1)，正常段錄影較長，異常段只有 60 秒，
                           # overlap 越大樣本數越多，但也越容易讓 train/val 太像

# ---- 溫度另外的統計窗 ----
TEMP_RESAMPLE_SEC = 1.0   # 溫度先降採樣到每秒一點，再做趨勢特徵

# ---- 嚴重度等級對照（拿來做健康分數 regression / ordinal 標籤）----
SEVERITY_ORDER = {
    "Normal": {None: 0},
    "BPFI": {"03": 1, "10": 2, "30": 3},
    "BPFO": {"03": 1, "10": 2, "30": 3},
    "Misalign": {"01": 1, "03": 2, "05": 3},
    "Unbalance": {"0583mg": 1, "1169mg": 2, "1751mg": 3, "2239mg": 4, "3318mg": 5},
}

FAULT_CLASSES = ["Normal", "BPFI", "BPFO", "Misalign", "Unbalance"]

RANDOM_SEED = 42


# =====================================================================
# 第二個資料集：變轉速（Subset1/2/3）
# ---------------------------------------------------------------------
# 論文：Vibration and Motor Current Dataset of Rolling Element Bearing
#       Under Varying Speed Conditions for Fault Diagnosis (KAIST)
#
# 與上面「變負載」資料集的關係：
#   相同：振動 4 通道（bearingA_x/y, bearingB_x/y 對應 housing A/B 的 x/y）
#         取樣率同樣是 25,600 Hz、每檔 300 秒
#   不同：1. 變動的是「轉速」(618~2480 RPM)，不是負載（那邊固定 3010 RPM）
#         2. 類別是 normal/inner/outer/ball —— 多了 ball（滾動體故障），
#            但沒有 misalign / unbalance
#         3. 沒有溫度資料，只有振動 + 電流
#         4. 電流三相 R/S/T 都完整（變負載資料集的 BPFO 缺 V/W 相）
#
# ⚠️ 兩個資料集不可以直接混成同一個分類器：類別集合不同、運轉條件不同、
#    感測器組合也不同，硬混會讓模型學到「這筆來自哪個資料集」而不是故障特徵。
#    正確做法是各自產生特徵檔（欄位格式相容），訓練時再決定怎麼用。
# =====================================================================

SPEED_DATA_ROOT = Path(__file__).parent.parent

# 三個 zip 的相對路徑（直接從 zip 串流讀取，不解壓縮 —— 解開要 52GB）
SPEED_ZIPS = [
    SPEED_DATA_ROOT / "Subset1" / (
        "Vibration and Motor Current Dataset of Rolling Element Bearing "
        "Under Varying Speed Conditions for Fault Diagnosis Subset1"
    ) / "part1.zip",
    SPEED_DATA_ROOT / "Subset2" / (
        "Vibration and Motor Current Dataset of Rolling Element Bearing "
        "Under Varying Speed Conditions for Fault Diagnosis Subset2"
    ) / "part2.zip",
    SPEED_DATA_ROOT / "Subset3" / (
        "Vibration and Motor Current Dataset of Rolling Element Bearing "
        "Under Varying Speed Conditions for Fault Diagnosis Subset3"
    ) / "part3.zip",
]

SPEED_OUTPUT_DIR = Path("./processed_speed")

SPEED_RECORD_SEC = 300.0       # 每個檔案的錄音長度（由 rpm 檔的 time 欄位確認）

SPEED_VIBRATION_FS = 25600     # Hz，實測 7,680,000 行 / 300 秒

# ⚠️ 電流的取樣率是 100 kHz，不是 25.6 kHz —— 實測 30,000,000 行 / 300 秒。
#    振動與電流「不同步取樣」，這點跟變負載資料集不一樣（那邊兩者都是 25.6k）。
#    如果照抄 25600 會導致：窗長只截到 0.256 秒的資料、轉速對應的時間也全錯。
SPEED_CURRENT_FS_RAW = 100000  # Hz，原始取樣率

# 電流降採樣倍率：100kHz 對 MCSA 來說是過度取樣（我們關心的諧波與邊帶都在 1kHz 以下），
# 降到 25kHz 有兩個好處：
#   1. 特徵計算快 7 倍（實測 21ms/窗 -> 3ms/窗）
#   2. 頻寬與「變負載資料集」的 25.6kHz 接近，兩個資料集的電流特徵才可比較
#      （kurtosis、envelope 這類特徵會隨頻寬改變，頻寬不同就不能直接比）
SPEED_CURRENT_DECIMATE = 4
SPEED_CURRENT_FS = SPEED_CURRENT_FS_RAW // SPEED_CURRENT_DECIMATE  # 25000 Hz

# 把這個資料集的類別名稱對應到「變負載資料集」的命名，方便日後比較／合併
SPEED_CONDITION_MAP = {
    "normal": "Normal",
    "inner": "BPFI",    # inner race fault = 軸承內圈，與變負載資料集的 BPFI 同義
    "outer": "BPFO",    # outer race fault = 軸承外圈
    "ball": "Ball",     # 滾動體故障 —— 變負載資料集沒有這個類別
}

SPEED_FAULT_CLASSES = ["Normal", "BPFI", "BPFO", "Ball"]

# 速度設定檔編號 0~6（7 種不同的轉速變化曲線）
SPEED_PROFILES = [0, 1, 2, 3, 4, 5, 6]
