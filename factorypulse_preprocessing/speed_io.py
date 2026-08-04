"""
變轉速資料集的讀取工具（直接從 zip 串流，不解壓縮）。

為什麼不解壓：三個 zip 加起來 20GB，解開後是 52GB。
Python 的 zipfile 支援串流讀取單一檔案，pandas 可以直接吃那個 file object，
所以完全不需要在硬碟上放解壓後的副本。

檔案結構（每個 zip 內都是扁平的，沒有子資料夾）：
    vibration_{class}_{profile}.csv   欄位: bearingA_x, bearingA_y, bearingB_x, bearingB_y
    current_{class}_{profile}.csv     欄位: current_R, current_S, current_T
    rpm_{class}_{profile}.csv         欄位: time, rpm

    class   : normal | inner | outer | ball
    profile : 0~6（7 種轉速變化曲線），另有 vibration_{class}_constant.csv
              （固定轉速版本，但沒有對應的 rpm/current 檔，預設不處理）
"""

import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

import config

VIBRATION_COLS = ["bearingA_x", "bearingA_y", "bearingB_x", "bearingB_y"]
CURRENT_COLS = ["current_R", "current_S", "current_T"]


@lru_cache(maxsize=1)
def build_zip_index() -> dict:
    """掃描三個 zip，建立 {檔名: zip路徑} 的索引。

    因為 Subset1/2/3 各自裝了不同的 profile（0-2 / 3-4 / 5-6），
    要用檔名反查它在哪一個 zip 裡。
    """
    index = {}
    for zpath in config.SPEED_ZIPS:
        if not zpath.exists():
            raise FileNotFoundError(
                f"找不到 zip：{zpath}\n"
                f"請確認 Subset1/2/3 三個資料夾都在 {config.SPEED_DATA_ROOT}"
            )
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    index[name] = zpath
    return index


def list_available(index: dict | None = None) -> pd.DataFrame:
    """列出所有可用的 (class, profile) 組合，以及三種檔案是否齊全。"""
    index = index or build_zip_index()
    rows = {}
    for name in index:
        stem = name[:-4]  # 去掉 .csv
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        kind, cls, profile = parts[0], parts[1], "_".join(parts[2:])
        key = (cls, profile)
        rows.setdefault(key, {"condition_raw": cls, "profile": profile})
        rows[key][kind] = True

    df = pd.DataFrame(rows.values())
    for col in ["vibration", "current", "rpm"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].notna() & (df[col] == True)  # noqa: E712
    return df.sort_values(["condition_raw", "profile"]).reset_index(drop=True)


def _read_csv_from_zip(name: str, index: dict, **kwargs) -> pd.DataFrame:
    zpath = index.get(name)
    if zpath is None:
        raise FileNotFoundError(f"三個 zip 裡都找不到 {name}")
    with zipfile.ZipFile(zpath) as z:
        with z.open(name) as f:
            return pd.read_csv(f, **kwargs)


def load_vibration(condition_raw: str, profile, index: dict | None = None) -> np.ndarray:
    """回傳 shape=(N, 4) 的振動訊號，欄位順序與『變負載資料集』一致：
        bearingA_x, bearingA_y, bearingB_x, bearingB_y
        ↔ x_direction_housing_A, y_direction_housing_A, x_..._B, y_..._B
    所以同一套 features_vibration 可以直接沿用。
    """
    index = index or build_zip_index()
    name = f"vibration_{condition_raw}_{profile}.csv"
    df = _read_csv_from_zip(name, index, dtype=np.float32)
    missing = [c for c in VIBRATION_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 缺少欄位 {missing}，實際欄位：{list(df.columns)}")
    return df[VIBRATION_COLS].to_numpy(dtype=np.float32)


def load_current(condition_raw: str, profile, index: dict | None = None,
                 decimate: int | None = None) -> np.ndarray:
    """回傳 shape=(N, 3) 的三相電流 (R, S, T)。

    ⚠️ 這個資料集的電流是 100 kHz 取樣（振動是 25.6 kHz），兩者並非同步取樣。

    decimate: 降採樣倍率，預設用 config.SPEED_CURRENT_DECIMATE（=4，100k -> 25k）。
        用「連續 N 點取平均」而不是直接跳點，因為跳點沒有抗混疊處理會把
        高頻雜訊摺疊回低頻，污染我們真正要看的諧波與邊帶。
        區塊平均本身就是一個低通濾波器，第一個零點在 fs/N = 25kHz，
        對我們關心的 1kHz 以下訊號完全無損。
        設 decimate=1 可以拿到原始 100kHz 資料。

    這個資料集三相都完整（變負載資料集的 BPFO 檔案缺 V/W 相），
    但為了讓兩邊特徵欄位一致，build_dataset_speed.py 只會用第一相（R）。

    ⚠️ 實作上採用「分塊讀取 + 邊讀邊降採樣」。
        電流檔有 3000 萬行，pandas 一次讀進來光是解析暫存就會吃掉數 GB，
        實測在 3GB 記憶體的環境會直接被系統 OOM kill（而且沒有 traceback，
        只會看到程式無聲無息消失）。分塊之後尖峰記憶體降到約 100MB。
    """
    index = index or build_zip_index()
    if decimate is None:
        decimate = config.SPEED_CURRENT_DECIMATE
    decimate = max(1, int(decimate))

    name = f"current_{condition_raw}_{profile}.csv"
    zpath = index.get(name)
    if zpath is None:
        raise FileNotFoundError(f"三個 zip 裡都找不到 {name}")

    # chunksize 必須是 decimate 的整數倍，否則區塊平均會跨越 chunk 邊界而錯位
    chunk_rows = 4_000_000 - (4_000_000 % decimate)

    pieces = []
    with zipfile.ZipFile(zpath) as z:
        with z.open(name) as f:
            reader = pd.read_csv(f, dtype=np.float32, chunksize=chunk_rows)
            leftover = None
            for chunk in reader:
                missing = [c for c in CURRENT_COLS if c not in chunk.columns]
                if missing:
                    raise KeyError(
                        f"{name} 缺少欄位 {missing}，實際欄位：{list(chunk.columns)}"
                    )
                arr = chunk[CURRENT_COLS].to_numpy(dtype=np.float32)
                del chunk

                if leftover is not None and len(leftover):
                    arr = np.concatenate([leftover, arr], axis=0)
                    leftover = None

                if decimate > 1:
                    n = (arr.shape[0] // decimate) * decimate
                    if n < arr.shape[0]:
                        leftover = arr[n:].copy()   # 不足一組的尾巴留到下一塊
                    arr = arr[:n].reshape(n // decimate, decimate, arr.shape[1]).mean(axis=1)
                pieces.append(arr.astype(np.float32))

    return np.concatenate(pieces, axis=0) if pieces else np.empty((0, 3), np.float32)


def infer_fs(n_samples: int, duration_sec: float = None) -> float:
    """由樣本數與錄音長度推算取樣率。

    寫死取樣率很容易踩雷（這個資料集振動 25.6kHz、電流 100kHz 就不一樣），
    所以在流程裡用這個函式核對，不一致時會及早發現。
    """
    duration_sec = duration_sec or config.SPEED_RECORD_SEC
    return n_samples / duration_sec


def load_rpm(condition_raw: str, profile, index: dict | None = None) -> pd.DataFrame:
    """回傳轉速時間序列，欄位 time(秒) / rpm。

    取樣率很低（約 9 Hz，每 0.109 秒一點），與振動的 25600 Hz 不同步，
    所以要用內插的方式對應到每個振動窗（見 rpm_for_windows）。
    """
    index = index or build_zip_index()
    name = f"rpm_{condition_raw}_{profile}.csv"
    return _read_csv_from_zip(name, index)


def rpm_for_windows(rpm_df: pd.DataFrame, window_start_times: np.ndarray,
                    window_sec: float) -> np.ndarray:
    """算出每個時間窗對應的平均轉速。

    做法：把低頻的 rpm 序列線性內插到窗的起點與終點，取兩者平均。
    比只取窗起點的瞬時值穩定，也反映窗內轉速正在變化的事實。
    """
    t = rpm_df["time"].to_numpy(dtype=np.float64)
    r = rpm_df["rpm"].to_numpy(dtype=np.float64)
    start = np.interp(window_start_times, t, r)
    end = np.interp(window_start_times + window_sec, t, r)
    return ((start + end) / 2).astype(np.float32)


if __name__ == "__main__":
    idx = build_zip_index()
    print(f"三個 zip 內共有 {len(idx)} 個 csv 檔\n")
    avail = list_available(idx)
    print(avail.to_string(index=False))
    print()
    complete = avail[avail.vibration & avail.current & avail.rpm]
    print(f"三種檔案都齊全的組合：{len(complete)} 組"
          f"（{complete.condition_raw.nunique()} 個類別 × 各轉速曲線）")
