"""
把使用者上傳的「原始感測檔」轉成模型看得懂的特徵。

支援三種格式：
  .mat   振動原始波形（MATLAB v5，4 通道：軸承座 A/B 的 x/y 方向）
  .tdms  電流與溫度原始波形（NI FlexLogger）
  .csv   已經算好的特徵表（直接使用，不再處理）

轉換流程與 build_dataset.py 完全相同——去直流、切 1 秒窗、算同一組特徵，
所以上傳的檔案跑出來的特徵，與訓練時用的特徵定義一致，模型才吃得下。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features_current
import features_vibration
import features_vibration_v2
import io_utils
import windowing

SUPPORTED = (".mat", ".tdms", ".csv")


class IngestError(Exception):
    """轉檔過程中可以明確告訴使用者原因的錯誤。"""


def _save_temp(uploaded) -> Path:
    """Streamlit 的上傳物件是記憶體檔案，但 scipy/nptdms 需要真實路徑。"""
    suffix = Path(uploaded.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return Path(tmp.name)


def ingest_vibration_mat(path: Path, fs: float = config.VIBRATION_FS,
                         feature_version: str = "v2",
                         rpm: float = 3010.0,
                         load_nm: float | None = None,
                         max_seconds: float | None = 60.0) -> pd.DataFrame:
    """.mat 振動檔 -> 特徵表（每列一個 1 秒窗）。

    修正紀錄（這支函式原本有三個會讓上傳直接失敗的問題）：

    1. 【讀檔格式】原本用 sio.loadmat() 直接找 x_direction_housing_A 這類
       頂層欄位，但這批資料的實際格式是巢狀 struct（Signal.y_values.values）——
       也就是說，拿本專案自己的 .mat 檔上傳會得到「找不到預期的振動欄位」。
       改成直接呼叫 io_utils.load_vibration_mat()，與 build_dataset 走同一條路。

    2. 【特徵版本】原本固定呼叫 v1 的 extract_classical_features，
       但模型已經換成 v2。上傳後會出現「缺少模型需要的欄位（176 個）」。

    3. 【缺 load_nm】模型訓練時把 load_nm 當成特徵（負載對訊號的影響比故障還大，
       不給模型就分不出「4Nm 正常」與「0Nm 不對心」）。上傳的檔案沒有這個欄位，
       推論時一樣會缺欄位。所以這裡要求呼叫端傳入負載。

    max_seconds: 只取前 N 秒。原始檔一個 300 秒 × 25.6kHz，全算要好幾分鐘，
                 demo 時使用者會以為當掉。取 60 秒（約 119 個窗）已經足夠穩定。
                 傳 None 代表整段都算。
    """
    try:
        sig = io_utils.load_vibration_mat(path)      # (N, 4)
    except ImportError as e:
        raise IngestError(f"讀取 .mat 需要 scipy：pip install scipy（{e}）")
    except Exception as e:
        raise IngestError(
            f"讀取 .mat 失敗：{type(e).__name__}: {e}\n"
            f"預期格式為 4 通道振動（軸承座 A/B 的 x/y 方向）。"
        )

    if max_seconds:
        sig = sig[: int(max_seconds * fs)]

    wins = windowing.sliding_window(sig, fs, config.WINDOW_SEC, config.WINDOW_OVERLAP)

    if feature_version == "v2":
        rows = [features_vibration_v2.extract_features_v2(w, fs, rpm=rpm) for w in wins]
    else:
        rows = [features_vibration.extract_classical_features(w, fs) for w in wins]

    df = pd.DataFrame(rows)

    # 與訓練時丟掉同一批欄位，否則上傳檔案算出來的欄位集合會多出幾欄，
    # 雖然 X[feature_names] 會自動忽略多的欄位，但保持一致比較不會出事。
    import feature_selection as _fsel
    df = df.drop(columns=[c for c in _fsel.EXACT_DUPLICATES if c in df.columns])

    if load_nm is not None:
        df["load_nm"] = float(load_nm)
    return df


def ingest_current_tdms(path: Path) -> tuple[pd.DataFrame, dict]:
    """.tdms 檔 -> (電流特徵表, 溫度摘要)。"""
    try:
        import io_utils
    except ImportError as e:
        raise IngestError(f"讀取 tdms 需要 nptdms：pip install nptdms（{e}）")

    try:
        data = io_utils.load_current_temp_tdms(path)
    except Exception as e:
        raise IngestError(f"讀取 .tdms 失敗：{type(e).__name__}: {e}")

    fs = data.get("fs") or config.CURRENT_TEMP_FS
    cur = data["current_U"]          # 只用第一相，與訓練時一致
    wins = windowing.sliding_window(cur, fs, config.WINDOW_SEC, config.WINDOW_OVERLAP)
    rows = [features_current.extract_current_features(w, fs) for w in wins]

    temp_a = data.get("temperature_A")
    temp_summary = {}
    if temp_a is not None and len(temp_a):
        import features_temperature as ft
        t1 = ft.resample_to_seconds(temp_a, fs)
        # 沒有該設備的正常基準可比，用這段自己的起始溫度當參考點，
        # 只能看「相對變化」，不能看「絕對偏離」——畫面上要標示清楚
        baseline = float(np.mean(t1[: max(1, len(t1) // 10)]))
        temp_summary = ft.extract_temperature_features(t1, baseline)
        temp_summary["_baseline_note"] = "以本段訊號起始 10% 作為參考基準"

    return pd.DataFrame(rows), temp_summary


def ingest(uploaded, feature_version: str = "v2", rpm: float = 3010.0,
           load_nm: float | None = None) -> dict:
    """統一入口。回傳 {kind, features, temperature, n_windows, note}。

    feature_version / rpm / load_nm 會往下傳給 ingest_vibration_mat，
    呼叫端（app.py）必須依「當前模式的模型」決定這三個值，
    否則算出來的特徵與模型對不上。
    """
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in SUPPORTED:
        raise IngestError(f"不支援的格式 {suffix}，目前支援：{', '.join(SUPPORTED)}")

    if suffix == ".csv":
        df = pd.read_csv(uploaded)
        meta_like = [c for c in df.columns
                     if c in ("condition", "severity_level", "load_nm", "rpm",
                              "profile", "file_id", "split")]
        feats = df.drop(columns=meta_like).select_dtypes("number")
        return {
            "kind": "csv", "features": feats, "temperature": {},
            "n_windows": len(feats),
            "note": "已是特徵表，直接使用（未重新計算特徵）",
        }

    path = _save_temp(uploaded)
    try:
        if suffix == ".mat":
            feats = ingest_vibration_mat(
                path, feature_version=feature_version, rpm=rpm, load_nm=load_nm)
            return {
                "kind": "vibration", "features": feats, "temperature": {},
                "n_windows": len(feats),
                "note": f"已從原始波形切出 {len(feats)} 個 1 秒窗，"
                        f"以 {feature_version} 特徵定義計算（共 {feats.shape[1]} 欄）",
            }
        else:  # .tdms
            feats, temp = ingest_current_tdms(path)
            return {
                "kind": "current", "features": feats, "temperature": temp,
                "n_windows": len(feats),
                "note": f"已從原始波形切出 {len(feats)} 個 1 秒窗並計算電流特徵"
                        + ("，另含溫度趨勢摘要" if temp else ""),
            }
    finally:
        path.unlink(missing_ok=True)
