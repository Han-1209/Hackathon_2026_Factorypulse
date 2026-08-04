"""
溫度特徵萃取。

溫度訊號變化慢，不用切成 1 秒窗跟振動/電流對齊，而是先把整段錄音降採樣成
「每秒一個點」的趨勢序列，再算移動平均、升溫速度、超過基準的時間等特徵。
這些數字之後可以拿來驗證/校正 GPT 建議書裡溫度規則的門檻值
（例如「10 分鐘內升溫超過 8°C」這種數字，先用這批真實資料看實際分布再定，
不要照抄原本憑感覺訂的門檻）。
"""

import numpy as np


def resample_to_seconds(temp: np.ndarray, fs: float, target_hz: float = 1.0) -> np.ndarray:
    """把高取樣率的溫度訊號降到每秒一點（用區塊平均，不是單純跳點取樣）。"""
    block = int(round(fs / target_hz))
    if block <= 1:
        return temp
    n_blocks = len(temp) // block
    trimmed = temp[: n_blocks * block]
    return trimmed.reshape(n_blocks, block).mean(axis=1)


def extract_temperature_features(temp_series_1hz: np.ndarray, baseline_temp: float) -> dict:
    """temp_series_1hz: 已經降到 1Hz 的溫度序列（單位 degC）
    baseline_temp: 這台設備在正常狀態下的平均溫度（需要另外從 Normal 檔案算出來，
                   在 build_dataset.py 裡按 load 分開算，因為不同負載下的正常溫度不同）
    """
    x = temp_series_1hz
    n = len(x)
    delta_from_baseline = x - baseline_temp

    # 升溫速度：每 10 秒（示範用短一點的窗，因為錄音本身只有 60~120 秒）的溫度變化
    rise_window = min(10, n - 1) if n > 1 else 1
    rise_rate = float(np.max(x[rise_window:] - x[:-rise_window])) if n > rise_window else 0.0

    over_baseline_ratio = float(np.mean(delta_from_baseline > 2.0))  # 高於基準 2°C 以上的時間比例
    slope = float(np.polyfit(np.arange(n), x, 1)[0]) if n >= 2 else 0.0  # 線性趨勢斜率 (°C/sec)

    return {
        "temp_mean": float(np.mean(x)),
        "temp_max": float(np.max(x)),
        "temp_delta_from_baseline_mean": float(np.mean(delta_from_baseline)),
        "temp_delta_from_baseline_max": float(np.max(delta_from_baseline)),
        "temp_rise_rate_per10s": rise_rate,
        "temp_over_baseline_ratio": over_baseline_ratio,
        "temp_trend_slope": slope,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    fs = 50000
    seconds = 60
    raw = 40 + 0.05 * np.arange(fs * seconds) / fs + 0.02 * rng.standard_normal(fs * seconds)
    resampled = resample_to_seconds(raw, fs)
    print("resampled length:", len(resampled))
    feats = extract_temperature_features(resampled, baseline_temp=40.0)
    print({k: round(v, 4) for k, v in feats.items()})
