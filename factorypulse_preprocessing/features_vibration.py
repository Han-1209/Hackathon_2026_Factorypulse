"""
振動特徵萃取。

兩種用途分開提供：
  1) prepare_cnn_input(): 給 1D CNN 用的「乾淨波形」(去直流、標準化)，模型自己學特徵。
  2) extract_classical_features(): 給 Random Forest / XGBoost 當 baseline 或做可解釋分析用的
     手工特徵（RMS、峰值因子、頻帶能量...），也方便你在還沒訓練好 CNN 前，
     先用簡單模型跑出一個「能動的版本」。
"""

import numpy as np


def _remove_dc(window: np.ndarray) -> np.ndarray:
    return window - np.mean(window, axis=0, keepdims=True)


def prepare_cnn_input(windows: np.ndarray) -> np.ndarray:
    """windows: (n_windows, window_len, n_channels)
    回傳: 去直流 + 每個窗獨立做 z-score 標準化後的陣列，shape 不變。
    """
    x = windows.astype(np.float32)
    x = x - np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std[std == 0] = 1e-8
    x = x / std
    return x


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask] ** 2))


def extract_classical_features(window: np.ndarray, fs: float) -> dict:
    """window: (window_len, n_channels) 單一一段窗（單一通道也可以先 reshape 成 (L,1)）
    回傳每個通道的統計 + 頻域特徵，key 用 ch0_xxx / ch1_xxx 命名。
    """
    window = _remove_dc(window)
    n_channels = window.shape[1] if window.ndim == 2 else 1
    if window.ndim == 1:
        window = window[:, None]

    feats = {}
    for ch in range(n_channels):
        x = window[:, ch]
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        crest_factor = peak / rms if rms > 1e-12 else 0.0
        std = float(np.std(x))
        kurt = float(((x - np.mean(x)) ** 4).mean() / (std ** 4 + 1e-12)) if std > 0 else 0.0
        skew = float(((x - np.mean(x)) ** 3).mean() / (std ** 3 + 1e-12)) if std > 0 else 0.0

        # FFT 頻域特徵
        n = len(x)
        window_fn = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(x * window_fn))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        low_band = _band_energy(spectrum, freqs, 0, 500)
        mid_band = _band_energy(spectrum, freqs, 500, 2000)
        high_band = _band_energy(spectrum, freqs, 2000, fs / 2)
        dominant_freq = float(freqs[np.argmax(spectrum)]) if len(spectrum) else 0.0

        prefix = f"ch{ch}_"
        feats.update(
            {
                prefix + "rms": rms,
                prefix + "peak": peak,
                prefix + "crest_factor": crest_factor,
                prefix + "std": std,
                prefix + "kurtosis": kurt,
                prefix + "skewness": skew,
                prefix + "band_energy_low": low_band,
                prefix + "band_energy_mid": mid_band,
                prefix + "band_energy_high": high_band,
                prefix + "dominant_freq": dominant_freq,
            }
        )
    return feats


if __name__ == "__main__":
    # 自我測試：用假訊號確認流程跑得動、數值合理
    rng = np.random.default_rng(0)
    fs = 25600
    t = np.arange(int(fs * 1.0)) / fs
    normal = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.02 * rng.standard_normal(len(t))
    faulty = normal + 0.5 * (np.sin(2 * np.pi * 1200 * t) * (np.sin(2 * np.pi * 20 * t) > 0.8))

    for name, sig in [("normal", normal), ("faulty(周期性衝擊)", faulty)]:
        feats = extract_classical_features(sig[:, None], fs)
        print(name, {k: round(v, 4) for k, v in feats.items()})
