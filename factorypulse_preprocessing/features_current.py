"""
電流特徵萃取。

⚠️ 重要決定：改用「單相（U 相）」特徵，不用三相特徵。

原因（實測後的結論，不是偷懶）：
1. 原始資料集裡 9 個 BPFO 檔案（3 個負載 × 3 個嚴重度）只錄到 U 相，
   V/W 相的長度是 0。如果堅持要三相，這 9 個檔案就得整個丟掉，
   等於電流模型完全看不到「軸承外圈故障」這個類別——那是最不能少的類別之一。
2. 三相不平衡度（current_unbalance_ratio）在實際資料上算出來是：
       BPFI 0.2367 / Misalign 0.2374 / Normal 0.2364 / Unbalance 0.2375
   各類別幾乎一模一樣，完全沒有鑑別力。也就是說，為了保留一個沒有用的特徵，
   去犧牲一整個故障類別，並不划算。

所以最終方案：所有檔案一律只用 U 相，特徵欄位在 45 個檔案之間保持一致。
（三相版本的函式保留在下方 extract_three_phase_features，之後若換資料集用得到。）
"""

import numpy as np


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def _harmonic_energy(x: np.ndarray, fs: float, fundamental: float = 60.0, n_harmonics: int = 5) -> float:
    """基頻的 2~6 次諧波能量總和。基頻本身是正常運轉電流，不列入異常特徵。"""
    n = len(x)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    energy = 0.0
    for h in range(2, n_harmonics + 2):
        target = fundamental * h
        idx = np.argmin(np.abs(freqs - target))
        energy += spectrum[idx] ** 2
    return float(energy)


def _envelope(x: np.ndarray) -> np.ndarray:
    """用 FFT 手刻 Hilbert transform 取包絡（避免依賴 scipy.signal）。"""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


def _sideband_ratio(x: np.ndarray, fs: float, line_freq: float = 60.0,
                    band: float = 30.0, guard: float = 5.0) -> float:
    """邊帶能量「相對於基頻」的比值 —— 調變程度指標。

    ⚠️ 這裡原本寫的是「邊帶絕對能量」，實測發現它跟 current_rms 的相關係數高達
    0.9996，等於只是在測「訊號幅度」而不是「調變」。原因是 1 秒窗的頻率解析度
    只有 1Hz，60Hz 主峰經過 Hanning 窗後會外洩到整個 ±30Hz 範圍，
    絕對能量因此被基頻幅度主導。

    改成除以基頻能量之後，這個特徵才真正代表「邊帶相對於載波有多強」，
    也就是機械故障造成的調變深度，跟訊號整體大小無關。
    guard 也從 ±2Hz 放寬到 ±5Hz，進一步避開主峰外洩。
    """
    n = len(x)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n)))
    power = spectrum ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    fundamental_mask = (freqs > line_freq - guard) & (freqs < line_freq + guard)
    sideband_mask = (freqs > line_freq - band) & (freqs < line_freq + band) & ~fundamental_mask

    fundamental_power = float(np.sum(power[fundamental_mask]))
    sideband_power = float(np.sum(power[sideband_mask]))
    if fundamental_power <= 1e-20:
        return 0.0
    return sideband_power / fundamental_power


def extract_current_features(x: np.ndarray, fs: float, line_freq: float = 60.0) -> dict:
    """單相電流特徵（本專案實際使用的版本）。

    x: 一段固定長度的單相電流訊號 (window_len,)
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    rms = _rms(x)
    peak = float(np.max(np.abs(x)))

    env = _envelope(x)
    env_ac = env - np.mean(env)
    n = len(x)
    env_spec = np.abs(np.fft.rfft(env_ac * np.hanning(n)))
    env_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    valid = env_freqs > 1.0
    env_dom_freq = float(env_freqs[valid][np.argmax(env_spec[valid])]) if np.any(valid) else 0.0

    std = float(np.std(x))
    kurt = float(((x - x.mean()) ** 4).mean() / (std ** 4 + 1e-12)) if std > 0 else 0.0

    # 註：current_std 與 current_mean_abs 已移除。
    # 對去直流/交流訊號來說 std 與 rms 在數學上相等（實測相關係數 1.0），
    # mean_abs 對正弦波也只是 rms 的固定倍數，三者是同一個「幅度」資訊。
    # 保留 rms 一個即可，避免特徵重要度被稀釋。
    return {
        "current_rms": rms,
        "current_peak": peak,
        "current_crest_factor": peak / rms if rms > 1e-12 else 0.0,
        "current_kurtosis": kurt,
        "current_harmonic_energy": _harmonic_energy(x, fs, line_freq),
        "current_sideband_ratio": _sideband_ratio(x, fs, line_freq),
        # 包絡標準差除以 rms -> 相對調變深度，同樣避免變成另一個幅度特徵
        "current_envelope_cv": float(np.std(env)) / rms if rms > 1e-12 else 0.0,
        "current_envelope_dom_freq": env_dom_freq,
    }


def extract_three_phase_features(u, v, w, fs: float, line_freq: float = 60.0) -> dict:
    """三相版本 —— 本專案目前沒有使用（因為 BPFO 檔案缺 V/W 相）。
    保留給之後換資料集、或補到完整三相資料時使用。
    """
    rms_u, rms_v, rms_w = _rms(u), _rms(v), _rms(w)
    rms_mean = (rms_u + rms_v + rms_w) / 3
    unbalance_ratio = (max(rms_u, rms_v, rms_w) - min(rms_u, rms_v, rms_w)) / (rms_mean + 1e-12)

    feats = {f"phaseU_{k}": val for k, val in extract_current_features(u, fs, line_freq).items()}
    feats.update({
        "current_rms_U": rms_u,
        "current_rms_V": rms_v,
        "current_rms_W": rms_w,
        "current_rms_mean": rms_mean,
        "current_unbalance_ratio": unbalance_ratio,
    })
    return feats


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    fs = 25608.19
    t = np.arange(int(fs)) / fs
    base = np.sin(2 * np.pi * 60 * t)

    normal = base + 0.01 * rng.standard_normal(len(t))
    # 模擬機械故障造成的邊帶調變（30Hz 調變）
    faulty = base * (1 + 0.15 * np.sin(2 * np.pi * 30 * t)) + 0.01 * rng.standard_normal(len(t))

    for name, sig in [("normal", normal), ("faulty(有邊帶調變)", faulty)]:
        f = extract_current_features(sig, fs)
        print(f"{name:22s} sideband={f['current_sideband_energy']:.3e} "
              f"env_std={f['current_envelope_std']:.5f} "
              f"env_dom={f['current_envelope_dom_freq']:.1f}Hz")
