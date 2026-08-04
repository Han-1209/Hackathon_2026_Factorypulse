"""
振動特徵萃取 v2 —— 階次域 (order domain) + 包絡解調
====================================================

為什麼要重寫 v1
---------------
v1 的頻域特徵只有三個粗頻帶（0-500 / 500-2000 / 2000-Nyquist）加一個
dominant_freq。用實際跑出來的特徵檔驗證後，發現一個致命問題：

    condition   ch0_dominant_freq 中位數(Hz)   低頻帶能量佔比
                0Nm    2Nm    4Nm              0Nm      2Nm      4Nm
    Normal      1453   1449   1445             0.0015   0.0006   0.0003
    Unbalance   1453   1449   1445             0.0017   0.0006   0.0004

    -> Normal 與 Unbalance 在 v1 的頻域特徵上「數值完全一樣」。

原因很直接：轉軸轉速 3010 RPM ≈ 50 Hz。
  - 不平衡 (Unbalance) 的特徵是 1x = 50 Hz 振幅上升
  - 不對心 (Misalign) 的特徵是 2x = 100 Hz 振幅上升（且 2x/1x 比值變大）
這兩個資訊全部被壓進同一個 0-500 Hz 的頻帶，而那個頻帶又只佔總能量的 0.1%
（總能量被 1.4 kHz 的結構共振主導）。等於把唯一能分辨這兩類的物理量
在特徵萃取階段就丟掉了。

模型還是跑出 0.96，是因為 train/val/test 是「同一段錄音切前中後」，
模型背的是這段錄音的雜訊指紋。跨負載一測就掉到 0.48，
而且 4Nm 的 Normal 有 100% 被判成 Unbalance —— 完全符合上面的診斷。

v2 做了什麼
-----------
1. 階次域諧波特徵：自動偵測轉頻 fr，量出 1x~5x 的振幅（用總 RMS 正規化）
   以及 2x/1x、3x/1x 比值。這是 Unbalance / Misalign / Normal 的教科書判準，
   而且因為是「比值」，跨負載時比絕對振幅穩定得多。
2. 包絡解調 (envelope analysis)：對高頻共振帶做帶通 -> Hilbert 取包絡 ->
   對包絡再做 FFT。軸承局部缺陷的衝擊會在包絡譜上出現 BPFO / BPFI 的譜線。
   這是軸承診斷的標準做法，也是 SHAP 出來之後能講成人話的關鍵
   （「包絡譜在 BPFO=179.8 Hz 有明顯譜線」比「band_energy_mid 很重要」有說服力）。
3. 細分頻帶佔比（log 間距）+ 頻譜熵：描述頻譜「形狀」而非「大小」，
   對負載變化不敏感。

只用 numpy，沒有 scipy 依賴（Hilbert 直接用 FFT 實作），所以在哪都跑得動。

用法
----
    from features_vibration_v2 import extract_features_v2
    feats = extract_features_v2(window, fs=25600, rpm=3010)

rpm 傳 None 的話會自動從頻譜偵測（在 SHAFT_FREQ_SEARCH 範圍內找最大峰）。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 軸承特徵頻率係數（單位：轉頻的倍數 / order）
#
# ⚠️ 這組數字對應 6205 型深溝球軸承（9 顆滾珠）。KAIST 這批資料的軸承型號
#    請務必回論文/Mendeley 說明頁確認後改這裡。填錯不會報錯，但包絡譜的
#    特徵頻帶會對到錯誤位置，等於白做。
#
#    確認方式：跑 verify_bearing_orders()，它會用 BPFO/BPFI 的實際資料
#    去掃描包絡譜，印出真正出現譜線的階次，跟這裡的設定值比對。
# --------------------------------------------------------------------------
BEARING_ORDERS = {
    "BPFO": 3.66,    # 外環通過頻率 —— verify_v2.py 實測 183.4Hz / 50.15Hz = 3.66
    "BPFI": 5.36,    # 內環通過頻率 —— verify_v2.py 實測 269.1Hz / 50.15Hz = 5.36
}

# BSF（滾動體）與 FTF（保持器）刻意「不」放進 BEARING_ORDERS，原因有兩個：
#
# 1. 這個資料集沒有這兩種故障類別（只有 Normal/BPFI/BPFO/Misalign/Unbalance），
#    所以它們不可能是真正的診斷證據。
#
# 2. 更嚴重的是它們會「冒名頂替」。在 fr=50.15Hz 下實測到三組重疊：
#        BSF_sb_hi 搜尋 [175.7, 183.4]  <-- BPFO 真實譜線就在 183.4
#        BSF       搜尋 [125.5, 133.3]  <-- BPFO 下邊帶就在 133.2
#        BSF_h2    搜尋 [251.0, 266.5]  <-- 與 BPFI 268.8 的視窗相接
#    第一版跑出來的特徵重要度前三名是 env_BSF_sb / env_FTF_sb / env_BSF，
#    它們量到的其實全是外環故障的能量，只是掛著滾動體的名字。
#    模型準確率不受影響（資訊是對的），但 SHAP 會產生錯誤的物理解釋，
#    LLM 再據此寫出一段「滾動體故障」的說明 —— 這是可解釋性產品最不能犯的錯。
#
# 未來若要納入有滾動體故障的資料集（例如變轉速那組有 ball 類別），
# 把它們加回 BEARING_ORDERS，並且務必先跑 check_order_overlaps() 確認不重疊。
OPTIONAL_ORDERS = {
    "BSF": 2.58,     # 滾動體自轉頻率
    "FTF": 0.407,    # 保持器頻率 = BPFO / 滾珠數
}

# 實測值的一致性驗證（這組數字彼此不是獨立的，能互相驗算）：
#   BPFO + BPFI = 滾珠數 Nb  ->  3.66 + 5.36 = 9.02 ≈ 9 顆滾珠 ✓
#   由 BPFO = Nb/2 * (1 - d/D)  ->  d/D = 0.187
#   回代 BPFI = Nb/2 * (1 + d/D) = 4.5 * 1.187 = 5.34  ≈ 實測 5.36 ✓
#   FTF = BPFO / Nb = 0.407 ,  BSF = D/(2d) * (1-(d/D)^2) = 2.58
# 三個獨立來源互相吻合，可以確定這組係數是對的（滾珠數 9 也符合 6205 型）。
BALL_COUNT = 9

SHAFT_FREQ_SEARCH = (20.0, 90.0)   # 自動偵測轉頻的搜尋範圍 (Hz)
ENVELOPE_BAND = (2000.0, 6000.0)   # 包絡解調用的共振帶 (Hz)
N_HARMONICS = 5                    # 抓到 5x
LOG_BANDS = [(10, 50), (50, 150), (150, 400), (400, 1000),
             (1000, 3000), (3000, 8000), (8000, 12800)]


# ============================================================ 基本工具
def _rfft(x: np.ndarray, fs: float):
    """回傳 (freqs, 單邊振幅譜)。用 Hann 窗並做振幅補償，讓譜線高度≈真實振幅。"""
    n = len(x)
    w = np.hanning(n)
    # Hann 窗的相干增益 0.5 -> 乘 2 補回來；rfft 單邊再乘 2
    amp = np.abs(np.fft.rfft(x * w)) * (4.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amp


def _peak_near(freqs, amp, f0, tol_hz):
    """在 f0 ± tol_hz 內取最大譜線高度。tol 是為了容忍轉速微幅飄移。"""
    if f0 <= 0 or f0 >= freqs[-1]:
        return 0.0
    m = (freqs >= f0 - tol_hz) & (freqs <= f0 + tol_hz)
    return float(amp[m].max()) if m.any() else 0.0


def check_order_overlaps(fr: float, orders: dict | None = None, verbose: bool = True):
    """檢查各特徵頻率的搜尋視窗有沒有互相重疊。

    重疊 = 兩個不同名字的特徵在量同一條譜線。準確率通常看不出差別
    （資訊還在），但 SHAP 會把功勞歸給錯的物理現象，
    導致解釋層講出資料裡根本不存在的故障類型。可解釋性產品必須擋掉這個。

    改動 BEARING_ORDERS、換軸承、或換到不同轉速的資料集時，都要重跑一次。
    回傳重疊配對的清單，沒有重疊就是空 list。
    """
    orders = orders or BEARING_ORDERS
    wins = []
    for name, o in orders.items():
        f0 = fr * o
        tol = max(3.0, f0 * 0.03)
        wins.append((name, f0, tol))
        wins.append((f"{name}_h2", 2 * f0, tol * 2))
        for sign, lbl in [(-1, "sb_lo"), (1, "sb_hi")]:
            f = f0 + sign * fr
            if f > 0:
                wins.append((f"{name}_{lbl}", f, tol))

    clashes = []
    for i in range(len(wins)):
        for j in range(i + 1, len(wins)):
            a, fa, ta = wins[i]
            b, fb, tb = wins[j]
            if a.split("_")[0] == b.split("_")[0]:
                continue          # 同一個特徵頻率的自家諧波/邊帶，重疊無所謂
            if abs(fa - fb) <= ta + tb:
                clashes.append((a, fa, b, fb))

    if verbose:
        if clashes:
            print(f"⚠️ 偵測到 {len(clashes)} 組特徵頻率視窗重疊（fr={fr:.2f} Hz）：")
            for a, fa, b, fb in clashes:
                print(f"     {a} ({fa:.1f} Hz)  <->  {b} ({fb:.1f} Hz)")
            print("   這代表兩個特徵在量同一條譜線，SHAP 會歸因到錯誤的物理現象。")
        else:
            print(f"✓ 特徵頻率視窗無重疊（fr={fr:.2f} Hz），SHAP 歸因不會張冠李戴。")
    return clashes


def detect_shaft_freq(x: np.ndarray, fs: float, search=SHAFT_FREQ_SEARCH) -> float:
    """從頻譜自動找轉頻：在搜尋範圍內取最大峰。

    不平衡會讓 1x 更明顯，正常機台 1x 也一定存在（沒有絕對平衡的轉子），
    所以這個偵測在五種狀態下都可用。
    """
    freqs, amp = _rfft(x, fs)
    m = (freqs >= search[0]) & (freqs <= search[1])
    if not m.any():
        return 0.0
    return float(freqs[m][amp[m].argmax()])


def _bandpass_fft(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """用 FFT 遮罩做帶通（不需要 scipy）。"""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    X[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(X, n=len(x))


def _envelope(x: np.ndarray) -> np.ndarray:
    """Hilbert 包絡，用 FFT 實作（等價於 scipy.signal.hilbert 取 abs）。"""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


# ============================================================ 特徵群
def order_features(x: np.ndarray, fs: float, fr: float, prefix: str = "") -> dict:
    """階次域諧波特徵：1x~5x 振幅（正規化）與諧波比值。

    正規化方式是「除以該窗的總 RMS」，所以整體振幅隨負載一起放大時，
    這些特徵不會跟著漂 —— 這是跨負載能撐住的關鍵。
    """
    freqs, amp = _rfft(x, fs)
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    tol = max(2.0, fr * 0.05)

    feats = {}
    harm = []
    for k in range(1, N_HARMONICS + 1):
        a = _peak_near(freqs, amp, fr * k, tol)
        harm.append(a)
        feats[f"{prefix}order_{k}x"] = a / rms
        # 絕對振幅也留一份（有量綱，跨負載實驗會被 scale_invariant_columns 排除）。
        # 保留的理由：正規化版本會被整體能量拉扯 —— 例如 BPFO 檔案的 RMS 是
        # Normal 的 9 倍，即使 1x 絕對值沒變，order_1x 也會被壓到只剩 1/9。
        # 兩種都給模型，讓它自己決定用哪個。
        feats[f"{prefix}order_{k}x_abs"] = a

    h1 = harm[0] + 1e-12
    feats[f"{prefix}ratio_2x_1x"] = harm[1] / h1          # 不對心的核心判準
    feats[f"{prefix}ratio_3x_1x"] = harm[2] / h1
    feats[f"{prefix}harmonic_sum_ratio"] = sum(harm[1:]) / h1
    # 1x 佔整體能量的比重：不平衡會讓這個值明顯上升
    feats[f"{prefix}order_1x_dominance"] = harm[0] / (float(amp.sum()) + 1e-12)
    feats[f"{prefix}shaft_freq"] = fr
    return feats


def envelope_features(x: np.ndarray, fs: float, fr: float,
                      band=ENVELOPE_BAND, prefix: str = "") -> dict:
    """包絡譜特徵：軸承局部缺陷診斷的標準做法。

    流程：帶通到共振帶 -> Hilbert 取包絡 -> 對包絡做 FFT ->
          在 BPFO/BPFI/BSF/FTF 及其 2 次諧波處取譜線高度，
          再除以包絡譜的平均值（得到類似訊噪比的無量綱量）。
    """
    hi = min(band[1], fs / 2 - 1)
    xb = _bandpass_fft(x, fs, band[0], hi)
    env = _envelope(xb)
    env = env - env.mean()
    efreqs, eamp = _rfft(env, fs)

    # 只看 1000 Hz 以下（軸承特徵頻率與其諧波都在這個範圍）
    m = efreqs <= 1000.0
    efreqs, eamp = efreqs[m], eamp[m]
    noise = float(eamp.mean()) + 1e-12

    feats = {}
    for name, order in BEARING_ORDERS.items():
        f0 = fr * order
        tol = max(3.0, f0 * 0.03)
        a1 = _peak_near(efreqs, eamp, f0, tol)
        a2 = _peak_near(efreqs, eamp, 2 * f0, tol * 2)
        feats[f"{prefix}env_{name}"] = a1 / noise
        feats[f"{prefix}env_{name}_h2"] = a2 / noise

        # ±1x 轉頻邊帶。內環故障的缺陷會隨轉軸進出承載區，
        # 造成振幅被轉頻調變 -> 包絡譜在 BPFI±1x 出現邊帶。
        # 外環故障位置固定不動，理論上邊帶弱很多。
        # verify_v2 實測支持這點：BPFI 檔案在 4.36 階(=5.36-1) 的 SNR 高達 228.7，
        # 幾乎與 BPFI 主譜線 237.9 同等強度 —— 這是內外環最乾淨的判別依據，
        # 而且它是「比值」，跨負載時比絕對振幅穩定。
        sb_lo = _peak_near(efreqs, eamp, f0 - fr, tol)
        sb_hi = _peak_near(efreqs, eamp, f0 + fr, tol)
        feats[f"{prefix}env_{name}_sb"] = (sb_lo + sb_hi) / (2 * noise)
        feats[f"{prefix}env_{name}_sb_ratio"] = (sb_lo + sb_hi) / (2 * a1 + 1e-12)

    # 包絡譜最強譜線落在第幾階 —— 不依賴軸承型號設定是否正確的保險特徵
    if len(eamp) > 2 and fr > 0:
        k = int(np.argmax(eamp[1:])) + 1
        feats[f"{prefix}env_peak_order"] = float(efreqs[k] / fr)
        feats[f"{prefix}env_peak_snr"] = float(eamp[k] / noise)
    else:
        feats[f"{prefix}env_peak_order"] = 0.0
        feats[f"{prefix}env_peak_snr"] = 0.0

    # 包絡本身的統計：衝擊性愈強，包絡的變異愈大
    e = _envelope(xb)
    feats[f"{prefix}env_cv"] = float(e.std() / (e.mean() + 1e-12))
    feats[f"{prefix}env_kurtosis"] = float(((e - e.mean()) ** 4).mean()
                                           / (e.std() ** 4 + 1e-12))
    return feats


def spectrum_shape_features(x: np.ndarray, fs: float, prefix: str = "") -> dict:
    """頻譜「形狀」特徵：全部是佔比或無量綱量，不隨整體振幅縮放。"""
    freqs, amp = _rfft(x, fs)
    p = amp ** 2
    tot = float(p.sum()) + 1e-12

    feats = {}
    for lo, hi in LOG_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        feats[f"{prefix}bandfrac_{lo}_{hi}"] = float(p[m].sum()) / tot

    pn = p / tot
    feats[f"{prefix}spec_centroid"] = float((freqs * pn).sum())
    feats[f"{prefix}spec_entropy"] = float(-(pn * np.log(pn + 1e-20)).sum())
    feats[f"{prefix}spec_flatness"] = float(
        np.exp(np.log(p + 1e-20).mean()) / (p.mean() + 1e-20))
    return feats


def time_features(x: np.ndarray, prefix: str = "") -> dict:
    """時域特徵。刻意只留無量綱的那幾個 + rms（rms 保留是因為在
    「同負載」情境仍然有用，跨負載實驗時可以用 include_scale=False 排除）。"""
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x)) + 1e-12
    return {
        f"{prefix}rms": rms,                                   # 有量綱
        f"{prefix}crest_factor": peak / (rms + 1e-12),
        f"{prefix}kurtosis": float(((x - x.mean()) ** 4).mean() / std ** 4),
        f"{prefix}skewness": float(((x - x.mean()) ** 3).mean() / std ** 3),
        f"{prefix}shape_factor": rms / (np.mean(np.abs(x)) + 1e-12),
        f"{prefix}impulse_factor": peak / (np.mean(np.abs(x)) + 1e-12),
    }


# ============================================================ 對外主函式
# 跨負載實驗要排除的有量綱特徵（數值會隨整體振幅一起放大／縮小）
SCALE_DEPENDENT_SUFFIX = ("_rms", "_abs")


def extract_features_v2(window: np.ndarray, fs: float, rpm: float | None = 3010,
                        do_envelope: bool = True) -> dict:
    """window: (window_len,) 或 (window_len, n_channels)

    rpm=None 時自動從第 0 通道偵測轉頻。回傳 dict，key 以 ch{i}_ 開頭。
    """
    if window.ndim == 1:
        window = window[:, None]
    window = window - window.mean(axis=0, keepdims=True)

    if rpm is not None and rpm > 0:
        fr = rpm / 60.0
    else:
        fr = detect_shaft_freq(window[:, 0], fs)

    feats = {}
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float64)
        p = f"ch{ch}_"
        feats.update(time_features(x, p))
        feats.update(spectrum_shape_features(x, fs, p))
        feats.update(order_features(x, fs, fr, p))
        if do_envelope:
            feats.update(envelope_features(x, fs, fr, prefix=p))

    # 跨通道比值：反映振動在機台上的空間分布，對不對心特別有用
    for a, b in [(0, 1), (2, 3), (0, 2)]:
        ka, kb = f"ch{a}_rms", f"ch{b}_rms"
        if ka in feats and kb in feats:
            feats[f"rms_ratio_{a}{b}"] = feats[ka] / (feats[kb] + 1e-12)
    return feats


def scale_invariant_columns(columns) -> list:
    """挑出跨負載實驗可以安全使用的無量綱特徵欄位。"""
    return [c for c in columns
            if not any(c.endswith(s) for s in SCALE_DEPENDENT_SUFFIX)]


# ============================================================ 自我測試
def _synth(fs, sec, fr, kind, seed=0):
    """合成訊號：驗證特徵方向是否符合物理直覺。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * sec)) / fs
    # 所有狀態共有的：結構共振底噪 + 白噪
    x = 0.3 * np.sin(2 * np.pi * 1450 * t) + 0.05 * rng.standard_normal(len(t))
    x += 0.10 * np.sin(2 * np.pi * fr * t)          # 基本 1x
    if kind == "unbalance":
        x += 0.60 * np.sin(2 * np.pi * fr * t)      # 1x 大幅上升
    elif kind == "misalign":
        x += 0.15 * np.sin(2 * np.pi * fr * t)
        x += 0.55 * np.sin(2 * np.pi * 2 * fr * t)  # 2x 主導
    elif kind in ("BPFO", "BPFI"):
        # 週期性衝擊激發 3 kHz 共振
        f_def = fr * BEARING_ORDERS[kind]
        impulses = np.zeros_like(t)
        for k in range(int(sec * f_def)):
            i = int(k / f_def * fs)
            if i < len(t):
                impulses[i] = 1.0
        decay = np.exp(-np.arange(int(0.002 * fs)) / (0.0004 * fs))
        carrier = decay * np.sin(2 * np.pi * 3000 * np.arange(len(decay)) / fs)
        x += 2.0 * np.convolve(impulses, carrier, mode="same")
    return x


if __name__ == "__main__":
    fs, fr = 25600, 3010 / 60
    print(f"合成訊號自我測試  fs={fs}  fr={fr:.2f} Hz\n")

    rows = {}
    for kind in ["normal", "unbalance", "misalign", "BPFO", "BPFI"]:
        sig = _synth(fs, 1.0, fr, kind)
        f = extract_features_v2(sig[:, None], fs, rpm=3010)
        rows[kind] = f

    import pandas as pd
    df = pd.DataFrame(rows).T
    show = ["ch0_order_1x", "ch0_order_2x", "ch0_ratio_2x_1x",
            "ch0_order_1x_dominance", "ch0_env_BPFO", "ch0_env_BPFI",
            "ch0_env_peak_order", "ch0_kurtosis", "ch0_crest_factor"]
    pd.set_option("display.width", 200)
    print(df[show].to_string(float_format=lambda v: f"{v:9.3f}"))

    print("\n檢查（這幾條若不成立代表實作有誤）：")
    ok = True

    def chk(name, cond):
        global ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    chk("unbalance 的 1x 明顯高於 normal",
        df.loc["unbalance", "ch0_order_1x"] > 2 * df.loc["normal", "ch0_order_1x"])
    chk("misalign 的 2x/1x 比值高於 unbalance",
        df.loc["misalign", "ch0_ratio_2x_1x"] > df.loc["unbalance", "ch0_ratio_2x_1x"])
    chk("BPFO 訊號在包絡譜的 BPFO 位置最突出",
        df.loc["BPFO", "ch0_env_BPFO"] > df.loc["BPFO", "ch0_env_BPFI"])
    chk("BPFI 訊號在包絡譜的 BPFI 位置最突出",
        df.loc["BPFI", "ch0_env_BPFI"] > df.loc["BPFI", "ch0_env_BPFO"])
    chk("BPFO/BPFI 的 kurtosis 高於 normal（衝擊性）",
        min(df.loc["BPFO", "ch0_kurtosis"], df.loc["BPFI", "ch0_kurtosis"])
        > df.loc["normal", "ch0_kurtosis"])
    chk("normal 與 unbalance 在 v2 特徵上可分（v1 的致命問題）",
        abs(df.loc["normal", "ch0_order_1x_dominance"]
            - df.loc["unbalance", "ch0_order_1x_dominance"]) > 1e-3)

    print("\n全部通過" if ok else "\n有檢查未通過，請看上面")
