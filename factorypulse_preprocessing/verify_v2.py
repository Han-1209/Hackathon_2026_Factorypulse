"""
在重跑整條前處理之前，先用少數幾個檔案驗證 v2 特徵在「真實資料」上成立。
====================================================================

為什麼要有這一步：
    全量重跑 45 個檔案 × 300 秒要花很久。如果軸承階次係數填錯、
    或轉頻不是 3010 RPM，跑完才發現等於整段時間白花。
    這支腳本只讀 5 個檔案的前 20 秒，幾分鐘內就能回答三個問題：

      Q1  轉頻到底是多少？(config 假設 3010 RPM = 50.17 Hz)
      Q2  BPFO/BPFI 檔案的包絡譜，譜線實際落在第幾階？
          -> 直接驗證 features_vibration_v2.BEARING_ORDERS 填得對不對
      Q3  v2 的 1x/2x 特徵，真的能把 Normal 和 Unbalance 分開嗎？
          (這是 v1 完全做不到的事，是整個修正的核心賭注)

執行：
    python verify_v2.py

需要 scipy（讀 .mat）。沒裝：pip install scipy
"""

import sys
from pathlib import Path

import numpy as np

import config
from io_utils import load_vibration_mat
import features_vibration_v2 as v2

SECONDS = 20          # 每個檔只讀前 20 秒就夠了
WINDOW_SEC = 1.0
CHANNEL = 0           # x_direction_housing_A

# 挑最有代表性的 5 個檔案，全部固定在 0Nm 排除負載干擾
FILES = {
    "Normal":    "0Nm_Normal.mat",
    "Unbalance": "0Nm_Unbalance_3318mg.mat",   # 最嚴重那級，訊號最明顯
    "Misalign":  "0Nm_Misalign_05.mat",
    "BPFO":      "0Nm_BPFO_30.mat",
    "BPFI":      "0Nm_BPFI_30.mat",
}


def cohens_d(a, b):
    """兩組樣本的標準化差異。|d|>0.8 算大效果，>2 就是肉眼可分。"""
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                / (na + nb - 2) + 1e-20)
    return float((np.mean(a) - np.mean(b)) / (s + 1e-20))


def envelope_top_orders(x, fs, fr, n=6):
    """回傳包絡譜上最突出的前 n 個階次（頻率 / 轉頻），用來反推軸承係數。"""
    xb = v2._bandpass_fft(x, fs, *v2.ENVELOPE_BAND)
    env = v2._envelope(xb)
    env = env - env.mean()
    f, a = v2._rfft(env, fs)
    m = (f > fr * 0.5) & (f <= 1000)
    f, a = f[m], a[m]
    # 只留局部極大值，避免同一個峰旁邊的點重複入榜
    peaks = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] > a[i + 1]]
    peaks.sort(key=lambda i: -a[i])
    noise = a.mean() + 1e-20
    out, used = [], []
    for i in peaks:
        if any(abs(f[i] - u) < fr * 0.3 for u in used):
            continue
        used.append(f[i])
        out.append((float(f[i]), float(f[i] / fr), float(a[i] / noise)))
        if len(out) >= n:
            break
    return out


def main():
    vib_dir = config.VIBRATION_DIR
    if not vib_dir.exists():
        sys.exit(f"找不到振動資料夾：{vib_dir}\n請檢查 config.DATA_ROOT")

    fs = config.VIBRATION_FS
    wlen = int(WINDOW_SEC * fs)
    sigs, feats = {}, {}

    print("=" * 70)
    print("讀取檔案（每個只取前 %d 秒，通道 ch%d）" % (SECONDS, CHANNEL))
    print("=" * 70)
    for cond, fname in FILES.items():
        p = vib_dir / fname
        if not p.exists():
            print(f"  ✗ {cond:10s} 找不到 {fname}，跳過")
            continue
        data = load_vibration_mat(p)
        x = data[: SECONDS * fs, CHANNEL].astype(np.float64)
        x = x - x.mean()
        sigs[cond] = x
        print(f"  ✓ {cond:10s} {fname:28s} {len(x):>9,d} 點  "
              f"RMS={np.sqrt((x**2).mean()):.4f}")

    if not sigs:
        sys.exit("一個檔案都沒讀到，先確認 config.DATA_ROOT")

    # ---------------------------------------------------------------- Q1
    print("\n" + "=" * 70)
    print("Q1  轉頻偵測（config 假設 3010 RPM = %.2f Hz）" % (3010 / 60))
    print("=" * 70)
    detected = []
    for cond, x in sigs.items():
        fr = v2.detect_shaft_freq(x, fs)
        detected.append(fr)
        print(f"  {cond:10s} fr = {fr:6.2f} Hz  ({fr*60:7.1f} RPM)")
    fr_med = float(np.median(detected))
    print(f"\n  中位數 fr = {fr_med:.2f} Hz ({fr_med*60:.0f} RPM)")
    if abs(fr_med - 3010 / 60) > 2.0:
        print(f"  ⚠️ 與假設的 3010 RPM 差距超過 2 Hz。"
              f"請把 build_dataset 的 rpm 參數改成 {fr_med*60:.0f}，"
              f"或直接傳 rpm=None 讓它每個窗自動偵測。")
    else:
        print("  ✓ 與 3010 RPM 相符，可以直接用固定值。")

    # ---------------------------------------------------------------- Q2
    print("\n" + "=" * 70)
    print("Q2  包絡譜實際譜線落在第幾階？（驗證 BEARING_ORDERS）")
    print("=" * 70)
    print(f"  目前設定：" + "  ".join(f"{k}={v}" for k, v in v2.BEARING_ORDERS.items()))
    for cond in ["Normal", "BPFO", "BPFI"]:
        if cond not in sigs:
            continue
        print(f"\n  [{cond}] 包絡譜前 6 個峰（頻率 / 階次 / 相對噪底）")
        for f_hz, order, snr in envelope_top_orders(sigs[cond], fs, fr_med):
            hit = ""
            for name, o in v2.BEARING_ORDERS.items():
                if abs(order - o) < 0.15:
                    hit = f"  <-- 命中 {name}"
                elif abs(order - 2 * o) < 0.3:
                    hit = f"  <-- 命中 {name} 的 2 次諧波"
            print(f"      {f_hz:8.1f} Hz   {order:6.2f} 階   SNR={snr:7.1f}{hit}")
    print("\n  判讀方式：BPFO 檔案應該在某個階次出現遠高於 Normal 的譜線，")
    print("  那個階次就是真正的 BPFO 係數。如果跟設定值差超過 0.15，")
    print("  把 features_vibration_v2.BEARING_ORDERS 改成實測值。")

    # ---------------------------------------------------------------- Q3
    print("\n" + "=" * 70)
    print("Q3  v2 特徵能否分開 Normal / Unbalance / Misalign？（v1 完全做不到）")
    print("=" * 70)
    for cond, x in sigs.items():
        rows = []
        for s in range(0, len(x) - wlen + 1, wlen):
            rows.append(v2.extract_features_v2(x[s:s + wlen, None], fs, rpm=fr_med * 60))
        feats[cond] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        print(f"  {cond:10s} 切出 {len(rows)} 個窗")

    key_feats = ["ch0_order_1x", "ch0_order_2x", "ch0_ratio_2x_1x",
                 "ch0_order_1x_dominance", "ch0_env_BPFO", "ch0_env_BPFI",
                 "ch0_env_peak_snr", "ch0_kurtosis", "ch0_crest_factor",
                 "ch0_rms"]

    print(f"\n  各狀態的特徵中位數：")
    hdr = "  " + "特徵".ljust(24) + "".join(c.rjust(12) for c in feats)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in key_feats:
        line = "  " + k.ljust(24)
        for c in feats:
            line += f"{np.median(feats[c][k]):12.4f}"
        print(line)

    print("\n  關鍵對比（Cohen's d，|d|>2 代表肉眼可分）：")
    pairs = [("Normal", "Unbalance"), ("Normal", "Misalign"),
             ("Unbalance", "Misalign"), ("BPFO", "BPFI")]
    for a, b in pairs:
        if a not in feats or b not in feats:
            continue
        ds = sorted(((abs(cohens_d(feats[a][k], feats[b][k])), k) for k in key_feats),
                    reverse=True)
        best = ", ".join(f"{k}(d={d:.1f})" for d, k in ds[:3])
        print(f"    {a:10s} vs {b:10s}  最強特徵: {best}")

    # 這是整個修正成敗的關鍵指標
    if "Normal" in feats and "Unbalance" in feats:
        d1x = abs(cohens_d(feats["Normal"]["ch0_order_1x"],
                           feats["Unbalance"]["ch0_order_1x"]))
        print(f"\n  ➜ Normal vs Unbalance 在 1x 振幅上的 d = {d1x:.2f}")
        if d1x > 2:
            print("     ✓ v2 成功。可以放心改 build_dataset.py 全量重跑。")
        elif d1x > 0.8:
            print("     ~ 有訊號但不夠強。建議先看看不同嚴重度的差異，"
                  "或把 Unbalance 最輕的 0583mg 拿掉再看一次。")
        else:
            print("     ✗ 1x 沒有分辨力。先不要重跑，把這段輸出貼給我，")
            print("       可能是通道對應、轉頻、或這個測試台的不平衡量太小。")


if __name__ == "__main__":
    main()
