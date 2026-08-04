"""
FactoryPulse 強健性測試：實驗 G（感測器缺失）與實驗 H（雜訊）
================================================================

這兩個實驗回答的是「模型成果資料需求清單」第 7 節的兩個空格。
兩者都掛在**跨負載設定**（實驗 B：0Nm+2Nm 訓練 → 4Nm 測試）之上，
理由是：只有跨負載那組的測試檔案完全沒參與訓練，數字才對外可用。
如果拿實驗 A（同一支錄音切前中後）當底，強健性數字會連同錄音指紋
一起虛高，做了等於沒做。

  實驗 G「感測器缺失」
      問題：部署現場少裝／掉線一顆加速規，模型還能用嗎？
      兩種情境分開測，因為它們對應不同的商業問題：
        G1 retrain  —— 訓練時就沒有那顆感測器（= 客戶只買得起兩顆）
        G2 dropout  —— 訓練時有、推論時掉線（= 運轉中線路鬆脫）
      G2 用訓練集中位數填補缺失欄位。這是「特徵層」的模擬，
      不等於真的把該通道的線拔掉，簡報時要照實說明（見下方 CAVEAT）。

  實驗 H「雜訊」
      問題：現場的電磁干擾／感測器底噪會不會讓模型失效？
      白雜訊必須加在**原始波形**上，再重新萃取特徵 —— 不能在特徵層加。
      v2 的包絡特徵本身就是 SNR 形式的比值（分母是包絡平均值），
      在特徵層加高斯雜訊在物理上沒有意義，被問就會倒。
      因此這支腳本會重新讀取 15 個 4Nm 的 .mat 原始檔。

      協定：訓練集乾淨（0Nm+2Nm 原始特徵），測試集加噪。
      這符合實務 —— 模型在乾淨的驗證環境訓練，部署到吵雜的現場。

用法：
    python train_robustness.py              # 兩個都跑
    python train_robustness.py --only G     # 只跑感測器缺失（快，約 1 分鐘）
    python train_robustness.py --only H     # 只跑雜訊（需重讀原始檔，約 15-30 分鐘）
    python train_robustness.py --only H --snr 10 0    # 只測指定 SNR

輸出：
    results/robustness_summary.csv
    results/confusion_G_感測器缺失_*.csv
    results/confusion_H_雜訊_*.csv

⚠️ CAVEAT（寫進簡報前必讀）
    G2 的缺失是用中位數填補模擬的，不是真的重新萃取。真實掉線時該通道
    會輸出接近零的訊號，其特徵值不會等於訓練集中位數。中位數填補是
    「模型完全不知道該通道的資訊」的理想化情境，比真實掉線樂觀一點
    （真實掉線會餵進誤導性的數值）。要完全精確就得回原始波形把通道歸零
    再重跑一次萃取 —— 那等於實驗 H 的成本。目前的取捨是 G1 給下限、
    G2 給上限，兩個一起報。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import feature_selection as fs
from train_baseline import train_and_eval, print_confusion

RESULTS_DIR = Path("./results")

MOD_VIB = "vibration_v2"
MOD_ALL = "all_v2"

# 通道對應（見 io_utils.load_vibration_mat 的欄位順序）
CHANNEL_MEANING = {
    "ch0": "Housing A - X 軸",
    "ch1": "Housing A - Y 軸",
    "ch2": "Housing B - X 軸",
    "ch3": "Housing B - Y 軸",
}

# 感測器缺失情境。keep_channels=None 代表全留。
# 每個情境都要能對應到一個真實的現場狀況，不是隨便組合欄位。
DROPOUT_SCENARIOS = [
    ("全部感測器（對照組）", ["ch0", "ch1", "ch2", "ch3"], True),
    ("只有 Housing A（近端軸承座）", ["ch0", "ch1"], False),
    ("只有 Housing B（遠端軸承座）", ["ch2", "ch3"], False),
    ("只有 X 軸（兩座各一顆）", ["ch0", "ch2"], False),
    ("只有 Y 軸（兩座各一顆）", ["ch1", "ch3"], False),
    ("只剩單顆 ch0", ["ch0"], False),
    ("振動全失效，只剩電流", [], True),
]


def channel_columns(cols, keep_channels):
    """從特徵欄位中挑出屬於指定通道的欄位。

    非 ch* 開頭的欄位（電流 cur_*）由 keep_current 另外控制，這裡一律保留，
    由呼叫端決定要不要再濾掉。
    """
    out = []
    for c in cols:
        prefix = c.split("_")[0]
        if prefix in CHANNEL_MEANING:
            if prefix in keep_channels:
                out.append(c)
        else:
            out.append(c)
    return out


def split_cross_load(modality):
    """重建實驗 B 的跨負載切分，回傳 (X_tr, y_tr, X_te, y_te)。

    與 train_baseline.experiment_b() 完全一致：不含 load_nm、排除有量綱特徵。
    這樣 G/H 的數字才能跟 0.8878 直接比較。
    """
    X, y, meta = fs.load_all_splits(modality, include_load=False)
    X = fs.drop_scale_dependent(X)
    X_tr, y_tr, _, X_te, y_te, _ = fs.split_by_load(X, y, meta, test_loads=(4,))
    return X_tr, y_tr, X_te, y_te


# ================================================================ 實驗 G
def experiment_g():
    print("=" * 72)
    print("實驗 G：感測器缺失（跨負載設定）")
    print("  問題：少裝或掉線一顆加速規，模型還能用嗎？")
    print("=" * 72)

    X_tr, y_tr, X_te, y_te = split_cross_load(MOD_ALL)
    print(f"\n  基準：訓練 {len(X_tr)} 筆 (0Nm+2Nm) / 測試 {len(X_te)} 筆 (4Nm)"
          f" / 特徵 {X_tr.shape[1]} 個")

    cur_cols = [c for c in X_tr.columns if c.startswith("cur_")]
    print(f"  其中電流特徵 {len(cur_cols)} 個，振動特徵 {X_tr.shape[1] - len(cur_cols)} 個")

    rows, results = [], {}

    for scen_name, keep_ch, keep_cur in DROPOUT_SCENARIOS:
        for mode in ["retrain", "dropout"]:
            if not keep_ch and mode == "dropout":
                continue  # 振動全失效 + dropout 等同全部填中位數，沒有資訊量
            label = f"{scen_name}｜{'重新訓練' if mode == 'retrain' else '推論時掉線'}"
            print(f"\n  [{label}]")

            cols = channel_columns(X_tr.columns, keep_ch)
            if not keep_cur:
                cols = [c for c in cols if not c.startswith("cur_")]

            if mode == "retrain":
                # 訓練與測試都只看得到剩下的感測器
                Xtr2, Xte2 = X_tr[cols], X_te[cols]
            else:
                # 訓練時全部都有；測試時把缺失欄位換成訓練集中位數
                missing = [c for c in X_tr.columns if c not in cols]
                Xtr2 = X_tr
                Xte2 = X_te.copy()
                med = X_tr[missing].median()
                for c in missing:
                    Xte2[c] = med[c]

            if Xtr2.shape[1] == 0:
                print("    略過：沒有任何特徵留下")
                continue

            key = f"{scen_name}_{mode}"
            res = train_and_eval(Xtr2, y_tr, Xte2, y_te, label)
            results[key] = res
            rows.append({
                "實驗": "G_感測器缺失",
                "情境": scen_name,
                "模式": "重新訓練" if mode == "retrain" else "推論時掉線",
                "保留通道": "+".join(keep_ch) if keep_ch else "（無振動）",
                "含電流": "是" if keep_cur else "否",
                "特徵數": res["n_features"],
                "test準確率": round(res["accuracy"], 4),
                "macro_F1": round(res["macro_f1"], 4),
            })

    df = pd.DataFrame(rows)
    print("\n" + "-" * 72)
    print("  實驗 G 總覽")
    print("-" * 72)
    print("  " + df.to_string(index=False).replace("\n", "\n  "))

    base = df[(df["情境"] == "全部感測器（對照組）") & (df["模式"] == "重新訓練")]
    if len(base):
        b = float(base["test準確率"].iloc[0])
        print(f"\n  ➜ 對照組準確率 {b:.4f}。相對於它的掉幅：")
        for _, r in df.iterrows():
            if r["情境"] == "全部感測器（對照組）":
                continue
            print(f"     {r['test準確率'] - b:+.4f}   {r['情境']}｜{r['模式']}")
        print("\n  提醒：掉幅小不代表那顆感測器沒用，只代表「在這個資料集的")
        print("        故障類型下，剩下的通道已經帶有足夠資訊」。")

    return df, results


# ================================================================ 實驗 H
def add_noise(sig: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """對每個通道各自加白高斯雜訊，達到指定 SNR。

    SNR 逐通道定義（不是整體），因為四個通道的振幅量級本來就不同，
    用整體 RMS 會讓振幅小的通道實際上被加了更重的雜訊，
    導致「SNR=10dB」這個標籤在不同通道上意義不一致。
    """
    if snr_db is None or not np.isfinite(snr_db):
        return sig
    out = np.empty_like(sig)
    for c in range(sig.shape[1]):
        x = sig[:, c]
        p_sig = float(np.mean(x.astype(np.float64) ** 2))
        sigma = np.sqrt(p_sig / (10.0 ** (snr_db / 10.0)))
        out[:, c] = x + rng.normal(0.0, sigma, size=x.shape).astype(x.dtype)
    return out


def extract_noisy_test_features(snr_list, rpm=3010.0, seed=42):
    """讀 15 個 4Nm 原始檔，對每個 SNR 產生一份特徵表。

    切窗方式與 build_dataset.process_vibration 完全一致
    （先 split_signal_by_time 切三段、再各自 sliding_window、依 train/val/test 順序串接），
    這樣窗數與順序才會跟 fs.load_all_splits 出來的 4Nm 測試集完全對得上。
    """
    import io_utils
    import windowing
    import features_vibration_v2 as v2

    manifest = pd.read_csv(config.OUTPUT_DIR / "manifest.csv")
    targets = manifest[manifest["load_nm"] == 4].reset_index(drop=True)
    print(f"\n  4Nm 原始檔 {len(targets)} 個，SNR 等級 {snr_list}")

    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    buckets = {snr: [] for snr in snr_list}

    for idx, e in targets.iterrows():
        t0 = time.time()
        print(f"  ({idx + 1}/{len(targets)}) {e['file_id']} ...", end="", flush=True)
        sig = io_utils.load_vibration_mat(Path(e["vibration_path"]))  # (N, 4)

        for snr in snr_list:
            # 每個 (檔案, SNR) 用固定 seed，重跑結果可重現
            rng = np.random.default_rng(seed + int(idx) * 1000
                                        + (0 if snr is None else int(snr)))
            noisy = add_noise(sig, snr, rng)
            chunks = windowing.split_signal_by_time(noisy, ratios)
            del noisy
            for split in ["train", "val", "test"]:
                try:
                    wins = windowing.sliding_window(
                        chunks[split], config.VIBRATION_FS,
                        config.WINDOW_SEC, config.WINDOW_OVERLAP)
                except ValueError:
                    continue
                for w in wins:
                    feats = v2.extract_features_v2(w, config.VIBRATION_FS, rpm=rpm)
                    feats["condition"] = e["condition"]
                    feats["file_id"] = e["file_id"]
                    buckets[snr].append(feats)
                del wins
            del chunks
        del sig
        print(f" 完成 ({time.time() - t0:.0f}s)", flush=True)

    return {snr: pd.DataFrame(rows) for snr, rows in buckets.items()}


def experiment_h(snr_list):
    print("\n" + "=" * 72)
    print("實驗 H：雜訊強健性（跨負載設定）")
    print("  問題：現場的電磁干擾與感測器底噪會不會讓模型失效？")
    print("  協定：訓練集乾淨（0Nm+2Nm），測試集（4Nm）加白雜訊")
    print("=" * 72)

    # 訓練集：與實驗 B 完全相同（振動單模態，因為雜訊只加在振動上）
    X_tr, y_tr, X_te_clean, y_te = split_cross_load(MOD_VIB)
    print(f"\n  訓練 {len(X_tr)} 筆 (0Nm+2Nm) / 測試 {len(X_te_clean)} 筆 (4Nm)"
          f" / 特徵 {X_tr.shape[1]} 個")

    feat_tables = extract_noisy_test_features(snr_list)

    rows, results = [], {}
    for snr in snr_list:
        tag = "乾淨（對照組）" if snr is None else f"SNR {snr} dB"
        df_noisy = feat_tables[snr]
        print(f"\n  [{tag}]  重新萃取出 {len(df_noisy)} 個窗")

        if len(df_noisy) != len(X_te_clean):
            print(f"    ⚠️ 窗數與原測試集 {len(X_te_clean)} 不符。"
                  f"切窗參數可能與 build_dataset 不一致，請先排除再看數字。")

        missing = [c for c in X_tr.columns if c not in df_noisy.columns]
        if missing:
            print(f"    ⚠️ 缺少 {len(missing)} 個訓練用欄位：{missing[:5]}")
            continue

        X_te = df_noisy[X_tr.columns]
        y_te_n = df_noisy["condition"]

        res = train_and_eval(X_tr, y_tr, X_te, y_te_n, f"跨負載・{tag}")
        results[tag] = res
        rows.append({
            "實驗": "H_雜訊",
            "情境": tag,
            "SNR_dB": "∞" if snr is None else snr,
            "特徵數": res["n_features"],
            "測試筆數": res["n_test"],
            "test準確率": round(res["accuracy"], 4),
            "macro_F1": round(res["macro_f1"], 4),
        })

    df = pd.DataFrame(rows)
    print("\n" + "-" * 72)
    print("  實驗 H 總覽")
    print("-" * 72)
    print("  " + df.to_string(index=False).replace("\n", "\n  "))

    if "乾淨（對照組）" in results:
        b = results["乾淨（對照組）"]["accuracy"]
        print(f"\n  ➜ 乾淨對照組 {b:.4f}"
              f"（應該要接近 results/baseline_summary.csv 的 0.8878，"
              f"差太多代表切窗或特徵流程有出入）")
        print_confusion(results["乾淨（對照組）"], "（雜訊實驗・乾淨對照）")

    worst = [s for s in snr_list if s is not None]
    if worst and f"SNR {min(worst)} dB" in results:
        print_confusion(results[f"SNR {min(worst)} dB"],
                        f"（最惡劣 SNR {min(worst)} dB）")

    return df, results


# ================================================================ 存檔
def save(dfs, results_map):
    RESULTS_DIR.mkdir(exist_ok=True)
    from sklearn.metrics import confusion_matrix

    df = pd.concat([d for d in dfs if d is not None and len(d)], ignore_index=True)
    out = RESULTS_DIR / "robustness_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n結果已存成 {out.resolve()}")

    for exp_name, results in results_map.items():
        for key, r in results.items():
            cm = confusion_matrix(r["y_true"], r["y_pred"])
            classes = list(r["label_encoder"].classes_)
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in key)
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                RESULTS_DIR / f"confusion_{exp_name}_{safe}.csv",
                encoding="utf-8-sig")
    print(f"混淆矩陣已存成 {RESULTS_DIR}/confusion_G_*.csv 與 confusion_H_*.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["G", "H"], help="只跑其中一個實驗")
    p.add_argument("--snr", nargs="*", type=float, default=[20, 10, 5, 0],
                   help="雜訊實驗的 SNR 等級 (dB)，預設 20 10 5 0")
    args = p.parse_args()

    dfs, results_map = [], {}

    if args.only != "H":
        df_g, res_g = experiment_g()
        dfs.append(df_g)
        results_map["G_感測器缺失"] = res_g

    if args.only != "G":
        snr_list = [None] + sorted(args.snr, reverse=True)  # None = 乾淨對照組
        df_h, res_h = experiment_h(snr_list)
        dfs.append(df_h)
        results_map["H_雜訊"] = res_h

    print("\n" + "=" * 72)
    print("強健性測試完成")
    print("=" * 72)
    save(dfs, results_map)


if __name__ == "__main__":
    main()
