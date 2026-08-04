"""
變轉速資料集的模型訓練。

沿用 train_baseline.py 的訓練/評估函式，只是換資料來源（processed_speed/），
所以兩個資料集的結果是用同一套標準算出來的，可以直接比較。

三組實驗：
  D「四分類」        Normal / BPFI / BPFO / Ball，振動 vs 振動+電流
                     -> 重點看 BPFO 的 recall（前處理時算出它的 Cohen's d 只有 0.61，
                        單一特徵很弱，但 XGBoost 能組合幾十個特徵，實際結果可能更好）
  E「跨轉速泛化」    用轉速曲線 0~4 訓練、5~6 測試
                     -> 對應變負載資料集的實驗 B（那邊跨負載只有 0.48）
  F「二元分類」      正常 vs 異常，這是產品的「健康分數」核心

用法：
    python train_speed.py            # 全部三組
    python train_speed.py --quick    # 只跑實驗 D
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from train_baseline import train_and_eval, print_confusion, top_features

warnings.filterwarnings("ignore", category=FutureWarning)

META_COLS = ["condition", "severity_level", "profile", "file_id", "split"]

# 存模型的位置。與變負載模型放同一個資料夾，但檔名加 _speed 區隔，
# 儀表板才能同時載入兩個模型讓使用者選擇運轉模式。
MODELS_DIR = Path("./models")

# 與變負載資料集相同的理由：去直流後 std 與 rms 數學上完全相等
EXACT_DUPLICATES = ["ch0_std", "ch1_std", "ch2_std", "ch3_std"]


def load_speed(split: str, modality: str = "vibration", include_rpm: bool = True):
    """載入變轉速資料集的特徵。

    include_rpm: 是否把轉速當特徵。
        這裡預設 True 是安全的——前處理後實測四個類別的轉速分布幾乎相同
        （平均 1555~1610 RPM，範圍都是 618~2480），所以轉速不會變成
        「哪一類」的指紋。這點跟變負載資料集不一樣，那邊負載與訊號強度高度相關。
    """
    out = config.SPEED_OUTPUT_DIR

    def _read(mod):
        p = out / f"{mod}_features_{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}，請先執行 build_dataset_speed.py")
        df = pd.read_csv(p)
        return df.drop(columns=[c for c in EXACT_DUPLICATES if c in df.columns])

    vib = _read("vibration")

    if modality == "vibration":
        merged = vib
    elif modality == "all":
        cur = _read("current")
        # 逐窗對齊：兩者每個檔案的窗數相同（實測皆為 11732/2492/2492）
        vib = vib.copy()
        cur = cur.copy()
        vib["_win"] = vib.groupby("file_id").cumcount()
        cur["_win"] = cur.groupby("file_id").cumcount()
        cur_feat = [c for c in cur.columns if c.startswith("current_")]
        cur_small = cur[["file_id", "_win"] + cur_feat].rename(
            columns={c: "cur_" + c for c in cur_feat}
        )
        before = len(vib)
        merged = vib.merge(cur_small, on=["file_id", "_win"], how="inner").drop(columns=["_win"])
        if len(merged) != before:
            print(f"    ⚠️ 逐窗對齊後筆數 {before} -> {len(merged)}（已取交集）")
    else:
        raise ValueError(f"未知的 modality: {modality}")

    y = merged["condition"]
    meta = merged[[c for c in META_COLS if c in merged.columns]].copy()
    X = merged.drop(columns=[c for c in META_COLS if c in merged.columns]).select_dtypes("number")
    if not include_rpm and "rpm" in X.columns:
        X = X.drop(columns=["rpm"])
    return X, y, meta


def load_all_splits(modality="vibration", include_rpm=True):
    Xs, ys, ms = [], [], []
    for s in ["train", "val", "test"]:
        X, y, m = load_speed(s, modality, include_rpm)
        Xs.append(X); ys.append(y); ms.append(m)
    return (pd.concat(Xs, ignore_index=True),
            pd.concat(ys, ignore_index=True),
            pd.concat(ms, ignore_index=True))


# ---------------------------------------------------------------- 實驗 D
def experiment_d():
    print("=" * 72)
    print("實驗 D：變轉速資料集 四分類（Normal / BPFI / BPFO / Ball）")
    print("  重點：BPFO 的 recall —— 前處理時它的 Cohen's d 只有 0.61")
    print("=" * 72)

    results = {}
    for modality, label in [("vibration", "只用振動"), ("all", "振動+電流融合")]:
        print(f"\n  [{label}]")
        X_tr, y_tr, _ = load_speed("train", modality)
        X_va, y_va, _ = load_speed("val", modality)
        X_te, y_te, _ = load_speed("test", modality)
        print(f"    訓練 {len(X_tr)} 筆 / 驗證 {len(X_va)} 筆 / 測試 {len(X_te)} 筆")
        res = train_and_eval(X_tr, y_tr, X_te, y_te, label, X_val=X_va, y_val=y_va)
        print_confusion(res, label)
        print("  最重要的特徵：")
        print("  " + top_features(res, 10).to_string(index=False).replace("\n", "\n  "))
        results[label] = res
    return results


# ---------------------------------------------------------------- 實驗 E
def experiment_e():
    print("\n" + "=" * 72)
    print("實驗 E：跨轉速泛化（用曲線 0~4 訓練，5~6 測試）")
    print("  問題：模型換一組沒看過的轉速變化曲線還能用嗎？")
    print("  對照：變負載資料集的跨負載實驗只有 0.48")
    print("=" * 72)

    results = {}
    for modality, label in [("vibration", "只用振動（跨轉速）"), ("all", "振動+電流融合（跨轉速）")]:
        print(f"\n  [{label}]")
        X, y, meta = load_all_splits(modality)
        is_test = meta["profile"].isin([5, 6]).values
        X_tr, y_tr = X[~is_test].reset_index(drop=True), y[~is_test].reset_index(drop=True)
        X_te, y_te = X[is_test].reset_index(drop=True), y[is_test].reset_index(drop=True)
        print(f"    訓練 {len(X_tr)} 筆（曲線 0-4）／測試 {len(X_te)} 筆（曲線 5-6）")
        res = train_and_eval(X_tr, y_tr, X_te, y_te, label)
        print_confusion(res, label)
        results[label] = res
    return results


# ---------------------------------------------------------------- 實驗 F
def experiment_f():
    print("\n" + "=" * 72)
    print("實驗 F：二元分類（正常 vs 異常）—— 產品「健康分數」的核心")
    print("=" * 72)

    results = {}
    for modality, label in [("vibration", "只用振動（二元）"), ("all", "振動+電流融合（二元）")]:
        print(f"\n  [{label}]")
        X_tr, y_tr, _ = load_speed("train", modality)
        X_va, y_va, _ = load_speed("val", modality)
        X_te, y_te, _ = load_speed("test", modality)
        to_bin = lambda s: s.apply(lambda c: "Normal" if c == "Normal" else "Fault")
        res = train_and_eval(X_tr, to_bin(y_tr), X_te, to_bin(y_te), label,
                             X_val=X_va, y_val=to_bin(y_va))
        print_confusion(res, label)
        results[label] = res
    return results


def save_models(results_d: dict):
    """把實驗 D 訓練好的模型存下來，給儀表板／端到端診斷使用。

    存實驗 D 而不是 E，理由與變負載那邊一致：
    D 是「同一台設備、依時間切分」，對應產品實際部署情境
    （設備先累積自己的資料，模型在該設備上運作）。
    E 是壓力測試，用來驗證泛化能力，不是要上線的模型。
    """
    MODELS_DIR.mkdir(exist_ok=True)
    saved = []
    for label, res in results_d.items():
        # 只用振動 -> model_speed_vibration.pkl
        # 振動+電流融合 -> model_speed_all.pkl
        tag = "vibration" if "只用振動" in label else "all"
        bundle = {
            "model": res["model"],
            "classes": list(res["label_encoder"].classes_),
            "feature_names": res["feature_names"],
            "modality": tag,
            "dataset": "speed",          # 標明是變轉速模型，避免與變負載模型混用
            # ⚠️ 同一段錄音切前中後測出來的數字，虛高，不得對外引用。
            # 變轉速資料集可對外的數字是「跨轉速泛化」(實驗 E)。
            # 見 results/數字口徑.md。
            "accuracy_same_recording_DO_NOT_PUBLISH": res["accuracy"],
        }
        path = MODELS_DIR / f"model_speed_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        saved.append(path.name)
    print(f"\n變轉速模型已存到 {MODELS_DIR.resolve()}：{saved}")


def save_summary(all_results: dict, out_dir: Path):
    rows = []
    for exp, group in all_results.items():
        for label, res in group.items():
            va = res.get("val_accuracy")
            rows.append({
                "實驗": exp,
                "設定": label,
                "val準確率": round(va, 4) if va is not None else "",
                "test準確率": round(res["accuracy"], 4),
                "val_test差距": round(abs(va - res["accuracy"]), 4) if va is not None else "",
                "macro_F1": round(res["macro_f1"], 4),
                "特徵數": res["n_features"],
                "訓練筆數": res["n_train"],
                "驗證筆數": res.get("n_val", 0),
                "測試筆數": res["n_test"],
            })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "speed_summary.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print("總結（變轉速資料集）")
    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\n已存到 {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑實驗 D")
    args = ap.parse_args()

    results_d = experiment_d()
    save_models(results_d)

    all_results = {"D_四分類": results_d}
    if not args.quick:
        all_results["E_跨轉速泛化"] = experiment_e()
        all_results["F_二元分類"] = experiment_f()

    save_summary(all_results, Path("./results"))


if __name__ == "__main__":
    main()
