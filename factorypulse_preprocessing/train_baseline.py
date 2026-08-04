"""
FactoryPulse 第一版模型：XGBoost baseline

這支腳本會跑三組實驗，每一組都在回答一個評審會問的問題：

  實驗 A「單模態 vs 多模態」
      振動 / 電流 各自單獨訓練，再訓練「振動+電流」逐窗融合版。
      -> 回答：多感測融合到底加了幾分？這是你們產品定位的核心證據。
         如果融合沒有加分，就要誠實承認，不要硬凹。

      ⚠️ 溫度不放進分類器。溫度每個檔案只有 1 筆（共 45 筆）且值完全不重複，
         等於檔案指紋 = 洩漏答案。詳見 feature_selection._load_all 的說明。

  實驗 B「跨負載泛化」
      用 0Nm + 2Nm 訓練，完全沒看過的 4Nm 當測試。
      -> 回答：模型是真的學到故障特徵，還是只是記住某個負載下的數值範圍？
         這是回應「資料洩漏」質疑最有力的設計。

  實驗 C「二元 vs 五類」
      正常/異常二元分類，以及五種狀態的多類分類。
      -> 回答：產品最基本的「這台機器有沒有問題」準確度如何？
         二元通常會比五類高，先確保這個數字漂亮，demo 才站得住腳。

處理類別不平衡：
    Normal 753 筆 vs Unbalance 2505 筆，直接訓練模型會偏向猜 Unbalance。
    用 sample_weight='balanced' 讓少數類別的每筆樣本權重更高來抵銷。

用法：
    python train_baseline.py                 # 跑全部實驗
    python train_baseline.py --quick         # 只跑實驗 A（比較快）
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

import config
import feature_selection as fs

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

RESULTS_DIR = Path("./results")
MODELS_DIR = Path("./models")

# 振動特徵版本。用 --features v1 可切回舊版做「改善前 vs 改善後」對照。
# v1 已知無法分辨 Normal 與 Unbalance（頻域特徵數值完全相同），
# 它的分數只能當作反面教材，不要拿去對外報告。
MOD_VIB = "vibration_v2"
MOD_ALL = "all_v2"


def make_model(n_classes: int, seed: int = 42):
    """XGBoost 優先，沒裝就退回 RandomForest（結果會略差但流程一樣能跑）。"""
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1
    )


def check_leakage(X, meta, label="", uniq_ratio_thr=0.8):
    """偵測「檔案指紋」特徵 —— 這個資料集最容易踩的標籤洩漏陷阱。

    原理：本資料集每個檔案只含一種故障，所以「在檔案內固定不變、
    而且在檔案之間幾乎都不同」的特徵，等同於檔案編號，也就等同於答案。
    模型會去背它，訓練分數漂亮但完全不能用。
    （溫度特徵就是這樣把融合準確率從 0.96 拉到 0.58。）

    兩個條件要同時成立才算指紋：
      1. 檔案內常數（nunique == 1）
      2. 檔案間幾乎都不同（不重複值 / 檔案數 > uniq_ratio_thr）

    只滿足第 1 點的不算，例如 load_nm 在檔案內固定，但 45 個檔案只有
    3 種值（0/2/4Nm），無法辨識出是哪個檔案，而且它是推論時本來就知道的
    運轉條件，屬於合法輸入。
    """
    if "file_id" not in meta.columns:
        return []
    files = meta["file_id"].values
    n_files = len(set(files))
    suspects, benign = [], []
    for col in X.columns:
        if not X[col].groupby(files).nunique().eq(1).all():
            continue  # 檔案內會變動 -> 不是指紋
        ratio = X[col].groupby(files).first().nunique() / n_files
        (suspects if ratio > uniq_ratio_thr else benign).append((col, round(ratio, 2)))

    if suspects:
        print(f"    🚨 偵測到檔案指紋特徵（標籤洩漏）{label}：")
        for c, r in suspects:
            print(f"       {c}（{n_files} 個檔案有 {r:.0%} 個不重複值）")
        print("       這些特徵等於檔案編號 = 答案，必須排除，否則測試分數不可信。")
    elif benign:
        print(f"    ✓ 無洩漏。檔案內常數但值域很小的合法特徵："
              f"{[c for c, _ in benign]}")
    return [c for c, _ in suspects]


def train_and_eval(X_tr, y_tr, X_te, y_te, label: str, class_names=None, verbose=True,
                   X_val=None, y_val=None):
    """訓練 + 評估，回傳一個 dict 結果。

    X_val / y_val 可選。有給的話會「額外」在驗證集上評估一次，
    但驗證集不參與訓練，也不用來挑任何參數。

    為什麼要看兩組分數：
      val 與 test 是同一批錄音中「不同時間區段」切出來的，
      兩者分數若接近，代表模型表現穩定、不是剛好在某一段運氣好；
      若差距明顯（例如超過 3~5 個百分點），就要懷疑資料在時間上有漂移
      （感測器溫度上升、轉速設定改變等），這種訊號用單一 test 分數看不出來。
    """
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)

    # 測試集若出現訓練集沒有的類別會出錯，先過濾（跨負載實驗理論上不會發生）
    mask = np.isin(y_te, le.classes_)
    if not mask.all():
        dropped = sorted(set(y_te[~mask]))
        print(f"    ⚠️ 測試集有訓練集沒出現的類別 {dropped}，已排除 {(~mask).sum()} 筆")
        X_te, y_te = X_te[mask], y_te[mask]
    y_te_enc = le.transform(y_te)

    # 對齊欄位順序，避免不同模態的欄位順序不一致
    X_te = X_te[X_tr.columns]

    weights = compute_sample_weight("balanced", y_tr_enc)
    model = make_model(len(le.classes_))
    model.fit(X_tr.values, y_tr_enc, sample_weight=weights)

    pred = model.predict(X_te.values)
    acc = accuracy_score(y_te_enc, pred)
    f1m = f1_score(y_te_enc, pred, average="macro")

    val_acc = val_f1 = None
    n_val = 0
    if X_val is not None and y_val is not None and len(X_val):
        vmask = np.isin(y_val, le.classes_)
        Xv, yv = X_val[vmask], y_val[vmask]
        if len(Xv):
            Xv = Xv[X_tr.columns]
            yv_enc = le.transform(yv)
            vpred = model.predict(Xv.values)
            val_acc = float(accuracy_score(yv_enc, vpred))
            val_f1 = float(f1_score(yv_enc, vpred, average="macro"))
            n_val = int(len(Xv))

    if verbose:
        if val_acc is not None:
            gap = abs(val_acc - acc)
            flag = "" if gap < 0.03 else "   ⚠️ 兩者差距偏大，模型表現可能不穩定"
            print(f"    驗證集 val  accuracy = {val_acc:.4f}   macro-F1 = {val_f1:.4f}")
            print(f"    測試集 test accuracy = {acc:.4f}   macro-F1 = {f1m:.4f}"
                  f"   (val-test 差距 {gap:.4f}){flag}")
        else:
            print(f"    準確率 accuracy = {acc:.4f}   macro-F1 = {f1m:.4f}")

    return {
        "label": label,
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "n_train": int(len(X_tr)),
        "n_val": n_val,
        "n_test": int(len(X_te)),
        "n_features": int(X_tr.shape[1]),
        "classes": list(le.classes_),
        "model": model,
        "label_encoder": le,
        "y_true": y_te_enc,
        "y_pred": pred,
        "feature_names": list(X_tr.columns),
    }


def print_confusion(res, title=""):
    le = res["label_encoder"]
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    df = pd.DataFrame(cm, index=[f"真實_{c}" for c in le.classes_],
                      columns=[f"預測_{c}" for c in le.classes_])
    print(f"\n  混淆矩陣 {title}")
    print("  " + df.to_string().replace("\n", "\n  "))
    print()
    print("  各類別細部指標：")
    rep = classification_report(res["y_true"], res["y_pred"],
                                target_names=le.classes_, zero_division=0)
    print("  " + rep.replace("\n", "\n  "))


def top_features(res, n=15):
    m = res["model"]
    if not hasattr(m, "feature_importances_"):
        return pd.DataFrame()
    imp = pd.DataFrame({
        "feature": res["feature_names"],
        "importance": m.feature_importances_,
    }).sort_values("importance", ascending=False).head(n)
    imp["importance"] = imp["importance"].round(5)
    return imp


# ---------------------------------------------------------------- 實驗 A
def experiment_a():
    print("=" * 72)
    print("實驗 A：單模態 vs 融合（五類分類）")
    print("  問題：多感測融合到底比只用振動好多少？")
    print("=" * 72)

    results = {}
    for mod in [MOD_VIB, "current", MOD_ALL]:
        name = {MOD_VIB: "只用振動", "current": "只用電流",
                MOD_ALL: "振動+電流融合"}[mod]
        print(f"\n  [{name}]")
        try:
            X_tr, y_tr, m_tr = fs.load_features("train", mod)
            X_va, y_va, _ = fs.load_features("val", mod)
            X_te, y_te, _ = fs.load_features("test", mod)
        except FileNotFoundError as ex:
            print(f"    略過：{ex}")
            continue
        print(f"    訓練 {len(X_tr)} 筆 / 驗證 {len(X_va)} 筆 / 測試 {len(X_te)} 筆"
              f" / 特徵 {X_tr.shape[1]} 個")
        check_leakage(X_tr, m_tr)
        results[mod] = train_and_eval(X_tr, y_tr, X_te, y_te, name,
                                      X_val=X_va, y_val=y_va)

    print("\n  [只用溫度] 已從分類實驗中移除")
    print("    原因：溫度每個檔案只有 1 筆（共 45 筆），且 45 個值完全不重複，")
    print("    等於檔案指紋 = 直接洩漏答案。溫度的正確用途是規則引擎的趨勢判斷")
    print("    （是否持續升溫、是否需要停機），不是逐窗分類特徵。")

    print("\n" + "-" * 72)
    print("  結果總覽")
    print("-" * 72)
    summary = pd.DataFrame([
        {"實驗": r["label"], "特徵數": r["n_features"],
         "準確率": round(r["accuracy"], 4), "macro-F1": round(r["macro_f1"], 4)}
        for r in results.values()
    ])
    print("  " + summary.to_string(index=False).replace("\n", "\n  "))

    if MOD_VIB in results and MOD_ALL in results:
        gain = results[MOD_ALL]["accuracy"] - results[MOD_VIB]["accuracy"]
        print(f"\n  ➜ 融合相對於單用振動的增益：{gain:+.4f}")
        if gain < 0.005:
            print("     ⚠️ 增益很小。誠實的說法是「振動已足以分類，"
                  "其他模態的價值在於交叉驗證與降低誤報，而非提升準確率」，")
            print("        不要在簡報上宣稱融合大幅提升準確率。")

    if MOD_ALL in results:
        print_confusion(results[MOD_ALL], "（振動+電流融合）")
        print("  特徵重要度 Top 15：")
        print("  " + top_features(results[MOD_ALL]).to_string(index=False).replace("\n", "\n  "))

    return results


# ---------------------------------------------------------------- 實驗 B
def experiment_b():
    print("\n" + "=" * 72)
    print("實驗 B：跨負載泛化（0Nm+2Nm 訓練 → 4Nm 測試）")
    print("  問題：模型是學到故障特徵，還是只記住某個負載的數值範圍？")
    print("=" * 72)

    results = {}
    for mod in [MOD_VIB, MOD_ALL]:
        name = {MOD_VIB: "只用振動", MOD_ALL: "振動+電流融合"}[mod]
        print(f"\n  [{name}]")
        try:
            # 跨負載時不能把 load_nm 當特徵（測試集的 load 訓練時沒出現過）
            X, y, meta = fs.load_all_splits(mod, include_load=False)
        except FileNotFoundError as ex:
            print(f"    略過：{ex}")
            continue
        # 有量綱特徵（rms / *_abs）在不同負載下落在完全不同的數值區間，
        # 樹模型的絕對切點會整批失效，必須排除。詳見 fs.drop_scale_dependent。
        X = fs.drop_scale_dependent(X)
        X_tr, y_tr, _, X_te, y_te, _ = fs.split_by_load(X, y, meta, test_loads=(4,))
        print(f"    訓練 {len(X_tr)} 筆 (0Nm+2Nm) / 測試 {len(X_te)} 筆 (4Nm)")
        results[mod] = train_and_eval(X_tr, y_tr, X_te, y_te, name + "（跨負載）")

    if MOD_ALL in results:
        print_confusion(results[MOD_ALL], "（跨負載・融合）")
    return results


# ---------------------------------------------------------------- 實驗 C
def experiment_c():
    print("\n" + "=" * 72)
    print("實驗 C：二元分類（正常 vs 異常）")
    print("  問題：產品最基本的『這台機器有沒有問題』準確度如何？")
    print("=" * 72)

    results = {}
    for mod in [MOD_VIB, MOD_ALL]:
        name = {MOD_VIB: "只用振動", MOD_ALL: "振動+電流融合"}[mod]
        print(f"\n  [{name}]")
        try:
            X_tr, y_tr, _ = fs.load_features("train", mod)
            X_va, y_va, _ = fs.load_features("val", mod)
            X_te, y_te, _ = fs.load_features("test", mod)
        except FileNotFoundError as ex:
            print(f"    略過：{ex}")
            continue
        y_tr_bin = np.where(y_tr == "Normal", "正常", "異常")
        y_va_bin = np.where(y_va == "Normal", "正常", "異常")
        y_te_bin = np.where(y_te == "Normal", "正常", "異常")
        results[mod] = train_and_eval(X_tr, pd.Series(y_tr_bin), X_te,
                                      pd.Series(y_te_bin), name + "（二元）",
                                      X_val=X_va, y_val=pd.Series(y_va_bin))

    if MOD_ALL in results:
        print_confusion(results[MOD_ALL], "（二元・融合）")
        print("  ➜ 產品意義：左下角『真實異常但被判為正常』是漏報，")
        print("     對預測維護來說代價最高，簡報時要主動說明這個數字。")
    return results


def save_models(results_a: dict):
    """把實驗 A 訓練好的模型存下來，給 diagnose_pipeline.py 做端到端推論用。

    存的是實驗 A（同機台、時間切分）的模型，因為那是產品實際部署的情境：
    設備先收集自己的正常基準與歷史資料，模型在該設備上運作。
    跨負載那組刻意不存 —— 它是壓力測試，不是要拿來上線的模型。
    """
    MODELS_DIR.mkdir(exist_ok=True)
    saved = []
    for mod in [MOD_VIB, "current"]:
        if mod not in results_a:
            continue
        r = results_a[mod]
        bundle = {
            "model": r["model"],
            "classes": list(r["label_encoder"].classes_),
            "feature_names": r["feature_names"],
            "modality": mod,
            # ⚠️ 這個數字是「實驗 A：同一段錄音切前中後」測出來的，
            # train/test 共用同一支錄音的雜訊指紋，數值虛高，**不得對外引用**。
            # 欄位名稱刻意取得很長，就是為了讓任何人想把它顯示在畫面上之前，
            # 先讀到這個警告。對外可用的數字只有跨負載泛化（實驗 B）。
            # 完整規範見 results/數字口徑.md。
            "accuracy_same_recording_DO_NOT_PUBLISH": r["accuracy"],
        }
        # 檔名固定叫 model_vibration.pkl / model_current.pkl，不帶版本後綴，
        # 這樣 diagnose_pipeline.py 與 dashboard_data.py 不用跟著改。
        # 用哪一版特徵記在 bundle["feature_version"] 裡，推論時才好核對。
        short = "vibration" if mod.startswith("vibration") else mod
        bundle["feature_version"] = "v2" if mod.endswith("_v2") else "v1"
        path = MODELS_DIR / f"model_{short}.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        saved.append(path.name)
    print(f"\n模型已存到 {MODELS_DIR.resolve()}：{saved}")


def save_summary(all_results: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for exp_name, res_dict in all_results.items():
        for r in res_dict.values():
            va = r.get("val_accuracy")
            rows.append({
                "實驗": exp_name,
                "設定": r["label"],
                "val準確率": round(va, 4) if va is not None else "",
                "test準確率": round(r["accuracy"], 4),
                # val 與 test 的差距：接近 0 代表模型表現穩定，
                # 差距大代表資料在時間上有漂移，單看一個分數會誤判
                "val_test差距": round(abs(va - r["accuracy"]), 4) if va is not None else "",
                "macro_F1": round(r["macro_f1"], 4),
                "特徵數": r["n_features"],
                "訓練筆數": r["n_train"],
                "驗證筆數": r.get("n_val", 0),
                "測試筆數": r["n_test"],
            })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "baseline_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n所有結果已存成 {out.resolve()}")

    # 特徵重要度另存，之後前端「異常證據」頁面會用到
    for exp_name, res_dict in all_results.items():
        for mod, r in res_dict.items():
            imp = top_features(r, n=30)
            if not imp.empty:
                imp.to_csv(RESULTS_DIR / f"importance_{exp_name}_{mod}.csv",
                           index=False, encoding="utf-8-sig")

    # 混淆矩陣另存成 CSV —— 簡報用的圖必須從這裡讀，不可以憑印象重建。
    # （做圖時曾經發生過拿別的實驗的矩陣來充數，數字總和對不上。
    #   圖表上的每一個數字都必須能追溯到一次真實執行。）
    for exp_name, res_dict in all_results.items():
        for mod, r in res_dict.items():
            cm = confusion_matrix(r["y_true"], r["y_pred"])
            classes = list(r["label_encoder"].classes_)
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                RESULTS_DIR / f"confusion_{exp_name}_{mod}.csv",
                encoding="utf-8-sig")
    print(f"混淆矩陣已存成 {RESULTS_DIR}/confusion_*.csv")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑實驗 A")
    parser.add_argument("--features", choices=["v1", "v2"], default="v2",
                        help="振動特徵版本。v2=階次域+包絡（預設）；"
                             "v1=舊版三頻帶，只用來做改善前後對照")
    args = parser.parse_args()

    global MOD_VIB, MOD_ALL
    if args.features == "v1":
        MOD_VIB, MOD_ALL = "vibration", "all"
        print("⚠️ 使用 v1 舊版振動特徵。這一版已知無法分辨 Normal 與 Unbalance，")
        print("   分數只能當作『改善前』的對照組，不要拿去對外報告。\n")

    if not HAS_XGB:
        print("⚠️ 沒有安裝 xgboost，改用 RandomForest。若要用 XGBoost：pip install xgboost\n")

    results_a = experiment_a()
    save_models(results_a)
    all_results = {"A_單模態vs融合": results_a}
    if not args.quick:
        all_results["B_跨負載泛化"] = experiment_b()
        all_results["C_二元分類"] = experiment_c()

    print("\n" + "=" * 72)
    print("全部實驗完成")
    print("=" * 72)
    save_summary(all_results)


if __name__ == "__main__":
    main()
