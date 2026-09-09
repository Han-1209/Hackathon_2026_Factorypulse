"""
訓練 leave-one-load-out（LOLO）模型 —— 給儀表板上線用。

    python train_lolo.py

為什麼要有這支
--------------
儀表板原本載入 `models/model_vibration.pkl`，那是 train_baseline.py 實驗 A 的
產物：train / val / test 是「同一支錄音切前中後」，45 個 file_id 三份都有。
《數字口徑表》第三節已經把這種切分的數字（0.9977 / 1.0000）列為不可對外引用，
理由是錄音指紋洩漏——1-NN 檢驗顯示 98.4% 的測試樣本最近鄰來自同一支錄音。

但畫面上跑的一直是那個模型。後果在 demo 現場看得很清楚：

    10 台機台的 top-1 機率中位數全部落在 0.9996~0.9998
    逐窗一致率 10 台有 9 台是 100%
    異常分數曲線是水平線（FP-10 十七個窗全距 0.000002）

模型不是「很有把握」，是在背答案。簡報講「我們不報同錄音的數字」，
demo 卻由那個模型撐著，這是評審一問就破的矛盾。

這支腳本的做法
--------------
對每一個負載 L ∈ {0, 2, 4} 各訓練一個模型：用「其他兩個負載」訓練，
L 完全排除在訓練之外。儀表板上每台機台，改用「沒看過它那個負載」的那一個。

於是畫面上可以寫這句話，而且是真的：

    畫面上每一台的診斷，都來自沒看過那個負載的模型。

這與 train_baseline.py 實驗 B 是同一套設計（`fs.split_by_load`），
差別只在：實驗 B 只測 4 Nm 且刻意不存檔（它是壓力測試），
這裡三個負載都做、而且存下來給儀表板用。

順帶產出
--------
`results/lolo_confidence.csv` —— 每一支被held out的錄音檔，記錄
「信心度四條件」各自的通過與否，以及最後判對還是判錯。
這張表是回答「你們的信心度憑什麼」最直接的證據，evidence 頁會讀它。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)

import confidence as cf
import feature_selection as fs

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"

LOADS = (0, 2, 4)

# 二元 AUC 低於此值，視為「這個模態跨負載不成立」，儀表板不讓它參與風險計算。
# 必須與 dashboard_data.USABLE_MIN_AUC 一致。
USABLE_MIN_AUC = 0.70

# 與 train_baseline.make_model 相同的超參數。這裡不調參是刻意的：
# 換模型設定會讓 LOLO 的數字沒辦法跟實驗 B 的 88.8% 直接比較。
def make_model(seed: int = 42):
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1, tree_method="hist",
        )
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def train_one(modality: str, held_out: int) -> dict | None:
    """訓練「排除 held_out 這個負載」的模型，回傳可以 pickle 的 bundle。"""
    try:
        X, y, meta = fs.load_all_splits(modality, include_load=False)
    except FileNotFoundError as e:
        print(f"    略過 {modality}：{e}")
        return None

    # 有量綱特徵（rms / *_abs）在不同負載下落在完全不同的數值區間，
    # 樹模型的絕對切點會整批失效 —— 理由與實驗 B 相同。
    X = fs.drop_scale_dependent(X)
    X_tr, y_tr, _, X_te, y_te, m_te = fs.split_by_load(
        X, y, meta, test_loads=(held_out,))

    classes = sorted(y_tr.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    model = make_model()
    model.fit(X_tr.values, y_tr.map(c2i).values)

    proba = model.predict_proba(X_te.values)
    pred = np.array([classes[i] for i in proba.argmax(1)])
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")

    # 二元 AUC（正常 vs 有故障）。這是判斷「這個模態跨負載到底成不成立」的門檻：
    # 多類別準確率會被類別不平衡拉高，AUC 不會。實測電流模態在這裡只有
    # 0.44~0.61（等同亂猜），所以儀表板會依這個數字決定要不要讓電流參與風險計算。
    auc = None
    if "Normal" in classes:
        ni = classes.index("Normal")
        try:
            auc = float(roc_auc_score((y_te != "Normal").astype(int).values,
                                      1.0 - proba[:, ni]))
        except ValueError:
            auc = None

    auc_s = f"{auc:.3f}" if auc is not None else "—"
    print(f"    訓練 {len(X_tr):>5} 筆（排除 {held_out} Nm）"
          f" / 測試 {len(X_te):>5} 筆"
          f" -> 逐窗準確率 {acc:.4f}  macro-F1 {f1:.4f}  二元AUC {auc_s}")

    return {
        "holdout_auc": auc,
        "model": model,
        "classes": classes,
        "feature_names": list(X_tr.columns),
        "feature_version": "v2" if modality.endswith("v2") else "v1",
        # 給畫面顯示與稽核用：這個模型「沒看過」哪個負載
        "held_out_load": held_out,
        "training_loads": [l for l in LOADS if l != held_out],
        "holdout_accuracy": float(acc),
        "holdout_macro_f1": float(f1),
        "holdout_report": classification_report(
            y_te, pred, output_dict=True, zero_division=0),
        "n_train": int(len(X_tr)),
        "n_holdout": int(len(X_te)),
    }, (X_te, y_te, m_te, proba, classes)


def audit_confidence(runs: dict) -> pd.DataFrame:
    """對每一支 held-out 錄音檔，記錄信心度四條件與最終對錯。

    這是「信心度不是我們自己說了算」的證據：等級是先算出來的，
    對錯是後來比對的，兩者分開記錄，可以直接拿去給評審看。
    """
    import dashboard_data as dd
    import evidence as ev_mod
    import knowledge_base as kb

    try:
        ref = ev_mod.load_reference()
    except FileNotFoundError:
        ref = None

    rows = []
    for held_out, (X_te, y_te, m_te, proba, classes) in runs.items():
        X_te = X_te.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)
        m_te = m_te.reset_index(drop=True)
        ni = classes.index("Normal") if "Normal" in classes else None

        for fid, g in m_te.groupby("file_id"):
            idx = g.index.values
            pp = proba[idx]
            labels = [classes[i] for i in pp.argmax(1)]
            vc = pd.Series(labels).value_counts()
            fault = vc.index[0]
            agreement = float(vc.iloc[0] / len(labels))
            anomaly = (1.0 - pp[:, ni]) if ni is not None else pp.max(1)

            # 大部分窗正常但中位數異常 -> 早期故障，改報次高類別（與 infer_modality 同邏輯）
            if fault == "Normal" and float(np.median(anomaly)) >= 0.5 and len(vc) > 1:
                fault = vc.index[1]

            feats = X_te.iloc[idx].median(numeric_only=True).to_dict()
            evidence = []
            if ref is not None:
                evidence = ev_mod.extract(feats, load_nm=held_out, top=5,
                                          reference=ref, fault_type=fault)

            # 物理閘門：與儀表板走同一套邏輯，否則這張稽核表會跟畫面對不上
            gate = dd.physical_gate(evidence)
            overrode = None
            if gate == "quiet" and fault != "Normal":
                overrode, fault = fault, "Normal"

            verdict = cf.assess(
                evidence=evidence,
                fault_type=fault,
                agreement=agreement,
                n_windows=len(idx),
                cross_sensor=None,        # 這支腳本只稽核振動模型
                known_confusion=kb.get_fault_info(fault).get("known_confusion", ""),
                overrode=overrode,
            )
            truth = y_te.iloc[idx].iloc[0]
            row = {
                "held_out_load": held_out,
                "file_id": fid,
                "truth": truth,
                "predicted": fault,
                "correct": int(fault == truth),
                "gate": gate,
                "gate_overrode": overrode or "",
                "level": verdict["level"],
                "agreement": round(agreement, 4),
                "n_windows": len(idx),
            }
            for c in verdict["checks"]:
                row[f"check_{c['key']}"] = ("—" if c["passed"] is None
                                            else int(c["passed"]))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("訓練 leave-one-load-out 模型（儀表板上線用）")
    print("  每個模型都排除一個負載，儀表板上每台機台改用沒看過它那個負載的模型")
    print("=" * 72)

    vib_runs = {}
    saved = 0
    for modality, tag in (("vibration_v2", "vib"), ("current", "cur")):
        print(f"\n[{modality}]")
        for held_out in LOADS:
            out = train_one(modality, held_out)
            if out is None:
                break
            bundle, run = out
            path = MODELS_DIR / f"model_{tag}_lolo_{held_out}.pkl"
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
            print(f"      已存 {path.name}")
            if (bundle.get("holdout_auc") or 1.0) < USABLE_MIN_AUC:
                print(f"      ⚠️ 二元 AUC 低於 {USABLE_MIN_AUC}，跨負載不成立。"
                      f"儀表板會自動把這個模態排除在風險計算之外。")
            saved += 1
            if tag == "vib":
                vib_runs[held_out] = run

    if not saved:
        print("\n沒有任何模型被訓練出來。請先確認 processed/ 底下有特徵檔。")
        return 1

    if vib_runs:
        audit = audit_confidence(vib_runs)
        path = RESULTS_DIR / "lolo_confidence.csv"
        audit.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n信心度稽核表已存 {path}")

        print("\n" + "=" * 72)
        print("信心度分級 vs 實際對錯（全部 45 支錄音，每支都由沒看過它的模型判定）")
        print("=" * 72)
        tab = (audit.groupby("level")
               .agg(檔案數=("correct", "size"), 判對=("correct", "sum"))
               .reindex(["高", "中", "低"]).dropna(how="all"))
        tab["正確率"] = (tab["判對"] / tab["檔案數"]).map(
            lambda v: "—" if pd.isna(v) else f"{v:.0%}")
        print(tab.to_string())

        wrong = audit[audit.correct == 0]
        if len(wrong):
            print(f"\n判錯的 {len(wrong)} 支（確認低信心有把它們攔下來）：")
            print(wrong[["held_out_load", "file_id", "truth",
                         "predicted", "level", "agreement"]].to_string(index=False))

        acc = audit.correct.mean()
        print(f"\n檔案層級整體正確率：{audit.correct.sum()}/{len(audit)} = {acc:.1%}")
        print("\n⚠️ 這是 45 支錄音的檔案層級結果，樣本很小。"
              "對外請講「分級 + 小樣本」，不要把上表的正確率當成信心度的機率保證。")

    print("\n完成。重新啟動 streamlit 就會自動改用這些模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
