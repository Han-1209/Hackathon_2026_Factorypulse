"""變轉速模式的 leave-one-profile-out（LOPO）模型 —— 給儀表板上線用。

    python train_speed_lopo.py

為什麼要有這支
--------------
變轉速模式原本載入 models/model_speed_all.pkl，那是 train_speed.py 實驗 D 的
產物：每支錄音切成前中後三段，train / val / test 都來自同一支錄音。
模型在 test 段上等於看過同一支錄音的前段，逐窗準確率 99% 是在背答案。

這支腳本的做法與變負載的 train_lolo.py 相同，只是「輪流排除」的對象換成
轉速曲線（profile 0~6）：

    對每條轉速曲線 P，用其他 6 條曲線的全部錄音訓練，P 完全排除。
    儀表板上每台機台，改用「沒看過它那條轉速曲線」的模型。

輸出
----
    models/model_speed_lopo_{0..6}.pkl     儀表板自動載入
    results/speed_lopo_summary.csv         每條曲線的逐窗準確率 / macro-F1 / 二元 AUC
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from train_baseline import make_model
from train_speed import load_all_splits

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"
PROFILES = (0, 1, 2, 3, 4, 5, 6)


def feature_columns(X: pd.DataFrame) -> list[str]:
    """與原本上線的融合模型使用同一組特徵欄位，畫面與上傳流程不必跟著改。"""
    old = MODELS_DIR / "model_speed_all.pkl"
    if old.exists():
        with open(old, "rb") as f:
            names = pickle.load(f)["feature_names"]
        if all(c in X.columns for c in names):
            return list(names)
    return list(X.columns)


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 72)
    print("訓練 leave-one-profile-out 模型（變轉速模式，儀表板上線用）")
    print("=" * 72)

    X, y, meta = load_all_splits("all")
    cols = feature_columns(X)
    X = X[cols]
    print(f"  {len(X)} 個時間窗，{len(cols)} 個特徵（振動＋電流）")

    rows = []
    for p in PROFILES:
        te = (meta["profile"] == p).values
        X_tr, y_tr = X[~te], y[~te]
        X_te, y_te = X[te], y[te]
        classes = sorted(y_tr.unique())
        c2i = {c: i for i, c in enumerate(classes)}
        model = make_model(len(classes))
        model.fit(X_tr.values, y_tr.map(c2i).values)

        proba = model.predict_proba(X_te.values)
        pred = np.array([classes[i] for i in proba.argmax(1)])
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        auc = None
        if "Normal" in classes:
            ni = classes.index("Normal")
            try:
                auc = float(roc_auc_score((y_te != "Normal").astype(int).values,
                                          1.0 - proba[:, ni]))
            except ValueError:
                auc = None

        # 檔案層級：每支錄音多數決（與儀表板 infer_modality 相同的聚合方式）
        m_te = meta[te].reset_index(drop=True)
        file_ok = []
        for fid, g in m_te.groupby("file_id"):
            vc = pd.Series(pred[g.index.values]).value_counts()
            file_ok.append(vc.index[0] == y_te.iloc[g.index.values[0]])

        bundle = {
            "model": model,
            "classes": classes,
            "feature_names": cols,
            "dataset": "speed",
            "held_out_profile": p,
            "training_profiles": [q for q in PROFILES if q != p],
            "holdout_accuracy": float(acc),
            "holdout_macro_f1": float(f1),
            "holdout_auc": auc,
            "holdout_report": classification_report(y_te, pred, output_dict=True,
                                                    zero_division=0),
            "n_train": int(len(X_tr)),
            "n_holdout": int(len(X_te)),
        }
        path = MODELS_DIR / f"model_speed_lopo_{p}.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        auc_s = f"{auc:.3f}" if auc is not None else "—"
        print(f"  排除曲線 {p}：訓練 {len(X_tr):>5} / 測試 {len(X_te):>5} 窗"
              f" -> 逐窗準確率 {acc:.4f}  macro-F1 {f1:.4f}  二元AUC {auc_s}"
              f"  檔案層級 {sum(file_ok)}/{len(file_ok)}　已存 {path.name}")
        rows.append({"排除的轉速曲線": p, "逐窗準確率": round(acc, 4),
                     "macro_F1": round(f1, 4), "二元AUC": auc and round(auc, 4),
                     "檔案判對": int(sum(file_ok)), "檔案數": len(file_ok),
                     "訓練窗數": len(X_tr), "測試窗數": len(X_te)})

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "speed_lopo_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))
    print(f"\n七組平均：逐窗準確率 {df['逐窗準確率'].mean():.4f}　"
          f"macro-F1 {df['macro_F1'].mean():.4f}　"
          f"檔案層級 {df['檔案判對'].sum()}/{df['檔案數'].sum()}")
    print(f"已存 {out}\n完成。重新啟動 streamlit 就會自動改用這些模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
