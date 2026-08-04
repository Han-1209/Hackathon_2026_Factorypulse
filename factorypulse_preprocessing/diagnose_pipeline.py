"""
FactoryPulse 端到端診斷流程

    一台設備的感測資料
        ↓  載入已訓練模型
    振動模型 / 電流模型 各自逐窗推論
        ↓  聚合成設備層級的異常機率與故障類別
    溫度趨勢（從前處理特徵直接讀，不經模型）
        ↓  規則引擎交叉驗證
    結構化診斷結果（健康分數 / 風險等級 / 是否停機 / 建議行動）

用法：
    python diagnose_pipeline.py                    # 診斷所有 45 台設備，輸出總覽
    python diagnose_pipeline.py --file 4Nm_BPFO_30 # 單台設備的完整診斷
    python diagnose_pipeline.py --json 4Nm_BPFO_30 # 輸出 JSON（給前端 / LLM 用）

前置作業：
    先跑過 python train_baseline.py（它會把模型存到 ./models/）
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config
import evidence
import feature_selection as fs
from rule_engine import ModalityInput, TemperatureInput, diagnose

MODELS_DIR = Path("./models")


# ---------------------------------------------------------------- 模型載入
def load_model(modality: str) -> dict:
    path = MODELS_DIR / f"model_{modality}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。請先執行 `python train_baseline.py` 訓練並儲存模型。"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- 逐窗推論
def infer_modality(bundle: dict, X: pd.DataFrame) -> dict:
    """對一台設備的所有窗做推論，聚合成單一結果。

    聚合方式的取捨：
      - 異常機率用「中位數」而不是平均。單一個窗可能因為瞬間干擾出現極端值，
        中位數對這種離群窗比較穩，符合工業現場「持續異常才算異常」的直覺。
      - 故障類別用「多數決」，並回報該類別佔了多少比例當作可信度。
        比例低代表模型在不同時間窗之間搖擺不定，這種情況應該提醒人工複檢，
        而不是硬給一個答案。
    """
    model = bundle["model"]
    classes = bundle["classes"]
    X = X[bundle["feature_names"]]  # 對齊訓練時的欄位順序

    proba = model.predict_proba(X.values)
    normal_idx = classes.index("Normal") if "Normal" in classes else None

    # 異常機率 = 1 - P(正常)
    if normal_idx is not None:
        anomaly = 1.0 - proba[:, normal_idx]
    else:
        anomaly = proba.max(axis=1)

    pred_idx = proba.argmax(axis=1)
    pred_labels = [classes[i] for i in pred_idx]
    vc = pd.Series(pred_labels).value_counts()
    top_label = vc.index[0]
    agreement = float(vc.iloc[0] / len(pred_labels))

    # 若多數決結果是 Normal，但異常機率中位數偏高，改回報次高的故障類別，
    # 避免「大部分窗正常、少數窗嚴重異常」被整段判成健康（早期故障常是這樣）
    median_anomaly = float(np.median(anomaly))
    if top_label == "Normal" and median_anomaly >= 0.5 and len(vc) > 1:
        top_label = vc.index[1]

    return {
        "anomaly_prob": median_anomaly,
        "anomaly_p90": float(np.percentile(anomaly, 90)),
        "fault_type": top_label,
        "confidence": agreement,
        "n_windows": int(len(X)),
        "n_features": int(X.shape[1]),   # 給畫面顯示用，不要在 UI 寫死特徵數
        "vote_distribution": vc.to_dict(),
    }


# ---------------------------------------------------------------- 溫度讀取
def get_temperature(file_id: str, split: str) -> TemperatureInput:
    """溫度不經模型，直接讀前處理算好的趨勢特徵。"""
    path = config.OUTPUT_DIR / f"temperature_features_{split}.csv"
    df = pd.read_csv(path)
    row = df[df.file_id == file_id]
    if row.empty:
        # 沒有溫度資料時回傳中性值，讓規則引擎照常運作而不是整個掛掉
        return TemperatureInput(delta_from_baseline=0.0, trend_slope=0.0)
    r = row.iloc[0]
    # 取 A/B 兩個測點中較嚴重的那個（保守做法：寧可高估溫升）
    delta = max(r["A_temp_delta_from_baseline_mean"], r["B_temp_delta_from_baseline_mean"])
    slope = max(r["A_temp_trend_slope"], r["B_temp_trend_slope"])
    return TemperatureInput(delta_from_baseline=float(delta), trend_slope=float(slope))


# ---------------------------------------------------------------- 主流程
def diagnose_equipment(file_id: str, split: str = "test", models: dict | None = None) -> dict:
    """對單一設備跑完整診斷，回傳可直接序列化的 dict。"""
    if models is None:
        models = {m: load_model(m) for m in ["vibration", "current"]}

    modality_results = {}
    vib_features = None
    for mod in ["vibration", "current"]:
        # ⚠️ 模型是用哪一版振動特徵訓練的，就必須載入哪一版的特徵檔。
        # model_vibration.pkl 現在預設是 v2（176 欄），若這裡還去讀 v1 的
        # vibration_features_*.csv（40 欄），X[bundle["feature_names"]]
        # 會直接 KeyError。bundle["feature_version"] 由 train_baseline 寫入。
        src = mod
        if mod == "vibration" and models[mod].get("feature_version") == "v2":
            src = "vibration_v2"

        X, y, meta = fs.load_features(split, src)
        mask = meta.file_id == file_id
        if not mask.any():
            raise ValueError(f"在 {split} 資料集裡找不到設備 {file_id}")
        modality_results[mod] = infer_modality(models[mod], X[mask.values])
        true_label = y[mask.values].iloc[0]
        load_nm = meta.loc[mask, "load_nm"].iloc[0]

        # 留一份這台設備的振動特徵中位數，給證據層算「幾倍於正常」。
        # 用中位數而不是平均，理由同 infer_modality：對瞬間干擾的窗比較穩。
        if mod == "vibration":
            vib_features = X[mask.values].median(numeric_only=True).to_dict()

    temp = get_temperature(file_id, split)

    vib = modality_results["vibration"]
    cur = modality_results["current"]
    d = diagnose(
        vibration=ModalityInput(vib["anomaly_prob"], vib["fault_type"], vib["confidence"]),
        current=ModalityInput(cur["anomaly_prob"], cur["fault_type"], cur["confidence"]),
        temperature=temp,
        load_nm=float(load_nm),
    )

    out = d.to_dict()
    out.update({
        "equipment_id": file_id,
        "load_nm": int(load_nm),
        "ground_truth": true_label,          # demo 用：對照實際標籤
        "correct": d.probable_fault == true_label,
        "modality_detail": modality_results,
        "temperature_input": {
            "delta_from_baseline": temp.delta_from_baseline,
            "trend_slope": temp.trend_slope,
        },
    })

    # ---- 物理證據：把特徵翻成「這台機器實際量到什麼、是正常的幾倍」----
    # 這是 LLM 解釋層的輸入。沒有這段，LLM 只能按故障類別去念知識庫通論，
    # 講不出任何屬於這台設備的數字。
    out["evidence_features"] = vib_features or {}
    try:
        out["physical_evidence"] = evidence.extract(
            vib_features or {}, load_nm=float(load_nm), top=5,
            fault_type=d.probable_fault)
    except FileNotFoundError:
        out["physical_evidence"] = []
        print("  ⚠️ 找不到 results/evidence_reference.json，證據層停用。"
              "跑 python evidence.py 可以確認基準檔是否存在。")
    return out


def print_report(r: dict):
    print("=" * 72)
    print(f"設備 {r['equipment_id']}　負載 {r['load_nm']}Nm")
    print("=" * 72)
    print(f"  健康分數：{r['health_scores']['綜合健康']}/100"
          f"　風險等級：{r['risk_level_display']}"
          f"　處理時限：{r['action_window']}")
    print(f"  是否立即停機：{'是' if r['should_stop'] else '否'}")
    print(f"  研判故障：{r['probable_fault_display']}"
          f"　（實際：{r['ground_truth']}　{'✓ 正確' if r['correct'] else '✗ 誤判'}）")
    print()
    print("  四面向健康分數：")
    for k, v in r["health_scores"].items():
        bar = "█" * int(v / 5)
        print(f"    {k:6s} {v:3d}  {bar}")
    print()
    print("  多感測證據：")
    for e in r["evidence"]:
        print(f"    [{e['來源']}] {e['狀態']}　{e['數值']}")
        print(f"           {e['說明']}")

    if r.get("physical_evidence"):
        print("\n  物理證據（與同負載正常機台比較）：")
        for e in r["physical_evidence"]:
            print(f"    {e['名稱']} @ {e['位置']}：{e['倍數']} 倍於正常"
                  f"（實測 {e['實測值']} / 基準 {e['正常基準']}）")
    if r["triggered_rules"]:
        print("\n  觸發規則：")
        for rule in r["triggered_rules"]:
            print(f"    - {rule}")
    if r["causes"]:
        print(f"\n  可能原因：{'、'.join(r['causes'])}")
        print(f"  建議檢查：{'、'.join(r['checks'])}")
        print(f"  建議行動：{'、'.join(r['actions'])}")
    print()


def run_all(split: str = "test"):
    """對所有設備跑診斷，輸出工廠總覽（給儀表板首頁用）。"""
    models = {m: load_model(m) for m in ["vibration", "current"]}
    manifest = pd.read_csv(config.OUTPUT_DIR / "manifest.csv")

    rows = []
    for fid in manifest.file_id:
        try:
            r = diagnose_equipment(fid, split, models)
        except Exception as ex:
            print(f"  ⚠️ {fid} 診斷失敗：{type(ex).__name__}: {ex}")
            continue
        rows.append({
            "設備": fid,
            "負載": r["load_nm"],
            "健康分數": r["health_scores"]["綜合健康"],
            "風險等級": r["risk_level_display"],
            "研判故障": r["probable_fault_display"],
            "實際狀態": r["ground_truth"],
            "正確": "✓" if r["correct"] else "✗",
            "立即停機": "是" if r["should_stop"] else "",
        })

    df = pd.DataFrame(rows).sort_values("健康分數")
    print("=" * 88)
    print(f"工廠總覽（{split} 資料集，共 {len(df)} 台設備）")
    print("=" * 88)
    print(df.to_string(index=False))

    print("\n--- 統計 ---")
    print(f"  設備層級診斷正確率：{(df['正確'] == '✓').mean():.1%}")
    print(f"  風險等級分布：{df['風險等級'].value_counts().to_dict()}")
    print(f"  建議立即停機：{(df['立即停機'] == '是').sum()} 台")

    out = Path("./results/factory_overview.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n總覽已存成 {out.resolve()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="診斷單一設備，例如 4Nm_BPFO_30")
    ap.add_argument("--json", help="輸出單一設備的 JSON（給前端/LLM 用）")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    if args.json:
        r = diagnose_equipment(args.json, args.split)
        r.pop("modality_detail", None)  # JSON 給 LLM 時不需要這麼細
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.file:
        print_report(diagnose_equipment(args.file, args.split))
    else:
        run_all(args.split)


if __name__ == "__main__":
    main()
