"""
產生證據層的「正常基準檔」：results/evidence_reference.json
==========================================================

基準的定義：同一個負載下、Normal 狀態的特徵中位數。
證據層用它算出「這台機器的某項量測是正常的幾倍」。

為什麼要按負載分開算：負載對訊號的影響比故障還大（4Nm 正常的振動幅值
比 0Nm 不對心還高），拿全體平均當基準會讓高負載的正常機台被誤標成異常。

什麼時候要重跑：
  - 改了 features_vibration_v2 的特徵定義
  - 改了 evidence.FEATURE_PHYSICS 收錄的特徵
  - 換了資料集

執行：
    python build_reference.py
"""

import json
from pathlib import Path

import pandas as pd

import config
import evidence
import features_vibration_v2 as v2

OUT = Path("./results/evidence_reference.json")


def main():
    dfs = []
    for split in ["train", "val", "test"]:
        p = config.OUTPUT_DIR / f"vibration_v2_features_{split}.csv"
        if not p.exists():
            raise SystemExit(f"找不到 {p}，請先執行 build_dataset.py")
        dfs.append(pd.read_csv(p))
    v = pd.concat(dfs, ignore_index=True)

    # 收錄哪些欄位：直接取 evidence.FEATURE_PHYSICS 的 key，
    # 這樣兩邊永遠同步，不會出現「解釋層想用但基準檔沒有」的狀況
    # （先前 env_*_sb_ratio 就是這樣被漏掉的）。
    cols = [f"ch{ch}_{feat}"
            for ch in range(4)
            for feat in evidence.FEATURE_PHYSICS]
    cols = [c for c in cols if c in v.columns]

    missing = [f"ch0_{f}" for f in evidence.FEATURE_PHYSICS
               if f"ch0_{f}" not in v.columns]
    if missing:
        print(f"⚠️ FEATURE_PHYSICS 有欄位不在特徵檔裡，將被忽略：{missing}")

    normal = v[v.condition == "Normal"]
    if normal.empty:
        raise SystemExit("特徵檔裡沒有 Normal 樣本，無法建立基準")

    ref = {
        "note": "由 Normal 檔案逐負載算出的基準值，供證據層計算『幾倍於正常』",
        "shaft_freq_hz": 50.15,
        "bearing_orders": dict(v2.BEARING_ORDERS),
        "n_features": len(cols),
        "baseline": {},
    }
    for load, g in normal.groupby("load_nm"):
        ref["baseline"][str(int(load))] = {
            c: {"median": float(g[c].median()), "p95": float(g[c].quantile(0.95))}
            for c in cols
        }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"已寫入 {OUT}")
    print(f"  負載：{list(ref['baseline'])}")
    print(f"  特徵：{len(cols)} 欄（{len(evidence.FEATURE_PHYSICS)} 種量測 × 4 通道）")
    print(f"  軸承階次：{ref['bearing_orders']}")

    # 驗證：每個故障類別是否都能點亮對應的證據
    print("\n各故障類別在 4Nm 下的前 3 項證據（驗證對角線是否乾淨）：")
    for cond in ["BPFO", "BPFI", "Misalign", "Unbalance", "Normal"]:
        g = v[(v.condition == cond) & (v.load_nm == 4)]
        if g.empty:
            continue
        row = g.median(numeric_only=True).to_dict()
        ev = evidence.extract(row, 4, top=3, reference=ref, fault_type=cond)
        items = "、".join(f"{e['名稱']}({e['倍數']}x)" for e in ev) or "無明顯偏離"
        print(f"  {cond:10s} {items}")


if __name__ == "__main__":
    main()
