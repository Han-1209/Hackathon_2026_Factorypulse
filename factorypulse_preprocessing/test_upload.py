"""
繞過 Streamlit，直接測試上傳轉檔流程，把完整 traceback 印出來。

Streamlit 的例外有時只顯示最後一行，看不到真正的出錯位置。
這支腳本模擬使用者上傳，走的是完全相同的程式路徑
（upload_ingest.ingest -> 特徵計算 -> 對齊模型欄位 -> 推論）。

執行：
    python test_upload.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config


class FakeUpload:
    """模擬 Streamlit 的 UploadedFile：需要 .name 與 .getbuffer()。"""

    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()

    def getbuffer(self):
        return self._data


def step(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    vib = config.VIBRATION_DIR / "4Nm_BPFO_30.mat"
    tdms = config.CURRENT_TEMP_DIR / "4Nm_BPFO_30.tdms"

    step("0. 環境檢查")
    for name in ["scipy", "nptdms", "sklearn", "xgboost", "numpy", "pandas"]:
        try:
            __import__(name)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} 未安裝 —— {e}")

    print(f"\n  振動檔 {vib}\n    存在？{vib.exists()}")
    print(f"  電流檔 {tdms}\n    存在？{tdms.exists()}")

    import dashboard_data as dd
    import upload_ingest as ui

    mode = "load"
    step("1. 檢查模型與特徵版本")
    try:
        src = dd.vibration_source(mode)
        bundle = dd.load_model(mode, "vibration")
        print(f"  vibration_source(load) = {src}")
        print(f"  model feature_version  = {bundle.get('feature_version')}")
        print(f"  模型需要 {len(bundle['feature_names'])} 個特徵")
        print(f"  前 5 個：{bundle['feature_names'][:5]}")
        print(f"  含 load_nm？{'load_nm' in bundle['feature_names']}")
    except Exception:
        traceback.print_exc()
        return

    fv = "v2" if src == "vibration_v2" else "v1"

    # ---------------------------------------------------------------- 振動
    step("2. 上傳 .mat 振動檔")
    if not vib.exists():
        print("  跳過：檔案不存在")
    else:
        try:
            res = ui.ingest(FakeUpload(vib), feature_version=fv, load_nm=4)
            X = res["features"]
            print(f"  ✓ 轉檔成功：{res['note']}")
            print(f"    特徵表 shape = {X.shape}")

            missing = [c for c in bundle["feature_names"] if c not in X.columns]
            extra = [c for c in X.columns if c not in bundle["feature_names"]]
            print(f"    模型缺少的欄位：{len(missing)} 個 {missing[:5]}")
            print(f"    多出來的欄位：  {len(extra)} 個 {extra[:5]}")

            if not missing:
                out = dd.infer_modality(bundle, X)
                print(f"  ✓ 推論成功")
                print(f"    研判故障 = {out['fault_type']}"
                      f"（實際應為 BPFO）")
                print(f"    異常機率 = {out['anomaly_prob']:.3f}"
                      f"　一致率 = {out['confidence']:.0%}")

                llm_diag = dd.build_llm_diag(mode, out, X)
                print(f"    物理證據 {len(llm_diag['evidence'])} 項")
                for e in llm_diag["evidence"][:3]:
                    print(f"      - {e['名稱']} @ {e['位置']}：{e['倍數']} 倍")
            else:
                print("  ✗ 欄位對不上，無法推論")
        except Exception:
            print("  ✗ 失敗，完整 traceback：\n")
            traceback.print_exc()

    # ---------------------------------------------------------------- 電流
    step("3. 上傳 .tdms 電流溫度檔")
    if not tdms.exists():
        print("  跳過：檔案不存在")
    else:
        try:
            res = ui.ingest(FakeUpload(tdms))
            print(f"  ✓ 轉檔成功：{res['note']}")
            print(f"    kind = {res['kind']}（預期 'current'，畫面會提示改上傳 .mat）")
            print(f"    特徵表 shape = {res['features'].shape}")
            if res["temperature"]:
                print(f"    溫度摘要欄位：{list(res['temperature'])}")
                # st.json 對 numpy 型別會出錯，這裡先驗證型別
                bad = {k: type(v).__name__ for k, v in res["temperature"].items()
                       if isinstance(v, (np.generic, np.ndarray))}
                if bad:
                    print(f"    ⚠️ 有 numpy 型別，st.json 可能出錯：{bad}")
                else:
                    print(f"    ✓ 皆為原生 Python 型別，st.json 可正常顯示")
        except Exception:
            print("  ✗ 失敗，完整 traceback：\n")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("測試結束。有 ✗ 的話把上面整段貼回去。")
    print("=" * 70)


if __name__ == "__main__":
    main()
