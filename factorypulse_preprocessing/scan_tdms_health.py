"""
掃描 current,temp/ 底下所有 tdms 檔案，檢查每個 channel 的實際長度。

起因：0Nm_BPFO_03.tdms 被發現「三相電流只有 U 相有資料，V/W 相長度是 0」，
導致那 9 個 BPFO 檔案的電流特徵全部被跳過。
這支腳本用來確認：到底有幾個檔案有缺相問題、缺哪幾相、錄音長度各是多少。

用法（在 factorypulse_preprocessing 資料夾裡執行）：
    python scan_tdms_health.py

輸出一張表，並在最後印出摘要與建議。
"""

from pathlib import Path

import pandas as pd
from nptdms import TdmsFile

import config


def scan_one(path: Path) -> dict:
    tdms = TdmsFile.read_metadata(str(path))  # 只讀 metadata，不載入資料，很快
    temp_lens, cur_lens = [], []
    for group in tdms.groups():
        for ch in group.channels():
            unit = ch.properties.get("unit_string", "")
            n = len(ch)
            if unit in ("degC", "°C", "C"):
                temp_lens.append((ch.name, n))
            elif unit in ("A", "Amp", "Ampere"):
                cur_lens.append((ch.name, n))

    temp_lens.sort()
    cur_lens.sort()
    fs = None
    for group in tdms.groups():
        for ch in group.channels():
            inc = ch.properties.get("wf_increment")
            if inc:
                fs = 1.0 / inc
                break
        if fs:
            break

    n_cur_ok = sum(1 for _, n in cur_lens if n > 0)
    n_temp_ok = sum(1 for _, n in temp_lens if n > 0)
    max_len = max([n for _, n in cur_lens + temp_lens] or [0])

    return {
        "file_id": path.stem,
        "溫度channel數": len(temp_lens),
        "溫度有資料": n_temp_ok,
        "電流channel數": len(cur_lens),
        "電流有資料": n_cur_ok,
        "最長長度": max_len,
        "秒數": round(max_len / fs, 1) if fs else None,
        "電流各相長度": [n for _, n in cur_lens],
    }


def main():
    files = sorted(config.CURRENT_TEMP_DIR.glob("*.tdms"))
    print(f"掃描 {len(files)} 個 tdms 檔案（只讀 metadata，很快）...\n")

    rows = []
    for i, p in enumerate(files, 1):
        print(f"  ({i}/{len(files)}) {p.stem}", end="\r", flush=True)
        try:
            rows.append(scan_one(p))
        except Exception as ex:
            rows.append({"file_id": p.stem, "錯誤": f"{type(ex).__name__}: {ex}"})

    df = pd.DataFrame(rows)
    print(" " * 60, end="\r")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    bad = df[df["電流有資料"] < 3]
    if len(bad):
        print(f"⚠️ 有 {len(bad)} 個檔案的三相電流不完整（少於 3 相有資料）：")
        for _, r in bad.iterrows():
            print(f"   {r['file_id']}: 電流各相長度 = {r['電流各相長度']}")
    else:
        print("✅ 所有檔案三相電流都完整")

    print()
    print("秒數分布：")
    print(df["秒數"].value_counts().sort_index().to_string())
    df.to_csv("tdms_health_report.csv", index=False, encoding="utf-8-sig")
    print("\n完整報告已存成 tdms_health_report.csv")


if __name__ == "__main__":
    main()
