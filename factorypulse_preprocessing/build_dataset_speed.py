"""
變轉速資料集的前處理主流程。

產出格式刻意與 build_dataset.py（變負載）保持一致，方便之後比較或合併：
    processed_speed/
        manifest.csv
        vibration_features_{train,val,test}.csv
        current_features_{train,val,test}.csv

欄位差異只有兩處：
    - 沒有 load_nm，改成 rpm（該窗的平均轉速）
    - condition 已對應成 Normal / BPFI / BPFO / Ball
      （原始檔名是 normal / inner / outer / ball）

用法：
    python build_dataset_speed.py                 # 處理全部 28 組
    python build_dataset_speed.py --profiles 0 1  # 只處理轉速曲線 0 和 1（快速試跑）
    python build_dataset_speed.py --skip-current  # 只做振動

⚠️ 資料量說明：每個振動 csv 約 600MB、電流 csv 約 1.1GB（純文字），
   共 56 個大檔。直接從 zip 串流讀取，不解壓縮（解開要 52GB）。
   全部跑完預估 20~40 分鐘，每處理完一個檔案會印進度。
   建議第一次先用 --profiles 0 試跑，確認流程正確再跑完整版。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features_current
import features_vibration
import speed_io
import windowing

SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}


def build_manifest(profiles) -> list[dict]:
    index = speed_io.build_zip_index()
    avail = speed_io.list_available(index)
    complete = avail[avail.vibration & avail.current & avail.rpm]

    entries = []
    for _, row in complete.iterrows():
        profile = row["profile"]
        if profile == "constant":
            continue
        if profiles is not None and int(profile) not in profiles:
            continue
        raw = row["condition_raw"]
        entries.append({
            "condition_raw": raw,
            "condition": config.SPEED_CONDITION_MAP[raw],
            "profile": int(profile),
            "file_id": f"{raw}_{profile}",
            # 這個資料集沒有嚴重度分級，統一給 1（正常給 0），
            # 保留這個欄位是為了跟變負載資料集的格式一致
            "severity_level": 0 if raw == "normal" else 1,
        })
    return entries


def process(entries, out_dir: Path, do_current: bool = True):
    index = speed_io.build_zip_index()
    fs = config.SPEED_VIBRATION_FS
    win_sec = config.WINDOW_SEC
    overlap = config.WINDOW_OVERLAP
    step = int(round(fs * win_sec * (1 - overlap)))

    vib_buckets = {s: [] for s in SPLIT_RATIO}
    cur_buckets = {s: [] for s in SPLIT_RATIO}

    total = len(entries)
    t0 = time.time()

    for idx, e in enumerate(entries, 1):
        fid = e["file_id"]
        print(f"[{idx}/{total}] {fid} 讀取振動 ...", end="", flush=True)
        vib = speed_io.load_vibration(e["condition_raw"], e["profile"], index)
        rpm_df = speed_io.load_rpm(e["condition_raw"], e["profile"], index)
        vib_fs = speed_io.infer_fs(vib.shape[0])
        if abs(vib_fs - fs) > 1:
            print(f"\n  ⚠️ {fid} 振動取樣率推算為 {vib_fs:,.0f} Hz，"
                  f"與設定的 {fs:,} Hz 不符，改用實測值", flush=True)
        fs_this = vib_fs
        step_this = int(round(fs_this * win_sec * (1 - overlap)))
        print(f" {vib.shape[0]:,} 點", end="", flush=True)

        # 依時間切 train/val/test，再各自切窗（與變負載資料集同樣的做法，
        # 避免幾乎重疊的窗同時落在訓練與測試集）
        chunks = windowing.split_signal_by_time(vib, SPLIT_RATIO)
        del vib

        offset = 0  # 這一段在整個訊號中的起始樣本數，用來換算絕對時間去查轉速
        n_win_total = 0
        for split in ["train", "val", "test"]:
            chunk = chunks[split]
            try:
                wins = windowing.sliding_window(chunk, fs_this, win_sec, overlap)
            except ValueError:
                offset += len(chunk)
                continue

            start_times = (offset + np.arange(len(wins)) * step_this) / fs_this
            rpms = speed_io.rpm_for_windows(rpm_df, start_times, win_sec)

            for i, w in enumerate(wins):
                feats = features_vibration.extract_classical_features(w, fs_this)
                feats.update({
                    "rpm": float(rpms[i]),
                    "condition": e["condition"],
                    "severity_level": e["severity_level"],
                    "profile": e["profile"],
                    "file_id": fid,
                    "split": split,
                })
                vib_buckets[split].append(feats)
            n_win_total += len(wins)
            offset += len(chunk)
            del wins
        del chunks
        print(f" -> 振動 {n_win_total} 窗", end="", flush=True)

        if do_current:
            print(" | 讀取電流 ...", end="", flush=True)
            # 電流原始取樣率 100kHz，load_current 內部會降採樣到 25kHz
            cur = speed_io.load_current(e["condition_raw"], e["profile"], index)
            cur_fs = speed_io.infer_fs(cur.shape[0])
            if abs(cur_fs - config.SPEED_CURRENT_FS) > 1:
                print(f"\n  ⚠️ {fid} 電流取樣率推算為 {cur_fs:,.0f} Hz，"
                      f"與設定的 {config.SPEED_CURRENT_FS:,} Hz 不符，改用實測值", flush=True)
            # 只用第一相（R），與變負載資料集的處理方式一致
            # （那邊 BPFO 檔案缺 V/W 相，所以統一只用單相，特徵欄位才對得起來）
            cur_r = cur[:, 0]
            del cur
            c_step = int(round(cur_fs * win_sec * (1 - overlap)))
            c_chunks = windowing.split_signal_by_time(cur_r, SPLIT_RATIO)
            del cur_r

            c_offset = 0
            n_cur = 0
            for split in ["train", "val", "test"]:
                chunk = c_chunks[split]
                try:
                    wins = windowing.sliding_window(chunk, cur_fs, win_sec, overlap)
                except ValueError:
                    c_offset += len(chunk)
                    continue
                start_times = (c_offset + np.arange(len(wins)) * c_step) / cur_fs
                rpms = speed_io.rpm_for_windows(rpm_df, start_times, win_sec)
                for i, w in enumerate(wins):
                    feats = features_current.extract_current_features(w, cur_fs)
                    feats.update({
                        "rpm": float(rpms[i]),
                        "condition": e["condition"],
                        "severity_level": e["severity_level"],
                        "profile": e["profile"],
                        "file_id": fid,
                        "split": split,
                    })
                    cur_buckets[split].append(feats)
                n_cur += len(wins)
                c_offset += len(chunk)
                del wins
            del c_chunks
            print(f" -> 電流 {n_cur} 窗", end="", flush=True)

        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx)
        print(f"  [已用 {elapsed/60:.1f} 分, 預估剩 {eta/60:.1f} 分]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in vib_buckets.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"vibration_features_{split}.csv", index=False)
            print(f"[vibration/{split}] {len(rows)} 筆已存")
    for split, rows in cur_buckets.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"current_features_{split}.csv", index=False)
            print(f"[current/{split}] {len(rows)} 筆已存")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=int, nargs="*", default=None,
                    help="只處理指定的轉速曲線編號（0~6），不指定就全部處理")
    ap.add_argument("--skip-current", action="store_true", help="只做振動，跳過電流")
    args = ap.parse_args()

    entries = build_manifest(args.profiles)
    if not entries:
        raise SystemExit("沒有符合條件的資料，請檢查 --profiles 參數或 zip 路徑設定")

    print("=" * 60)
    print(f"變轉速資料集前處理")
    print(f"  待處理組合：{len(entries)} 組")
    df = pd.DataFrame(entries)
    print(f"  類別分布：{df.condition.value_counts().to_dict()}")
    print(f"  轉速曲線：{sorted(df.profile.unique())}")
    print(f"  輸出目錄：{config.SPEED_OUTPUT_DIR.resolve()}")
    print("=" * 60)

    config.SPEED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_csv(config.SPEED_OUTPUT_DIR / "manifest.csv", index=False)

    process(entries, config.SPEED_OUTPUT_DIR, do_current=not args.skip_current)
    print("\n完成，輸出在", config.SPEED_OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
