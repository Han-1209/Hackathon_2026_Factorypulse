"""
主流程：把振動 / 電流 / 溫度三個模態的原始檔案，整理成訓練用的資料集
（聲音因為資料量不足 45 個檔案的完整集合，預設不處理，見下方「聲音模態」說明）。

執行前請先：
  1. pip install scipy nptdms librosa numpy pandas scikit-learn
  2. 跑一次 `python inspect_tdms.py <任一個 tdms 檔案>`，核對 io_utils.py 裡的
     欄位比對邏輯跟你資料集實際的 channel 名稱一致。
  3. 確認 config.py 裡的 DATA_ROOT 指到你放資料的資料夾。

用法：
    python build_dataset.py                    # 處理振動/電流/溫度三個模態（預設不含聲音）
    python build_dataset.py --include-acoustic  # 聲音資料補齊到 45 個檔案後才加這個 flag

輸出（存在 config.OUTPUT_DIR 底下）：
  manifest.csv         每個原始檔案的標籤 (load / condition / severity / severity_level)
  vibration_windows_{split}.npz   raw windows，給 1D CNN，key = X (n,win_len,4), y_condition, y_severity
  vibration_features_{split}.csv  手工特徵，給 RandomForest/XGBoost baseline
  current_features_{split}.csv    三相電流 MCSA 特徵
  temperature_features_{split}.csv
  acoustic_specs_{split}.npz      (只有加 --include-acoustic 才會產生) mel spectrogram

train/val/test 是「同一個檔案內部依時間先後切」（前 70% train / 中間 15% val /
後 15% test），不是整檔分組，也不是隨機抽窗——因為這個資料集每個
(load, condition, severity) 組合只有一個檔案，整檔分組會讓某些類別整個不在
訓練或測試集裡；隨機抽窗則會讓幾乎重疊的窗同時落在 train 跟 test，造成
data leakage。細節見 split_by_file() 的說明。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import io_utils
import label_utils
import windowing
import features_vibration
import features_vibration_v2
import features_current
import features_temperature

try:
    import features_acoustic
except Exception:
    features_acoustic = None


def _index_by_label(directory: Path, glob_pattern: str) -> dict:
    """把資料夾裡的檔案依「解析後的標籤」(load, condition, severity) 建索引，
    而不是直接比對檔名字串 —— 因為原始資料集裡 2Nm 的振動檔名有拼字錯誤
    ("Unbalalnce")，如果直接用檔名去對應 current,temp/ 資料夾，
    那 5 個 2Nm_Unbalance_*.tdms 會全部對不到，process 時會被誤判成缺檔。
    """
    index = {}
    for p in sorted(directory.glob(glob_pattern)):
        try:
            label = label_utils.parse_filename(p)
        except label_utils.LabelParseError:
            continue
        key = (label["load_nm"], label["condition"], label["severity"])
        index[key] = p
    return index


def build_manifest() -> list[dict]:
    """掃描 vibration/ 資料夾當作『這個檔案存在』的基準名單，
    current_temp / acoustic 用標籤（load, condition, severity）去比對，
    不是用檔名字串直接比對（見 _index_by_label 的說明）。
    """
    ct_index = _index_by_label(config.CURRENT_TEMP_DIR, "*.tdms")
    acoustic_index = _index_by_label(config.ACOUSTIC_DIR, "*.mat")

    entries = []
    for p in sorted(config.VIBRATION_DIR.glob("*.mat")):
        label = label_utils.parse_filename(p)
        key = (label["load_nm"], label["condition"], label["severity"])
        tdms_path = ct_index.get(key)
        acoustic_path = acoustic_index.get(key)
        entries.append(
            {
                **label,
                "vibration_path": str(p),
                "current_temp_path": str(tdms_path) if tdms_path else None,
                "acoustic_path": str(acoustic_path) if acoustic_path else None,
            }
        )
    return entries


def split_by_file(entries: list[dict], train=0.7, val=0.15) -> None:
    """⚠️ 這個資料集每個 (condition, severity, load) 組合本來就只有『一個檔案』，
    所以不能像一般狀況那樣『整個檔案分去 train 或 test』——那樣每一種故障條件
    就只會出現在單一個 split 裡（要嘛全部在 train，要嘛全部在 test），
    模型會學不到某些類別，或是測試集根本沒有某些類別可以驗證。

    正確做法是『同一個檔案內部按時間切』：每個檔案的訊號依時間順序切成
    train/val/test 三段（不重疊、按時間先後，不是隨機抽窗），再各自切窗。
    這樣可以確保每個類別在三個 split 都有出現，同時窗與窗之間的切分點
    仍然是乾淨的時間邊界，不會讓幾乎重疊的兩個窗一個在 train 一個在 test。

    這個函式改成：不對 entries 本身分 split，而是回傳每個檔案的時間切分比例，
    真正的切分在 process_* 函式裡對訊號陣列做。
    """
    for e in entries:
        e["split_ratio"] = {"train": train, "val": val, "test": 1 - train - val}


def normal_temp_baseline(entries: list[dict]) -> dict:
    """對每個負載算出 Normal 狀態的平均溫度，當作該負載下的基準值。"""
    baseline = {}
    for e in entries:
        if e["condition"] != "Normal":
            continue
        temps = io_utils.load_current_temp_tdms(Path(e["current_temp_path"]))
        temp_1hz_a = features_temperature.resample_to_seconds(temps["temperature_A"], temps["fs"])
        temp_1hz_b = features_temperature.resample_to_seconds(temps["temperature_B"], temps["fs"])
        baseline[e["load_nm"]] = {
            "A": float(np.mean(temp_1hz_a)),
            "B": float(np.mean(temp_1hz_b)),
        }
    return baseline


def process_vibration(entries, out_dir: Path, raw_windows: str = "float16",
                      feature_version: str = "v2", rpm: float = 3010.0):
    """feature_version 控制用哪一版振動特徵，兩版的輸出檔名不同、可以並存：

        "v1" -> features_vibration.extract_classical_features
                輸出 vibration_features_{split}.csv       （40 欄）
        "v2" -> features_vibration_v2.extract_features_v2
                輸出 vibration_v2_features_{split}.csv    （207 欄）

    ⚠️ v1 有一個致命缺陷，別再拿它的數字對外報告：
        它的頻域特徵只有三個粗頻帶 + dominant_freq，而轉頻 50Hz 的
        1x/2x（不平衡/不對心的唯一物理判準）全被壓在 0-500Hz 這一個
        佔總能量僅 0.1% 的頻帶裡。實測結果是 Normal 與 Unbalance 的
        dominant_freq 與低頻佔比「數值完全相同」，模型根本沒有資訊可用。
        跨負載測試時 4Nm 的 Normal 有 100% 被判成 Unbalance，就是這個原因。
        v1 保留著只為了在報告裡做「改善前 vs 改善後」的對照。

    rpm: 轉軸轉速。verify_v2.py 在 5 個檔案上實測為 3009~3012 RPM，
         用固定值 3010 即可；傳 None 會改成每個窗自動偵測（慢一點但更保險）。

    raw_windows 控制「給 1D CNN 用的原始波形」要怎麼存：
        "none"    -> 完全不存，只算手工特徵（最快、最省記憶體，先跑 XGBoost baseline 用這個）
        "float16" -> 存成半精度（預設，容量減半；波形已 z-score 標準化，精度夠 CNN 用）
        "float32" -> 存成單精度（最完整，但很吃記憶體與硬碟）

    ⚠️ 為什麼要有這個選項：45 個檔案會切出約 5700 個窗，每個窗 25600 點 × 4 通道。
    用 float32 全部堆在記憶體裡約 2.3 GB，np.concatenate 時尖峰會到 4.7 GB，
    再加上 savez_compressed 對 GB 級陣列做壓縮會非常慢（可能十幾分鐘沒有任何輸出）。
    所以預設改成 float16 + 不壓縮存檔，並且全程印進度。
    """
    buckets = {s: {"X": [], "y_cond": [], "y_sev": [], "feat_rows": []} for s in ["train", "val", "test"]}
    total = len(entries)

    if feature_version == "v2":
        # 在算任何特徵之前先確認：不同名字的特徵沒有在量同一條譜線。
        # 若這裡報重疊，SHAP 的歸因會張冠李戴，解釋層就會講錯故障類型。
        features_vibration_v2.check_order_overlaps((rpm or 3010.0) / 60.0)

        def extract(w):
            return features_vibration_v2.extract_features_v2(
                w, config.VIBRATION_FS, rpm=rpm)
        out_prefix = "vibration_v2_features"
    elif feature_version == "v1":
        def extract(w):
            return features_vibration.extract_classical_features(w, config.VIBRATION_FS)
        out_prefix = "vibration_features"
    else:
        raise ValueError(f"未知的 feature_version: {feature_version}")
    print(f"[vibration] 特徵版本 = {feature_version}，輸出檔名 {out_prefix}_*.csv", flush=True)

    for idx, e in enumerate(entries, 1):
        print(f"[vibration] ({idx}/{total}) 讀取 {e['file_id']} ...", end="", flush=True)
        sig = io_utils.load_vibration_mat(Path(e["vibration_path"]))  # (N,4)
        chunks = windowing.split_signal_by_time(sig, e["split_ratio"])
        del sig  # 245MB 的原始陣列用完就放掉，不然 45 個檔案累積起來會爆記憶體

        n_win_this_file = 0
        for split, chunk in chunks.items():
            try:
                windows = windowing.sliding_window(chunk, config.VIBRATION_FS, config.WINDOW_SEC, config.WINDOW_OVERLAP)
            except ValueError:
                continue  # 這段時間切出來太短，切不出一個完整窗，正常會發生在 test 段

            if raw_windows != "none":
                cnn_ready = features_vibration.prepare_cnn_input(windows)
                if raw_windows == "float16":
                    cnn_ready = cnn_ready.astype(np.float16)
                buckets[split]["X"].append(cnn_ready)

            buckets[split]["y_cond"] += [e["condition"]] * len(windows)
            buckets[split]["y_sev"] += [e["severity_level"]] * len(windows)
            for w in windows:
                feats = extract(w)
                feats.update({"condition": e["condition"], "severity_level": e["severity_level"], "load_nm": e["load_nm"], "file_id": e["file_id"], "split": split})
                buckets[split]["feat_rows"].append(feats)
            n_win_this_file += len(windows)
            del windows
        del chunks
        print(f" 切出 {n_win_this_file} 個窗", flush=True)

    for split, b in buckets.items():
        if not b["feat_rows"]:
            continue
        # 手工特徵先存（檔案小、寫得快，這是接下來跑 XGBoost baseline 真正要用的東西）
        df = pd.DataFrame(b["feat_rows"])
        df.to_csv(out_dir / f"{out_prefix}_{split}.csv", index=False)
        print(f"[vibration/{split}] {out_prefix}_{split}.csv 已存，"
              f"{len(df)} 筆 × {df.shape[1]} 欄", flush=True)

        if raw_windows != "none" and b["X"]:
            print(f"[vibration/{split}] 正在合併並寫入原始波形 npz（檔案較大，請稍候）...", end="", flush=True)
            X = np.concatenate(b["X"], axis=0)
            b["X"].clear()  # 釋放暫存 list，降低尖峰記憶體
            # 用 savez（不壓縮）而不是 savez_compressed：GB 級陣列做 zlib 壓縮會慢到讓人以為當機
            np.savez(
                out_dir / f"vibration_windows_{split}.npz",
                X=X,
                y_condition=np.array(b["y_cond"]),
                y_severity=np.array(b["y_sev"]),
            )
            print(f" 完成，shape={X.shape}, dtype={X.dtype}", flush=True)
            del X


def process_current_and_temperature(entries, out_dir: Path, baseline: dict):
    """電流與溫度合併在同一個迴圈處理。

    ⚠️ 原本拆成 process_current() 和 process_temperature() 兩個函式，
    會導致每個 300MB 的 tdms 檔案被完整讀取兩次（再加上算 baseline 那次共三次），
    45 個檔案就是多讀了 27GB。合併成一次讀取、同時算兩種特徵，速度快很多。
    """
    cur_buckets = {s: [] for s in ["train", "val", "test"]}
    tmp_buckets = {s: [] for s in ["train", "val", "test"]}
    targets = [e for e in entries if e["current_temp_path"] is not None]
    total = len(targets)

    for idx, e in enumerate(targets, 1):
        print(f"[current+temp] ({idx}/{total}) 讀取 {e['file_id']} ...", end="", flush=True)
        data = io_utils.load_current_temp_tdms(Path(e["current_temp_path"]))
        fs = data["fs"]

        # ---- 溫度 ----
        temp_1hz_a = features_temperature.resample_to_seconds(data["temperature_A"], fs)
        temp_1hz_b = features_temperature.resample_to_seconds(data["temperature_B"], fs)
        base = baseline.get(e["load_nm"], {"A": float(temp_1hz_a.mean()), "B": float(temp_1hz_b.mean())})
        chunks_a = windowing.split_signal_by_time(temp_1hz_a, e["split_ratio"])
        chunks_b = windowing.split_signal_by_time(temp_1hz_b, e["split_ratio"])
        for split in ["train", "val", "test"]:
            if len(chunks_a[split]) < 2:
                continue
            feats_a = {f"A_{k}": v for k, v in features_temperature.extract_temperature_features(chunks_a[split], base["A"]).items()}
            feats_b = {f"B_{k}": v for k, v in features_temperature.extract_temperature_features(chunks_b[split], base["B"]).items()}
            tmp_buckets[split].append({
                **feats_a, **feats_b,
                "condition": e["condition"], "severity_level": e["severity_level"],
                "load_nm": e["load_nm"], "file_id": e["file_id"], "split": split,
            })

        # ---- 電流（只用 U 相，原因見 features_current.py 開頭說明）----
        n_phases = data.get("available_phases", 3)
        chunks_u = windowing.split_signal_by_time(data["current_U"], e["split_ratio"])
        del data  # 300MB 用完就放掉

        n_win_this_file = 0
        for split in ["train", "val", "test"]:
            try:
                u = windowing.sliding_window(chunks_u[split], fs, config.WINDOW_SEC, config.WINDOW_OVERLAP)
            except ValueError as ex:
                # 不要靜靜跳過！之前這裡直接 continue，導致 9 個 BPFO 檔案的電流特徵
                # 全部消失卻沒有任何提示，等到訓練前才發現少了一整個故障類別。
                print(
                    f"\n  ⚠️ [{e['file_id']}/{split}] 電流切窗失敗，這段資料被跳過："
                    f"{ex}\n     （該段長度={len(chunks_u[split])} 點，"
                    f"一個窗需要 {int(round(config.WINDOW_SEC * fs))} 點）",
                    flush=True,
                )
                continue
            for i in range(len(u)):
                feats = features_current.extract_current_features(u[i], fs)
                feats.update({
                    "condition": e["condition"], "severity_level": e["severity_level"],
                    "load_nm": e["load_nm"], "file_id": e["file_id"], "split": split,
                })
                cur_buckets[split].append(feats)
            n_win_this_file += len(u)
            del u
        del chunks_u
        phase_note = "" if n_phases == 3 else f"（此檔只有 {n_phases} 相有資料）"
        print(f" 電流切出 {n_win_this_file} 個窗{phase_note}", flush=True)

    for split, rows in cur_buckets.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"current_features_{split}.csv", index=False)
            print(f"[current/{split}] {len(rows)} windows saved", flush=True)
    for split, rows in tmp_buckets.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"temperature_features_{split}.csv", index=False)
            print(f"[temperature/{split}] {len(rows)} segments saved", flush=True)


def process_acoustic(entries, out_dir: Path):
    # ⚠️ 這條路徑目前不可用，而且不是因為缺套件。
    # features_acoustic.py 這支模組並不存在（從未實作），上面的 import 被
    # try/except 吞掉之後 features_acoustic 會是 None。
    # 原本的訊息寫「librosa 未安裝」會把人引導到錯誤的方向去查。
    #
    # 保留這個函式與 --include-acoustic 參數的理由：聲音資料只有 5/45 個檔案
    # （0Nm 下的 BPFI/BPFO/Normal），資料量不足以訓練獨立模態，
    # 是有意識的取捨而非遺漏。若日後把資料補齊，接回來的位置就在這裡。
    if features_acoustic is None:
        print("[acoustic] 略過：features_acoustic.py 尚未實作。\n"
              "           聲音資料目前只有 5/45 個檔案，不足以訓練獨立模態，\n"
              "           因此這個模態被列為未來擴充方向，並非缺少套件。")
        return
    available = [e for e in entries if e["acoustic_path"] is not None]
    print(f"[acoustic] 只有 {len(available)}/{len(entries)} 個檔案有聲音資料，先只處理這些")

    buckets = {s: {"specs": [], "y_cond": [], "y_sev": []} for s in ["train", "val", "test"]}
    for e in available:
        sig = io_utils.load_acoustic_mat(Path(e["acoustic_path"]))
        chunks = windowing.split_signal_by_time(sig, e["split_ratio"])
        for split, chunk in chunks.items():
            try:
                windows = windowing.sliding_window(chunk, config.ACOUSTIC_FS, config.WINDOW_SEC, config.WINDOW_OVERLAP)
            except ValueError:
                continue
            for w in windows:
                spec = features_acoustic.extract_mel_spectrogram(w, config.ACOUSTIC_FS)
                buckets[split]["specs"].append(features_acoustic.normalize_spectrogram(spec))
                buckets[split]["y_cond"].append(e["condition"])
                buckets[split]["y_sev"].append(e["severity_level"])

    for split, b in buckets.items():
        if b["specs"]:
            np.savez_compressed(
                out_dir / f"acoustic_specs_{split}.npz",
                X=np.stack(b["specs"], axis=0),
                y_condition=np.array(b["y_cond"]),
                y_severity=np.array(b["y_sev"]),
            )
            print(f"[acoustic/{split}] {len(b['specs'])} windows saved")


def check_paths() -> None:
    """在開始處理之前先確認路徑設定正確，並印出實際看到什麼。

    之前的版本在找不到檔案時，會一路跑到 pandas 才丟出看不懂的
    KeyError: ['split_ratio'] not found in axis，
    真正的原因其實只是「掃到 0 個檔案」。這個函式讓問題在第一時間就講清楚。
    """
    print("=" * 60)
    print("路徑檢查")
    print(f"  DATA_ROOT        = {config.DATA_ROOT}")
    print(f"    存在嗎？        {config.DATA_ROOT.exists()}")
    print(f"  VIBRATION_DIR    = {config.VIBRATION_DIR}")
    print(f"    存在嗎？        {config.VIBRATION_DIR.exists()}")
    print(f"  CURRENT_TEMP_DIR = {config.CURRENT_TEMP_DIR}")
    print(f"    存在嗎？        {config.CURRENT_TEMP_DIR.exists()}")
    print(f"  ACOUSTIC_DIR     = {config.ACOUSTIC_DIR}")
    print(f"    存在嗎？        {config.ACOUSTIC_DIR.exists()}")

    if not config.DATA_ROOT.exists():
        raise SystemExit(
            f"\n❌ 找不到 DATA_ROOT：{config.DATA_ROOT}\n"
            f"請打開 config.py，把 DATA_ROOT 改成你資料實際存放的資料夾路徑。\n"
            f"（Windows 路徑記得用 r\"...\" 這種寫法，例如：\n"
            f"    DATA_ROOT = Path(r\"C:\\Users\\你的名字\\dark_song\\acoustic_temp_vibration\")\n"
            f"）"
        )

    if config.DATA_ROOT.exists():
        print(f"\n  DATA_ROOT 底下實際有這些項目：")
        for item in sorted(config.DATA_ROOT.iterdir()):
            kind = "資料夾" if item.is_dir() else "檔案"
            n = len(list(item.glob("*"))) if item.is_dir() else ""
            print(f"    [{kind}] {item.name} {f'({n} 個項目)' if n != '' else ''}")

    n_vib = len(list(config.VIBRATION_DIR.glob("*.mat"))) if config.VIBRATION_DIR.exists() else 0
    if n_vib == 0:
        raise SystemExit(
            f"\n❌ 在 {config.VIBRATION_DIR} 裡找不到任何 .mat 檔案。\n"
            f"請比對上面印出來的『DATA_ROOT 底下實際有這些項目』，\n"
            f"確認資料夾名稱是不是真的叫 vibration / acoustic / 'current,temp'。\n"
            f"如果名稱不一樣（例如多包了一層資料夾），請修改 config.py 裡的\n"
            f"VIBRATION_DIR / ACOUSTIC_DIR / CURRENT_TEMP_DIR。"
        )
    print(f"\n✅ 路徑檢查通過，vibration/ 底下有 {n_vib} 個 .mat 檔案")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    # 聲音資料目前只有 5/45 個檔案（0Nm 下 BPFI/BPFO/Normal），資料量太少不足以
    # 訓練/驗證一個獨立模態的模型，所以預設「不處理聲音」。
    # 如果之後把 acoustic/ 資料夾補齊到 45 個檔案，加 --include-acoustic 再打開。
    parser.add_argument("--include-acoustic", action="store_true", help="聲音資料補齊後才需要加這個 flag")
    parser.add_argument("--skip-vibration", action="store_true")
    parser.add_argument("--skip-current-temp", action="store_true", help="跳過電流與溫度")
    parser.add_argument(
        "--feature-version",
        choices=["v1", "v2"],
        default="v2",
        help=(
            "振動特徵版本。v2=階次域+包絡解調（預設，207 欄，輸出 "
            "vibration_v2_features_*.csv）；v1=舊版三頻帶（40 欄，只留著做對照，"
            "已知無法分辨 Normal 與 Unbalance）"
        ),
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=3010.0,
        help="轉軸轉速。傳 0 代表每個窗自動偵測轉頻（較慢但不怕轉速設定錯）",
    )
    parser.add_argument(
        "--raw-windows",
        choices=["none", "float16", "float32"],
        default="float16",
        help=(
            "給 1D CNN 用的原始波形要怎麼存。"
            "none=不存（最快，先跑 XGBoost baseline 建議用這個）；"
            "float16=半精度（預設）；float32=單精度（最吃記憶體與硬碟）"
        ),
    )
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    check_paths()

    entries = build_manifest()
    print(f"共掃到 {len(entries)} 個振動檔案（電流溫度/聲音是否齊全見下方統計）")
    print("  有 current_temp 對應檔:", sum(1 for e in entries if e["current_temp_path"]))
    n_acoustic = sum(1 for e in entries if e["acoustic_path"])
    print(f"  有 acoustic 對應檔: {n_acoustic}（目前預設跳過聲音，不足 45 個不建議拿來訓練）")
    print(f"  原始波形儲存模式: --raw-windows {args.raw_windows}")
    print("  （每個檔案處理完都會印一行進度，沒有輸出代表還在讀那個檔案，屬正常現象）\n")

    split_by_file(entries)
    manifest_df = pd.DataFrame(entries).drop(columns=["split_ratio"])
    manifest_df.to_csv(config.OUTPUT_DIR / "manifest.csv", index=False)

    if not args.skip_vibration:
        process_vibration(entries, config.OUTPUT_DIR, raw_windows=args.raw_windows,
                          feature_version=args.feature_version,
                          rpm=(args.rpm if args.rpm > 0 else None))
    if not args.skip_current_temp:
        print("\n[current+temp] 開始計算 Normal 檔案的基準溫度 ...", flush=True)
        baseline = normal_temp_baseline(entries)
        print(f"[current+temp] 各負載基準溫度: {baseline}\n", flush=True)
        process_current_and_temperature(entries, config.OUTPUT_DIR, baseline)
    if args.include_acoustic:
        process_acoustic(entries, config.OUTPUT_DIR)
    else:
        print("\n[acoustic] 已跳過（預設行為，見上方說明；補齊資料後用 --include-acoustic 打開）")

    print("\n完成，輸出在", config.OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
