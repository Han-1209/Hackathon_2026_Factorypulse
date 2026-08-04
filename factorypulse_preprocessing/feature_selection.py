"""
特徵載入與去重複工具。

為什麼需要這支：
  前處理算出來的特徵裡有不少是「數學上等價」或高度相關的，例如
    - ch0_rms 與 ch0_std 相關係數剛好是 1.0
      （去除直流後 mean=0，標準差的定義就等於 RMS，兩者必然相等）
    - A_temp_mean 與 A_temp_max 相關 0.998
  留著這些重複特徵，對 XGBoost 的準確率影響不大（樹模型本來就耐得住共線性），
  但會讓「特徵重要度」被拆散：同一個資訊被兩欄分掉分數，
  結果每一欄看起來都不重要。而你們的產品要在畫面上顯示「關鍵異常證據」，
  這個重要度必須是可解釋的，所以要處理。

用法：
    from feature_selection import load_features
    X_train, y_train, meta_train = load_features('train', modality='vibration')

    # 或三個模態合併（以 file_id + split 對齊，取每個檔案的統計摘要）
    X, y, meta = load_features('train', modality='all')
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

META_COLS = ["condition", "severity_level", "load_nm", "file_id", "split"]

# 明確指定要丟掉的欄位（數學上與其他欄位等價）。
# 之所以用「寫死清單」而不是每次自動算相關矩陣，是因為要保證
# 訓練與推論時丟掉的欄位完全一致——推論時只有一筆資料，算不出相關係數。
EXACT_DUPLICATES = [
    # 去直流後 std == rms，四個通道都是
    "ch0_std", "ch1_std", "ch2_std", "ch3_std",
    # v2 的 shaft_freq：轉速固定 3010 RPM，8865 筆全部都是 50.17，
    # 是純粹的常數欄位。零資訊量，但會佔據特徵重要度圖表的位置，
    # 對可解釋性產品來說是雜訊。（若之後改用 rpm=None 逐窗偵測轉頻，
    # 這四欄會變成有變異的真實特徵，屆時把這幾行拿掉即可。）
    "ch0_shaft_freq", "ch1_shaft_freq", "ch2_shaft_freq", "ch3_shaft_freq",
    # 溫度：max 與 mean 相關 0.996~0.998，保留 mean（比較不受單一尖峰影響）
    "A_temp_max", "B_temp_max",
    "A_temp_delta_from_baseline_max", "B_temp_delta_from_baseline_max",
]


def _drop_known_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXACT_DUPLICATES if c in df.columns]
    return df.drop(columns=cols)


def find_redundant(df: pd.DataFrame, threshold: float = 0.98) -> list[tuple[str, str, float]]:
    """回傳高度相關的欄位配對，用來檢查/報告，不會自動刪除。"""
    X = df.drop(columns=[c for c in META_COLS if c in df.columns]).select_dtypes("number")
    X = X.loc[:, X.std() > 0]
    corr = X.corr().abs()
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if r > threshold:
                out.append((cols[i], cols[j], float(round(r, 4))))
    return sorted(out, key=lambda t: -t[2])


def load_features(
    split: str,
    modality: str = "vibration",
    drop_duplicates: bool = True,
    include_load: bool = True,
    include_temperature: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """載入某個 split 的特徵。

    modality: 'vibration' | 'current' | 'temperature' | 'all'
        'all' = 振動 + 電流（逐窗對齊）。預設不含溫度，原因見 _load_all 的說明。

    include_load: 是否把 load_nm 當成模型特徵。

        ⚠️ 強烈建議保持 True。實測顯示負載對訊號的影響比故障還大：
        振動 RMS 在「4Nm 正常」是 2.54，在「0Nm 不對心」是 2.08——
        如果模型不知道現在是幾 Nm，這兩種狀況幾乎無法分辨。
        控制負載之後，同一負載內 Normal 與 BPFI 的 Cohen's d 高達 20.9。
        （跨負載實驗例外，那時測試集的 load 訓練時沒出現過，要設 False）

    include_temperature: 只在 modality='all' 時有效，預設 False。
        溫度每個檔案只有 1 筆，放進逐窗分類器等於洩漏答案，詳見 _load_all。

    回傳 (X, y, meta)：X 只含數值特徵，y 是 condition，meta 保留 file_id 等欄位。
    """
    if modality in ("all", "all_v2"):
        return _load_all(split, drop_duplicates, include_load,
                         include_temperature=include_temperature,
                         vib_modality="vibration_v2" if modality == "all_v2" else "vibration")

    fname = {
        "vibration": f"vibration_features_{split}.csv",
        "vibration_v2": f"vibration_v2_features_{split}.csv",
        "current": f"current_features_{split}.csv",
        "temperature": f"temperature_features_{split}.csv",
    }[modality]

    path = config.OUTPUT_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。請先執行 build_dataset.py 產生特徵檔。"
        )

    df = pd.read_csv(path)
    if drop_duplicates:
        df = _drop_known_duplicates(df)

    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    y = df["condition"]
    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    X = X.select_dtypes("number")
    if include_load and "load_nm" in df.columns:
        X = X.copy()
        X["load_nm"] = df["load_nm"]
    return X, y, meta


def _load_all(split: str, drop_duplicates: bool, include_load: bool,
              include_temperature: bool = False, vib_modality: str = "vibration"):
    """振動 + 電流融合（逐窗對齊）。

    vib_modality='vibration_v2' 時改用 v2 的階次域/包絡特徵當振動側輸入。

    ⚠️⚠️ 這裡曾經有一個嚴重的標籤洩漏 bug，修正過程記錄如下，不要改回去：

    【舊版做法（錯的）】
        把電流與溫度用 groupby(file_id).mean() 聚合成「每個檔案一個值」，
        再 broadcast 回振動的每一個窗。

    【為什麼是災難】
        這個資料集每個檔案 = 一種 (load, condition, severity) 組合，
        也就是「一個檔案只有一種故障」。任何『檔案層級的常數』
        都等同於檔案指紋，也就等同於直接洩漏答案。
        實測：45 個檔案的 A_temp_mean 有 45 個完全不重複的值
        （27.94, 26.99, 27.38, ...），XGBoost 會學成
        「溫度≈27.94 → BPFI」這種記憶式規則，訓練集完美，
        但測試段溫度漂移到 28.46 就跨過閾值、整個崩掉。
        實測後果：融合準確率 0.58，比單用振動的 0.96 還低 38 個百分點。

    【現在的做法】
      - 電流：與振動『逐窗對齊』（實測兩者每個檔案的窗數 100% 相同），
              保留每個窗自己的電流特徵，不做檔案層級聚合。
      - 溫度：預設「不放進分類器」。溫度每個檔案只有 1 筆（共 45 筆），
              放進來必然是檔案指紋。溫度的正確用途是規則引擎裡的
              趨勢判斷（是否持續升溫、是否需要停機），不是逐窗分類特徵。
              真的要實驗可以設 include_temperature=True，但要知道結果不可信。
    """
    vib_X, vib_y, vib_meta = load_features(split, vib_modality, drop_duplicates, include_load=False)
    vib_meta = vib_meta.reset_index(drop=True)

    # 一次 concat 把所有欄位組起來，不要先 concat 再逐欄 vib["_win"] = ...
    # v2 有 170+ 欄，對這麼寬的 DataFrame 做單欄插入會觸發 pandas 的
    # PerformanceWarning（DataFrame is highly fragmented），
    # 每次載入都刷三次，terminal 很難看。功能無影響，但 demo 時要乾淨。
    vib = pd.concat(
        [vib_X.reset_index(drop=True), vib_meta,
         vib_meta.groupby("file_id").cumcount().rename("_win")],
        axis=1,
    )

    merged = vib
    try:
        cur_X, _, cur_meta = load_features(split, "current", drop_duplicates, include_load=False)
        cur_meta = cur_meta.reset_index(drop=True)
        cur = pd.concat(
            [cur_X.add_prefix("cur_").reset_index(drop=True),
             cur_meta[["file_id"]],
             cur_meta.groupby("file_id").cumcount().rename("_win")],
            axis=1,
        )

        before = len(merged)
        merged = merged.merge(cur, on=["file_id", "_win"], how="inner")
        if len(merged) != before:
            print(f"  ⚠️ 逐窗對齊後筆數從 {before} 變成 {len(merged)}"
                  f"（振動與電流的窗數不完全相同，已取交集）")
    except FileNotFoundError:
        print("  （略過 current：特徵檔尚未產生）")

    if include_temperature:
        try:
            t_X, _, t_meta = load_features(split, "temperature", drop_duplicates, include_load=False)
            agg = pd.concat([t_X, t_meta[["file_id"]].reset_index(drop=True)], axis=1)
            agg = agg.groupby("file_id").mean().add_prefix("tmp_")
            merged = merged.merge(agg, left_on="file_id", right_index=True, how="left")
            print("  ⚠️ 已加入溫度特徵，但它是檔案層級常數，會造成標籤洩漏，結果僅供對照")
        except FileNotFoundError:
            print("  （略過 temperature：特徵檔尚未產生）")

    merged = merged.drop(columns=["_win"])
    y = merged["condition"]
    meta = merged[[c for c in META_COLS if c in merged.columns]].copy()
    X = merged.drop(columns=[c for c in META_COLS if c in merged.columns]).select_dtypes("number")
    if include_load:
        X = X.copy()
        X["load_nm"] = merged["load_nm"].values
    return X, y, meta


def load_all_splits(modality: str = "vibration", **kwargs):
    """把 train/val/test 三份合併成一份，給「跨負載切分」用。

    跨負載實驗要重新決定誰是訓練、誰是測試（依 load_nm 而不是依時間），
    所以要先把原本的三份合起來，再按負載重切。
    """
    Xs, ys, metas = [], [], []
    for split in ["train", "val", "test"]:
        X, y, meta = load_features(split, modality, **kwargs)
        Xs.append(X)
        ys.append(y)
        metas.append(meta)
    return (
        pd.concat(Xs, ignore_index=True),
        pd.concat(ys, ignore_index=True),
        pd.concat(metas, ignore_index=True),
    )


def drop_scale_dependent(X: pd.DataFrame) -> pd.DataFrame:
    """移除「有量綱」特徵，只留下比值 / 佔比 / 無量綱統計量。

    為什麼跨負載實驗要這樣做：
        負載從 0Nm 加到 4Nm，整體振動振幅會整條抬上去。像 rms、peak、
        band_energy 這類有量綱的特徵，它們的絕對數值在不同負載下落在
        完全不同的區間。XGBoost 學的是「某欄 > 某個閾值」這種絕對切點，
        訓練時只看過 0/2Nm 的區間，遇到 4Nm 就整批落在切點的同一側，
        於是整個類別一起塌陷（實測：4Nm 的 Normal 有 100% 被判成 Unbalance）。

        改用無量綱特徵（2x/1x 比值、頻帶佔比、包絡邊帶比、峰度…）後，
        數值不隨整體振幅縮放，切點才有跨負載的意義。

    注意：這只用在「跨負載/跨轉速」實驗。同機台部署情境（實驗 A）
    應該保留有量綱特徵 —— 振幅大小本來就是很強的診斷資訊，
    在負載已知的情況下沒有理由丟掉。
    """
    import features_vibration_v2 as v2
    keep = v2.scale_invariant_columns(X.columns)
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print(f"    跨負載模式：排除 {len(dropped)} 個有量綱特徵，"
              f"保留 {len(keep)} 個無量綱特徵")
    return X[keep]


def split_by_load(X, y, meta, test_loads=(4,)):
    """依負載切分：指定的負載當測試集，其餘當訓練集。

    這是回應評審「你們是不是資料洩漏／模型只是背答案」最有力的實驗設計：
    測試集的負載在訓練時模型完全沒看過，如果還能分類正確，
    代表它學到的是故障本身的物理特徵，而不是記住某個負載下的數值範圍。

    注意：這個模式下不能把 load_nm 當特徵（測試集的 load 值訓練時沒出現過，
    樹模型會無法正確分割），所以呼叫 load_features 時要設 include_load=False。
    """
    test_loads = set(test_loads)
    is_test = meta["load_nm"].isin(test_loads).values
    return (
        X[~is_test].reset_index(drop=True), y[~is_test].reset_index(drop=True), meta[~is_test].reset_index(drop=True),
        X[is_test].reset_index(drop=True), y[is_test].reset_index(drop=True), meta[is_test].reset_index(drop=True),
    )


if __name__ == "__main__":
    for mod in ["vibration", "current", "temperature"]:
        try:
            X, y, meta = load_features("train", mod)
        except FileNotFoundError as ex:
            print(f"{mod}: {ex}\n")
            continue
        red = find_redundant(pd.concat([X, meta], axis=1))
        print(f"=== {mod} ===")
        print(f"  特徵數（去重後）: {X.shape[1]}   樣本數: {X.shape[0]}")
        if red:
            print(f"  仍有 r>0.98 的配對 {len(red)} 組，前 5 組：")
            for a, b, r in red[:5]:
                print(f"     r={r}  {a} <-> {b}")
        else:
            print("  已無 r>0.98 的重複特徵")
        print()
