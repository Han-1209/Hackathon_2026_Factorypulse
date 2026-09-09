"""
儀表板的資料存取層。

把「載入模型、跑推論、整理成畫面要的格式」集中在這裡，
app.py 只負責畫面，兩者分開比較好維護，也方便單獨測試資料邏輯。

支援兩種運轉模式：
  load  ── 變負載資料集（定轉速 3010 RPM，改變負載 0/2/4 Nm）
            5 類：正常 / 內圈 / 外圈 / 不對心 / 不平衡
            有溫度資料，可做完整三模態證據
  speed ── 變轉速資料集（618~2480 RPM 連續變化）
            4 類：正常 / 內圈 / 外圈 / 滾動體
            沒有溫度資料，證據頁只有振動與電流
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config

MODELS_DIR = Path(__file__).parent / "models"

MODE_LABELS = {
    "load": "定轉速・變負載",
    "speed": "變轉速",
}

# 每種模式對應的模型檔、特徵目錄、是否有溫度
MODE_SPEC = {
    "load": {
        "models": {"vibration": "model_vibration.pkl", "current": "model_current.pkl"},
        "processed": config.OUTPUT_DIR,
        "has_temperature": True,
        "condition_col": "load_nm",
        "condition_unit": "Nm",
    },
    "speed": {
        # 變轉速資料集有電流，但沒有「純電流」模型——train_speed.py 訓練的是
        # 振動單模態與「振動+電流融合」兩種。所以這裡用融合模型當主力，
        # 它實際吃 37 個振動特徵 + 8 個電流特徵（欄位前綴 cur_）。
        "models": {"fused": "model_speed_all.pkl"},
        "processed": config.SPEED_OUTPUT_DIR,
        "has_temperature": False,
        "condition_col": "rpm",
        "condition_unit": "RPM",
    },
}

FAULT_DISPLAY = {
    "Normal": "正常",
    "BPFI": "軸承內圈故障",
    "BPFO": "軸承外圈故障",
    "Ball": "滾動體故障",
    "Misalign": "軸心不對中",
    "Unbalance": "轉子不平衡",
}

META_COLS = ["condition", "severity_level", "load_nm", "rpm", "profile", "file_id", "split"]

# ⚠️ 必須與 feature_selection.EXACT_DUPLICATES 保持一致。
# 兩邊不同步會導致訓練與推論的欄位集合不同，模型直接對不上。
EXACT_DUPLICATES = [
    # 去直流後 std 與 rms 數學上相等
    "ch0_std", "ch1_std", "ch2_std", "ch3_std",
    # v2 的 shaft_freq 是常數欄位（轉速固定），零資訊量
    "ch0_shaft_freq", "ch1_shaft_freq", "ch2_shaft_freq", "ch3_shaft_freq",
    "A_temp_max", "B_temp_max",
    "A_temp_delta_from_baseline_max", "B_temp_delta_from_baseline_max",
]


def fault_display(name: str) -> str:
    return FAULT_DISPLAY.get(name, name)


# ---------------------------------------------------------------- LOLO 模型
# leave-one-load-out：每個模型都排除一個負載，儀表板上每台機台改用
# 「沒看過它那個負載」的那一個。理由見 train_lolo.py 的說明——
# 原本上線的 model_vibration.pkl 是同錄音切分訓練的，那組數字《數字口徑表》
# 明文不可對外引用，但畫面上一直在用它。
LOLO_TAG = {"vibration": "vib", "current": "cur"}

# 二元 AUC（正常 vs 有故障）低於此值，視為這個模態「跨負載不成立」，
# 不讓它參與風險計算。必須與 train_lolo.USABLE_MIN_AUC 一致。
#
# ⚠️ 這條門檻是為了電流模態存在的，實測結果：
#     排除 0Nm  逐窗 AUC 0.438     排除 2Nm  0.535     排除 4Nm  0.606
# 也就是換到沒看過的負載，電流模型判斷「有沒有故障」等同亂猜。
# 原因不難理解：去掉有量綱特徵後電流只剩 7 個無量綱特徵，
# 而電流的訊號幅度本來就由負載主導，跨負載等於把唯一的資訊來源抽掉。
# 這與《數字口徑表》「融合電流沒有提升準確率」是同一件事的量化版本。
#
# 不擋掉的話後果很具體：電流模型對正常機台也給出 0.985 的異常機率，
# 規則引擎 R1「振動與電流同時異常」會對每一台都成立，風險分數整片虛高。
USABLE_MIN_AUC = 0.70


@lru_cache(maxsize=16)
def load_lolo(modality: str, held_out_load: int) -> dict | None:
    """載入排除某個負載的模型。找不到就回 None，讓呼叫端退回原模型。"""
    tag = LOLO_TAG.get(modality)
    if tag is None:
        return None
    path = MODELS_DIR / f"model_{tag}_lolo_{int(held_out_load)}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def lolo_available() -> bool:
    """三個負載的振動模型是否都齊了。畫面用它決定要不要顯示警告橫幅。"""
    return all(load_lolo("vibration", l) is not None for l in (0, 2, 4))


def load_model_for(mode: str, modality: str, load_nm: float | None = None) -> dict:
    """取得該用哪個模型。

    變負載模式且該負載有對應的 LOLO 模型時優先用它（模型沒看過這個負載）；
    否則退回原本的模型，並在 bundle 裡標記 held_out_load=None，
    讓畫面可以誠實顯示「這個結果來自看過同錄音的模型」。
    """
    if mode == "load" and load_nm is not None:
        bundle = load_lolo(modality, int(round(load_nm)))
        if bundle is not None:
            return bundle
    return load_model(mode, modality)


@lru_cache(maxsize=8)
def load_model(mode: str, modality: str) -> dict:
    fname = MODE_SPEC[mode]["models"].get(modality)
    if fname is None:
        raise FileNotFoundError(f"{mode} 模式沒有 {modality} 模型")
    path = MODELS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"找不到模型 {path}。請先執行 "
            f"{'train_baseline.py' if mode == 'load' else 'train_speed.py'}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def vibration_source(mode: str) -> str:
    """這個模式的振動特徵，該讀 v1 還是 v2 的檔案？

    ⚠️ 這裡不能寫死。振動特徵改版之後，model_vibration.pkl 是用 v2 的
    176 個特徵訓練的，但特徵檔仍然叫 vibration_features_*.csv（v1，40 欄）
    與 vibration_v2_features_*.csv（v2）兩份並存。讀錯版本會讓
    infer_modality 裡的 X[bundle["feature_names"]] 直接 KeyError，
    或更糟——欄位剛好對上但意義不同，畫面顯示看似正常的錯誤結果。

    判斷依據優先序：
      1. bundle["feature_version"]（train_baseline.py 寫入的正式標記）
      2. 特徵名稱特徵（v2 才有 env_/order_ 這類欄位）—— 這是給
         model_speed_all.pkl 這種舊模型的後備判斷，它沒有版本標記。
    """
    spec = MODE_SPEC[mode]
    for modality in ("vibration", "fused"):
        if modality not in spec["models"]:
            continue
        try:
            bundle = load_model(mode, modality)
        except FileNotFoundError:
            continue
        if bundle.get("feature_version") == "v2":
            return "vibration_v2"
        names = bundle.get("feature_names") or []
        if any("_env_" in n or "_order_" in n for n in names):
            return "vibration_v2"
        return "vibration"
    return "vibration"


@lru_cache(maxsize=16)
def _read_features(mode: str, modality: str, split: str) -> pd.DataFrame:
    path = MODE_SPEC[mode]["processed"] / f"{modality}_features_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到特徵檔 {path}，請先執行對應的 build_dataset 腳本")
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in EXACT_DUPLICATES if c in df.columns])


def split_features(mode: str, modality: str, split: str):
    """回傳 (X, y, meta)。X 含運轉條件欄位（load_nm 或 rpm），與訓練時一致。

    modality="fused" 會把振動與電流「逐窗對齊」後合併，欄位命名與
    train_speed.py 訓練時完全相同（電流欄位加 cur_ 前綴），
    否則模型會因為欄位對不上而拒收。
    """
    if modality == "fused":
        return _fused_features(mode, split)

    # "vibration" 是邏輯名稱，實際檔案可能是 v1 或 v2，交給 vibration_source 決定
    if modality == "vibration":
        modality = vibration_source(mode)

    df = _read_features(mode, modality, split)
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    y = df["condition"]
    X = df.drop(columns=[c for c in META_COLS if c in df.columns]).select_dtypes("number")
    cond_col = MODE_SPEC[mode]["condition_col"]
    if cond_col in df.columns:
        X = X.copy()
        X[cond_col] = df[cond_col]
    return X, y, meta


def _fused_features(mode: str, split: str):
    """振動 + 電流逐窗對齊。

    對齊方式用「同一檔案內的第幾個窗」而不是時間戳，因為兩者取樣率不同
    （振動 25.6kHz、電流降採樣後 25kHz），但切窗規則相同，
    所以第 n 個窗代表的是同一段時間。實測兩者每個檔案的窗數完全一致。
    """
    vib = _read_features(mode, vibration_source(mode), split).copy()
    cur = _read_features(mode, "current", split).copy()
    vib["_win"] = vib.groupby("file_id").cumcount()
    cur["_win"] = cur.groupby("file_id").cumcount()

    cur_cols = [c for c in cur.columns if c.startswith("current_")]
    cur_small = cur[["file_id", "_win"] + cur_cols].rename(
        columns={c: "cur_" + c for c in cur_cols}
    )
    merged = vib.merge(cur_small, on=["file_id", "_win"], how="inner").drop(columns=["_win"])

    meta = merged[[c for c in META_COLS if c in merged.columns]].copy()
    y = merged["condition"]
    X = merged.drop(columns=[c for c in META_COLS if c in merged.columns]).select_dtypes("number")
    cond_col = MODE_SPEC[mode]["condition_col"]
    if cond_col in merged.columns:
        X = X.copy()
        X[cond_col] = merged[cond_col]
    return X, y, meta


def list_equipment(mode: str, split: str = "test") -> pd.DataFrame:
    """回傳這個模式下所有設備的清單（一台設備 = 一個原始錄音檔）。"""
    _, y, meta = split_features(mode, "vibration", split)
    cond_col = MODE_SPEC[mode]["condition_col"]
    rows = []
    for fid, g in meta.groupby("file_id"):
        idx = g.index
        rows.append({
            "設備編號": fid,
            "運轉條件": round(float(meta.loc[idx, cond_col].mean()), 1) if cond_col in meta else None,
            "實際狀態": fault_display(y.loc[idx].iloc[0]),
            "_truth": y.loc[idx].iloc[0],
            "窗數": len(idx),
        })
    return pd.DataFrame(rows).sort_values("設備編號").reset_index(drop=True)


def infer_modality(bundle: dict, X: pd.DataFrame) -> dict:
    """對一台設備的所有時間窗做推論並聚合。

    聚合邏輯與 diagnose_pipeline.py 一致（異常機率取中位數、類別取多數決），
    這裡重寫一份是為了讓儀表板不必依賴那支腳本的 CLI 流程。
    """
    model = bundle["model"]
    classes = list(bundle["classes"])

    # 欄位對不上時，直接講清楚是版本不一致，不要丟一個看不懂的 KeyError。
    # 這個錯誤最常見的原因是：重新訓練了模型，但特徵檔沒重跑（或反過來）。
    missing = [c for c in bundle["feature_names"] if c not in X.columns]
    if missing:
        raise KeyError(
            f"特徵檔與模型對不上：模型需要 {len(bundle['feature_names'])} 個特徵，"
            f"但特徵檔缺少 {len(missing)} 個，例如 {missing[:3]}。\n"
            f"通常代表模型與特徵檔的版本不一致。請確認：\n"
            f"  1. build_dataset.py 與 train_baseline.py 用的是同一個 --feature-version\n"
            f"  2. 兩者都重跑過（改了特徵就一定要重新訓練）"
        )
    X = X[bundle["feature_names"]]

    proba = model.predict_proba(X.values)
    if "Normal" in classes:
        anomaly = 1.0 - proba[:, classes.index("Normal")]
    else:
        anomaly = proba.max(axis=1)

    pred_labels = [classes[i] for i in proba.argmax(axis=1)]
    vc = pd.Series(pred_labels).value_counts()
    top_label = vc.index[0]
    agreement = float(vc.iloc[0] / len(pred_labels))
    median_anomaly = float(np.median(anomaly))

    # 大部分窗正常但少數窗嚴重異常 -> 早期故障的典型樣態，改報次高類別
    if top_label == "Normal" and median_anomaly >= 0.5 and len(vc) > 1:
        top_label = vc.index[1]

    return {
        "anomaly_prob": median_anomaly,
        "anomaly_series": anomaly,          # 給趨勢圖用
        "fault_type": top_label,
        "confidence": agreement,
        "n_windows": int(len(X)),
        "vote_distribution": vc.to_dict(),
    }


def feature_importance(mode: str, modality: str = "vibration", top: int = 10) -> pd.DataFrame:
    """直接從模型物件取特徵重要度。

    不從 results/importance_*.csv 讀，是因為那些檔案只有變負載資料集有，
    變轉速模式若共用同一個檔案會顯示錯誤的重要度。
    從模型本身取一定對應到當下選的模式，不會出錯。
    """
    bundle = load_model(mode, modality)
    model = bundle["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({
            "feature": bundle["feature_names"],
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def build_llm_diag(mode: str, res: dict, X: pd.DataFrame) -> dict:
    """把「即時診斷」頁的推論結果，補成 LLM 助手需要的完整診斷 dict。

    ⚠️ 為什麼需要這個：即時診斷頁原本直接把 infer_modality() 的輸出丟給
    LLMAssistant，但那個 dict 只有 fault_type / anomaly_prob / confidence，
    沒有知識庫內容、沒有規則引擎結果、也沒有量測證據。
    結果就是這一頁的 LLM 只能憑一個類別名稱去講通論 —— 而這一頁偏偏是
    demo 時最常操作的頁面。

    這裡補齊三塊：
      1. 知識庫（可能原因 / 建議檢查 / 建議行動）
      2. 由異常機率推出的風險等級與是否建議停機
      3. evidence_features：這批窗的特徵中位數，讓證據層算出「幾倍於正常」
    """
    import knowledge_base as kb

    fault = res.get("fault_type", "Normal")
    info = kb.get_fault_info(fault)
    ap = float(res.get("anomaly_prob", 0.0))

    if fault == "Normal" or ap < 0.3:
        risk, stop, window = "健康", False, "無需處理"
    elif ap < 0.7:
        risk, stop, window = "注意", False, "兩週內安排檢查"
    elif ap < 0.9:
        risk, stop, window = "高風險", False, "72 小時內處理"
    else:
        risk, stop, window = "危急", True, "盡快安排停機"

    cond_col = MODE_SPEC[mode]["condition_col"]
    load_nm = float(X[cond_col].median()) if cond_col in X.columns else 0.0

    diag = {
        "fault_type": fault,
        "fault_label": info.get("display", fault),
        "risk_level_display": risk,
        "anomaly_prob": ap,
        "confidence": float(res.get("confidence", 0.0)),
        "should_stop": stop,
        "action_window": window,
        "triggered_rules": [
            f"逐窗多數決分類為 {fault_display(fault)}，"
            f"一致率 {res.get('confidence', 0):.0%}（共 {res.get('n_windows', 0)} 個時間窗）"
        ],
        "causes": info.get("causes", []),
        "checks": info.get("checks", []),
        "actions": info.get("actions", []),
        "tools": info.get("tools", []),
        "parts": info.get("parts", []),
        "safety": info.get("safety", []),
        "effort": info.get("effort", ""),
        "known_confusion": info.get("known_confusion", ""),
        "load_nm": load_nm,
        "evidence_features": X.median(numeric_only=True).to_dict(),
    }

    diag["evidence"] = []
    diag["severity_series"] = None
    has_baseline = False
    if mode == "load":     # 理由同 diagnose()：基準檔只適用變負載資料集
        try:
            import evidence
            evidence.load_reference()
            has_baseline = True
            diag["evidence"] = evidence.extract(diag["evidence_features"],
                                                load_nm=load_nm, top=5,
                                                fault_type=fault)
            diag["severity_series"] = evidence.severity_series(
                X, load_nm=load_nm, fault_type=fault)
        except (FileNotFoundError, ImportError):
            pass

    # 信心度與「設備診斷」頁走同一套判定，避免同一台機器在兩頁看到不同等級
    import confidence as _cf
    diag["confidence_verdict"] = _cf.assess(
        evidence=diag["evidence"],
        fault_type=fault,
        agreement=float(res.get("confidence", 0.0)),
        n_windows=int(res.get("n_windows", 0)),
        cross_sensor=None,      # 上傳的檔案只有單一模態
        known_confusion=info.get("known_confusion", ""),
        has_physical_baseline=has_baseline,
    )
    return diag


def get_temperature(mode: str, file_id: str, split: str):
    """回傳 (delta_from_baseline, trend_slope)。變轉速資料集沒有溫度，回傳 (0, 0)。"""
    if not MODE_SPEC[mode]["has_temperature"]:
        return 0.0, 0.0
    try:
        df = _read_features(mode, "temperature", split)
    except FileNotFoundError:
        return 0.0, 0.0
    row = df[df.file_id == file_id]
    if row.empty:
        return 0.0, 0.0
    r = row.iloc[0]
    delta = float(np.mean([r.get("A_temp_delta_from_baseline_mean", 0.0),
                           r.get("B_temp_delta_from_baseline_mean", 0.0)]))
    slope = float(np.mean([r.get("A_temp_trend_slope", 0.0),
                           r.get("B_temp_trend_slope", 0.0)]))
    return delta, slope


def diagnose(mode: str, file_id: str, split: str = "test",
             window_minute: int | None = None) -> dict:
    """對單一設備跑完整診斷，回傳畫面可直接用的 dict。

    window_minute: 指定「這一分鐘要看訊號的哪一段」。
        給了就只取那一段時間窗來推論，讓廠房畫面上的數值隨時間自然變化。
        不給就用全部的窗（單台設備診斷頁用這個，結果比較穩定）。
    """
    from rule_engine import ModalityInput, TemperatureInput, diagnose as run_rules

    spec = MODE_SPEC[mode]
    detail = {}
    truth = None
    cond_value = None
    vib_features = None      # 給證據層算「幾倍於正常」用

    vib_frame = None         # 逐窗特徵表，給物理嚴重度趨勢圖用
    model_scope = {}         # 每個模態實際用了哪個模型（沒看過哪個負載）
    unusable = {}            # 跨負載驗證不通過的模態 -> 它的 AUC

    for modality in spec["models"]:
        X, y, meta = split_features(mode, modality, split)
        mask = (meta.file_id == file_id).values
        if not mask.any():
            raise ValueError(f"{split} 資料集裡找不到設備 {file_id}")
        Xf = X[mask]
        if window_minute is not None:
            from factory_sim import observation_slice
            idx = observation_slice(len(Xf), window_minute)
            Xf = Xf.iloc[idx]

        truth = y[mask].iloc[0]
        cond_col = spec["condition_col"]
        if cond_col in meta.columns:
            cond_value = float(meta.loc[mask, cond_col].mean())

        # ⚠️ 模型要先知道負載才能挑：這台機台的診斷必須由「沒看過這個負載」
        # 的模型做出來，否則畫面上的數字就是《數字口徑表》禁止對外的那一類。
        bundle = load_model_for(mode, modality, cond_value)
        model_scope[modality] = bundle.get("held_out_load")
        detail[modality] = infer_modality(bundle, Xf)

        # 跨負載站不住腳的模態，只留著顯示，不讓它進風險計算（見 USABLE_MIN_AUC）
        auc = bundle.get("holdout_auc")
        if auc is not None and auc < USABLE_MIN_AUC:
            unusable[modality] = auc

        # 振動（或融合模型，它也含振動欄位）的特徵中位數 -> 證據層的輸入。
        # 用中位數而非平均，理由同 infer_modality：對瞬間干擾的窗比較穩。
        if modality in ("vibration", "fused"):
            vib_features = Xf.median(numeric_only=True).to_dict()
            vib_frame = Xf

    # 兩種模式的模型組成不同，這裡統一成規則引擎要的介面：
    #   load  : vibration + current 兩個獨立模型，可做跨感測交叉驗證
    #   speed : 單一融合模型（振動+電流一起輸入），無法拆出各感測器的獨立意見，
    #           所以交叉驗證那幾條規則在這個模式下不會有實質作用——
    #           畫面上會標明「單一融合模型」，不假裝有兩個獨立判斷。
    current_usable = "current" in detail and "current" not in unusable
    if "fused" in detail:
        vib = cur = detail["fused"]
        primary_key = "fused"
    else:
        vib = detail["vibration"]
        # 電流跨負載不成立時，用振動的結果代入 —— 規則引擎的介面需要兩個輸入，
        # 但不能拿一個等同亂猜的訊號去觸發「兩來源同時異常」那類規則。
        cur = detail["current"] if current_usable else vib
        primary_key = "vibration"

    delta, slope = get_temperature(mode, file_id, split)
    d = run_rules(
        vibration=ModalityInput(vib["anomaly_prob"], vib["fault_type"], vib["confidence"]),
        current=ModalityInput(cur["anomaly_prob"], cur["fault_type"], cur["confidence"]),
        temperature=TemperatureInput(delta, slope),
        load_nm=cond_value,
        use_current=current_usable,
    )

    out = d.to_dict()

    # 這個模式沒有的感測來源，要把對應的健康分項拿掉。
    # 規則引擎為了介面一致，在缺電流時用振動結果代入、缺溫度時當作 0，
    # 那些是計算用的後備值，不是真的量到的東西——直接顯示會讓使用者以為
    # 系統量了電流與溫度。畫面上寧可少一項，也不要給看似有意義的假數字。
    if not current_usable:
        # 兩種情況都會走到這裡：
        #   1. 融合模型 —— 輸入含電流，但給不出「電流自己的意見」
        #   2. 電流模型跨負載不成立（AUC < USABLE_MIN_AUC）
        # 兩種情況下「負載健康」都不是獨立量到的東西，顯示它等於給假數字。
        out["health_scores"].pop("負載健康", None)
    if not spec["has_temperature"]:
        out["health_scores"].pop("熱狀態", None)
        out["temperature_status"] = "未配置溫度感測"

    # 這個模式實際用到哪些感測器（給畫面顯示用，不是模型結構）
    if primary_key == "fused":
        sensors = ["振動", "電流"]
    else:
        sensors = ["振動"] + (["電流"] if current_usable else [])
    if spec["has_temperature"]:
        sensors.append("溫度")

    # 融合模型只會給出「一個」判斷，規則引擎為了介面一致把同一份結果
    # 同時當成振動與電流的意見，導致證據表出現兩列一模一樣的內容。
    # 那會讓人誤以為系統做了兩次獨立驗證，所以改寫成單一列並講清楚。
    if primary_key == "fused":
        from rule_engine import _level_text
        f = detail["fused"]
        # 特徵數不要寫死。振動特徵從 v1 的 37 個換成 v2 的 176 個之後，
        # 原本寫死的「37 個」就變成錯的了 —— 畫面上出現與實際不符的數字，
        # 是 demo 最容易被抓到的細節。改成從模型 bundle 直接讀。
        n_feat = int(f.get("n_features") or 0)
        n_desc = f"{n_feat} 個振動與電流特徵" if n_feat else "振動與電流特徵"
        out["evidence"] = [{
            "來源": "振動 + 電流（融合）",
            "狀態": _level_text(f["anomaly_prob"]),
            "數值": f"異常機率 {f['anomaly_prob']:.2f}",
            "說明": f"單一模型同時輸入 {n_desc}，"
                    f"分類為 {fault_display(f['fault_type'])}"
                    f"（逐窗一致率 {f['confidence']:.0%}）。"
                    f"此架構下無法拆出各感測器的獨立意見，故不做跨感測交叉驗證。",
        }]

    # 電流跨負載不成立時改寫證據表那一列。不改的話畫面會列出
    # 「電流・高異常・異常機率 0.99」——那個數字在沒看過的負載上等同亂猜，
    # 留著會變成一條看似有力、實際無效的證據。
    if not current_usable and "current" in unusable:
        auc = unusable["current"]
        for row in out["evidence"]:
            if row.get("來源") == "電流":
                row.update({
                    "狀態": "跨負載不成立",
                    "數值": f"二元 AUC {auc:.2f}",
                    "說明": (f"電流模型在沒看過的負載上，判斷「有沒有故障」的 AUC 只有 "
                             f"{auc:.2f}（0.5 等同亂猜），因此本次不讓它參與風險計算。"
                             f"去掉有量綱特徵後電流只剩 7 個無量綱特徵，"
                             f"而電流幅度本來就由負載主導。"),
                })

    # ---- 物理證據：實際量到的數值 vs 同負載正常基準 ----
    # 「多感測證據」頁的核心。上面 out["evidence"] 是各感測來源的異常機率
    # （模型的意見），這裡是證據本身（頻譜量到什麼），兩者不同層次。
    # 變轉速模式暫時沒有基準檔（evidence_reference.json 只由變負載資料集產生），
    # 取不到就給空清單，畫面會自動不顯示這一區，不會壞掉。
    out["evidence_features"] = vib_features or {}
    out["physical_evidence"] = []
    # ⚠️ 只在變負載模式啟用。基準檔 evidence_reference.json 是由變負載資料集的
    # Normal 檔案、按 0/2/4 Nm 分別算出來的。變轉速模式若也套用，
    # 會發生兩件錯事：
    #   1. 運轉條件是 RPM（例如 1500），在基準表裡找不到，會退回用 0Nm 的基準
    #   2. 兩個資料集的機台、感測器配置都不同，數值本來就不可比
    #      （而 ch0_kurtosis 這類欄位剛好兩邊都有，會靜靜算出一個看似合理的倍數）
    # 與其顯示一個錯的數字，不如不顯示。
    out["severity_series"] = None
    has_baseline = False
    if mode == "load":
        try:
            import evidence as _ev
            _ev.load_reference()          # 取不到基準檔就不宣稱做過物理驗證
            has_baseline = True
            out["physical_evidence"] = _ev.extract(
                vib_features or {}, load_nm=cond_value or 0.0, top=5,
                fault_type=d.probable_fault)
            # 逐窗物理嚴重度：趨勢圖的主線。分類器機率是決策不是量測，
            # 在明確樣本上會飽和成一條水平線（見 evidence.severity_series）。
            if vib_frame is not None:
                out["severity_series"] = _ev.severity_series(
                    vib_frame, load_nm=cond_value or 0.0,
                    fault_type=d.probable_fault)
        except (FileNotFoundError, ImportError):
            pass

    # ---- 物理閘門：先偵測再診斷 ----
    # 物理層說「各項都在基準內」時，推翻分類器的故障結論（見 _apply_physical_gate）。
    # 順序很重要：閘門必須在 contextual_fault 與信心度之前，
    # 否則畫面上的措辭與信心度仍然依據被推翻的那個結論算出來。
    out["gate_overrode"] = None
    gate = "unknown"
    if mode == "load":
        gate = _apply_physical_gate(out, d, out["physical_evidence"])
    out["physical_gate"] = gate
    fault_now = out["probable_fault"]

    # 依嚴重度調整故障措辭（見 contextual_fault 的說明）
    ctx_label, ctx_note = contextual_fault(
        out["probable_fault_display"], out["risk_score"], fault_now == "Normal"
    )
    if out["gate_overrode"]:
        ctx_note = (f"分類器判為「{out['gate_overrode']}」，但頻譜上各項特徵都在"
                    f"同負載正常基準範圍內，找不到支持它的證據，因此依基準判定為正常。")

    # ---- 信心度：四條可檢核條件，不是模型機率 ----
    # 實測顯示模型機率分不出對錯（跨負載測試裡判錯的檔案一致率高達 96~100%，
    # 判對的反而只有 76%），所以信心度改由 confidence.py 依證據判定。
    import confidence as _cf
    out["confidence_verdict"] = _cf.assess(
        evidence=out["physical_evidence"],
        fault_type=fault_now,
        agreement=float(detail[primary_key]["confidence"]),
        n_windows=int(detail[primary_key]["n_windows"]),
        cross_sensor=((float(detail["vibration"]["anomaly_prob"]),
                       float(detail["current"]["anomaly_prob"]))
                      if current_usable else None),
        known_confusion=out["known_confusion"],
        overrode=out["gate_overrode"],
        has_physical_baseline=has_baseline,
    )

    out.update({
        "model_scope": model_scope,
        "model_is_holdout": model_scope.get(primary_key) is not None,
        "equipment_id": file_id,
        "mode": mode,
        "fault_label": ctx_label,      # 畫面顯示用，已依嚴重度調整措辭
        "fault_note": ctx_note,        # 補充說明，沒有就是空字串
        "condition_value": cond_value,
        "condition_unit": spec["condition_unit"],
        "has_temperature": spec["has_temperature"],
        "has_current": current_usable,               # 電流是否作為獨立來源參與判斷
        "uses_current": primary_key == "fused" or current_usable,
        "unusable_modalities": unusable,             # 跨負載驗證不通過的模態
        "current_usable": current_usable,
        "primary_key": primary_key,                  # 主模型在 modality_detail 裡的 key
        "sensors": sensors,
        "ground_truth": truth,
        "ground_truth_display": fault_display(truth),
        "correct": fault_now == truth,
        "modality_detail": detail,
    })
    return out


# ---------------------------------------------------------------- 物理閘門
# 「先偵測、再診斷」—— 現場振動監測系統的標準架構，也是這套系統原本缺的一層：
#
#   偵測（有沒有異常）  由「與同負載正常基準的比值」決定，完全不經模型
#   診斷（是哪一種）    才交給分類器命名
#
# 為什麼一定要加：分類器認不出「正常」。這個資料集只有 3 支正常錄音
# （0/2/4 Nm 各一支），把其中一支保留起來不給模型看，它就只剩 2 支正常樣本
# 對上 42 支故障樣本。實測結果是正常錄音被判成轉子不平衡，
# 而且逐窗一致率高達 76~100% —— 分類器不但錯，還錯得很有把握。
#
# 物理層在同一批檔案上乾淨俐落（45 支錄音全跑過）：
#
#   3 支正常錄音   最大偏離全部 1.0 倍（連 evidence.MIN_RATIO 的 1.5 倍都沒到）
#   42 支故障錄音  最大偏離中位數 15.9 倍，最高 51.3 倍
#
# 漏抓的是最輕微的不平衡（0583mg / 1169mg，1.0~2.1 倍）。那是物理事實不是缺陷：
# 最輕的不平衡訊號本來就幾乎與正常無異，而它的正確處置本來就是「持續監測」。
GATE_MILD = 3.0     # 偏離低於此倍數：有跡象但不明確

# 「有跡象但不明確」時，風險分數的上限。
#
# 為什麼要有這個上限：嚴重度應該由物理偏離決定，不是由分類器的機率決定。
# 分類器對 1.5 倍的輕微不平衡一樣給出 0.99 的異常機率（機率飽和），
# 規則引擎照單全收就會算出 6/100 的健康分數 —— 畫面顯示「危急・立即停機」，
# 但頻譜上只有 1.5 倍的偏離。那是對現場的誤導：真的照著停機，
# 停下來會發現沒什麼好修的，下次就沒人相信這個系統了。
#
# 55 分對應規則引擎的 warning 級（50~70），也就是「排程檢查」而不是「立即停機」，
# 這才是 1.5~3 倍偏離的正確處置。
GATE_MILD_RISK_CAP = 55


def physical_gate(evidence: list[dict] | None) -> str:
    """回傳 'quiet'（各項都在基準內）／'mild'（有跡象）／'clear'（明確異常）。"""
    if not evidence:
        return "quiet"
    return "mild" if max(e["倍數"] for e in evidence) < GATE_MILD else "clear"


def _apply_physical_gate(out: dict, diagnosis, evidence: list[dict]) -> str:
    """依物理偏離程度調整結論與嚴重度，就地修改 out。

    quiet -> 推翻分類器的故障結論，判為正常
    mild  -> 保留故障名稱，但把風險壓到「排程檢查」等級
    clear -> 不動
    """
    from knowledge_base import RISK_ACTIONS, get_fault_info

    gate = physical_gate(evidence)

    if gate == "mild" and out["risk_score"] > GATE_MILD_RISK_CAP:
        top = max(e["倍數"] for e in evidence)
        out["risk_score"] = GATE_MILD_RISK_CAP
        out["health_scores"]["綜合健康"] = 100 - GATE_MILD_RISK_CAP
        # 機械健康也要跟著改。它原本是 (1 - 分類器異常機率)×100，
        # 而分類器對 1.5 倍的輕微偏離一樣輸出 0.99 -> 機械健康 0。
        # 畫面上「機械健康 0」配「綜合健康 45」自相矛盾，而且 0 分那條
        # 紅色長條會蓋過真正的結論。嚴重度既然由物理決定，這一項也一樣。
        out["health_scores"]["機械健康"] = max(
            int(out["health_scores"].get("機械健康", 0)), 100 - GATE_MILD_RISK_CAP)
        out["risk_level"] = "warning"
        lv = RISK_ACTIONS["warning"]
        out["risk_level_display"] = lv["display"]
        out["action_window"] = lv["window"]
        out["should_stop"] = False
        out["triggered_rules"] = [
            f"R0 物理基準比對：最大偏離為同負載正常基準的 {top:.1f} 倍，"
            f"屬輕度偏離，風險上限設為 {GATE_MILD_RISK_CAP} 分（排程檢查而非立即停機）"
        ] + list(out.get("triggered_rules") or [])
        return gate

    if gate != "quiet" or diagnosis.probable_fault == "Normal":
        return gate

    info = get_fault_info("Normal")
    risk = min(int(out["risk_score"]), 25)
    lv = RISK_ACTIONS["healthy"]

    out.update({
        "risk_score": risk,
        "risk_level": "healthy",
        "risk_level_display": lv["display"],
        "action_window": lv["window"],
        "should_stop": False,
        "probable_fault": "Normal",
        "probable_fault_display": info["display"],
        "causes": info["causes"], "checks": info["checks"],
        "actions": info["actions"], "tools": info["tools"],
        "parts": info["parts"], "safety": info["safety"],
        "effort": info["effort"], "known_confusion": info["known_confusion"],
        "gate_overrode": fault_display(diagnosis.probable_fault),
    })
    out["health_scores"]["綜合健康"] = 100 - risk
    # 同上：物理層說一切都在基準內，機械健康就不能還掛著分類器算出來的 0 分
    out["health_scores"]["機械健康"] = max(
        int(out["health_scores"].get("機械健康", 0)), 100 - risk)
    out["triggered_rules"] = [
        f"R0 物理基準比對：各項頻譜特徵均未超過同負載正常基準 1.5 倍，"
        f"依基準判定為正常"
        f"（分類器判為「{fault_display(diagnosis.probable_fault)}」，"
        f"但頻譜上找不到支持它的譜線，不予採信）"
    ] + list(out.get("triggered_rules") or [])
    return gate


def contextual_fault(fault_display_name: str, risk_score: float,
                     is_normal: bool) -> tuple[str, str]:
    """依嚴重程度調整故障的措辭，回傳 (顯示文字, 補充說明)。

    為什麼需要這個
    --------------
    分類與嚴重度是兩件獨立的事：模型回答「最像哪種故障」，
    健康分數回答「有多嚴重」。資料集裡的轉子不平衡有 5 個嚴重度等級，
    最輕的（0583mg）訊號幾乎與正常無異——模型正確認出不平衡的特徵，
    但異常機率可能只有 0.35，算出來健康 74 分，確實該是綠色。

    但畫面上「轉子不平衡」配綠色會讓人以為系統出錯。
    這裡不改判斷邏輯（邏輯是對的），只改措辭：
    健康度還很好時講「早期跡象」，明確表達「已偵測到但尚不需處理」——
    這正是預測維護的價值所在，不是矛盾。
    """
    if is_normal:
        return "正常", ""
    if risk_score < 30:          # 對應「健康」等級
        return (f"早期跡象：{fault_display_name}",
                "已偵測到特徵但嚴重度低，持續監測即可，無須立即處理")
    if risk_score < 50:          # 「注意」
        return (f"輕度{fault_display_name}",
                "建議納入下次計畫保養一併檢查")
    return fault_display_name, ""


def primary_modality(mode: str) -> str:
    """這個模式的主模型 key。

    load 模式用 vibration（另有獨立的 current 模型做交叉驗證），
    speed 模式用 fused（振動+電流一起輸入的單一模型）。
    """
    models = MODE_SPEC[mode]["models"]
    for k in ("vibration", "fused"):
        if k in models:
            return k
    return next(iter(models))


def sensor_list(mode: str) -> list[str]:
    """這個模式實際使用的感測來源（給側邊列顯示）。"""
    spec = MODE_SPEC[mode]
    if "fused" in spec["models"]:
        s = ["振動", "電流"]
    else:
        s = ["振動"] + (["電流"] if "current" in spec["models"] else [])
    if spec["has_temperature"]:
        s.append("溫度")
    return s


def diagnose_all(mode: str, split: str = "test", progress=None) -> pd.DataFrame:
    """跑完這個模式下所有設備，回傳總覽表格。"""
    equip = list_equipment(mode, split)
    rows = []
    for i, fid in enumerate(equip["設備編號"]):
        r = diagnose(mode, fid, split)
        rows.append({
            "設備編號": fid,
            "健康分數": r["health_scores"].get("綜合健康", 100 - r["risk_score"]),
            "風險分數": r["risk_score"],
            "風險等級": r["risk_level_display"],
            "研判故障": r["probable_fault_display"],
            "實際狀態": r["ground_truth_display"],
            "判斷正確": "✓" if r["correct"] else "✗",
            "處理時限": r["action_window"],
            "立即停機": "是" if r["should_stop"] else "否",
            "運轉條件": r["condition_value"],
        })
        if progress:
            progress((i + 1) / len(equip), fid)
    return pd.DataFrame(rows)
