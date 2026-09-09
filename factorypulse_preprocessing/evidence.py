"""
證據層：把模型特徵翻譯成「這台機器實際量到什麼」
================================================

為什麼需要這一層
----------------
原本的流程是：模型判定類別 -> 知識庫用類別當 key 查出一段故障說明 -> LLM 潤稿。
問題是整條路徑上沒有任何一個數字來自這台設備本身。評審問「你怎麼知道是外環
故障」，得到的會是教科書通論，不是證據。這跟「可解釋 AI」的定位是矛盾的。

這一層做的事：拿這台設備實際算出來的特徵，跟「同負載下正常機台」的基準值比，
算出「幾倍於正常」，再附上該特徵對應的物理意義。輸出長這樣：

    包絡譜外環通過頻率(183.5 Hz)譜線強度   實測 59.6  基準 3.3   17.9 倍
      -> 軸承外環有局部剝落或裂痕時，滾珠每次滾過缺陷會產生衝擊，
         在包絡譜的 BPFO 頻率形成譜線。

這種句子 LLM 拿到才有東西可講，而且講的每一個數字都可以回頭驗證。

實測驗證（全資料集中位數 / 同負載 Normal 基準）：
    故障類別      env_BPFO   env_BPFI   ratio_3x_1x   order_1x_dominance
    BPFO            17.9x       1.8x        0.3x            0.3x
    BPFI             0.6x      20.6x        0.9x            0.2x
    Misalign         0.8x       1.0x        7.2x            0.8x
    Unbalance        1.0x       1.1x        0.4x            2.3x
每一類都由不同的特徵點亮，對角線很乾淨 —— 這張表本身就是一頁簡報。

用法
----
    import evidence
    ev = evidence.extract(feature_row, load_nm=4)      # -> list[dict]
    print(evidence.format_for_llm(ev))
"""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE_PATH = Path(__file__).parent / "results" / "evidence_reference.json"

# ---------------------------------------------------------------------------
# 特徵 -> (顯示名稱, 物理意義, 異常方向)
#
# 異常方向 "high" = 數值越高越異常；"low" = 數值越低越異常。
#
# 為什麼需要 "low" 這個方向：env_*_sb_ratio 定義是「邊帶 / 主譜線」，
# 展開之後會發現它是「主譜線相對邊帶的突出程度」的倒數：
#       env_X_sb       = 邊帶 / 噪底
#       env_X          = 主譜線 / 噪底
#       env_X_sb_ratio = 邊帶 / 主譜線 = 1 / 突出度
# 所以它的值「越低」代表主譜線越孤立尖銳、故障訊號越明確。
# 實測完全吻合：BPFO 檔案的 BPFO_sb_ratio = 0.13（主譜線極突出），
#               BPFI 檔案的 BPFO_sb_ratio = 1.10（該處沒有突出譜線）。
#
# 這個特徵在模型裡的重要度排前十，如果解釋層完全不收，評審看特徵重要度
# 時會問到一個我們自己講不出來的欄位。與其迴避，不如把方向定義清楚、
# 用正確的說法呈現。（先前 BSF/FTF 的狀況不同：那是真的量錯位置，只能移除。）
# ---------------------------------------------------------------------------
FEATURE_PHYSICS = {
    "env_BPFO": (
        "包絡譜・外環通過頻率譜線",
        "軸承外環有局部剝落或裂痕時，滾珠每次滾過缺陷會產生衝擊，"
        "激發軸承座共振；解調後在包絡譜的 BPFO 頻率形成譜線。",
        "high",
    ),
    "env_BPFI": (
        "包絡譜・內環通過頻率譜線",
        "軸承內環缺陷的特徵頻率。內環隨軸旋轉，缺陷會週期性進出承載區，"
        "因此譜線通常伴隨轉頻邊帶。",
        "high",
    ),
    "env_BPFO_h2": ("包絡譜・外環頻率二次諧波", "外環缺陷衝擊的諧波成分，缺陷擴大時會變明顯。", "high"),
    "env_BPFI_h2": ("包絡譜・內環頻率二次諧波", "內環缺陷衝擊的諧波成分。", "high"),
    "env_BPFO_sb_ratio": (
        "外環譜線的孤立程度",
        "外環通過頻率的主譜線相對於其兩側邊帶有多突出（數值為邊帶／主譜線，"
        "所以越低代表主譜線越孤立尖銳）。外環缺陷位置固定不動、不受轉軸調變，"
        "因此譜線乾淨、邊帶弱，這個值會明顯偏低。",
        "low",
    ),
    "env_BPFI_sb_ratio": (
        "內環譜線的孤立程度",
        "內環通過頻率的主譜線相對於其兩側邊帶有多突出（越低代表越孤立）。"
        "內環缺陷隨轉軸進出承載區、振幅被轉頻調變，理應產生較強的邊帶，"
        "因此這個值通常不會像外環那麼低。",
        "low",
    ),
    "order_1x": ("轉頻基頻(1x)振幅", "轉子質量分布不均會在每轉產生一次離心力，直接反映在 1x。", "high"),
    # order_2x / ratio_2x_1x 刻意不收進解釋層。
    # 教科書用 2 倍頻判斷不對心，但本測試台實測 ratio_2x_1x 在 Misalign 是 0.66，
    # 比 Normal 的 1.19 還低，方向完全相反。留在模型裡無妨（樹模型會自己權衡），
    # 但拿它當「證據」呈現，等於在畫面上放一個我們自己知道不成立的說法。
    "order_3x": ("轉頻三次諧波(3x)振幅", "本測試台的不對心主要能量落在 3x（實測，非教科書預期的 2x）。", "high"),
    "ratio_3x_1x": (
        "3x/1x 諧波比",
        "不對心會讓高次諧波相對於基頻抬升。本測試台實測 Misalign 為 2.70，"
        "其餘四類皆在 0.13~0.35，是最乾淨的不對心判準；且為比值，跨負載穩定。",
        "high",
    ),
    "order_1x_dominance": (
        "1x 能量佔比",
        "1x 振幅佔整體頻譜能量的比重。不平衡會讓能量集中在基頻，"
        "這個比值比絕對振幅更不受負載影響。",
        "high",
    ),
    "kurtosis": ("波形峰度", "衡量訊號的衝擊性。軸承局部缺陷產生的脈衝會讓峰度高於高斯訊號的 3。", "high"),
    "crest_factor": ("峰值因子", "峰值與 RMS 的比。早期軸承故障時峰值先上升，峰值因子因此提前反應。", "high"),
    "impulse_factor": ("脈衝因子", "峰值與平均絕對值的比，對衝擊型故障比 RMS 敏感。", "high"),
}

CHANNEL_NAMES = {
    0: "軸承座A・X向", 1: "軸承座A・Y向",
    2: "軸承座B・X向", 3: "軸承座B・Y向",
}

# 幾倍於正常才算「證據」。低於這個倍數的特徵不會被列出來，
# 避免 LLM 拿一堆 1.1 倍的雜訊當作診斷依據。
MIN_RATIO = 1.5


def load_reference(path: Path | str = REFERENCE_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到基準檔 {p}。請先跑 build_reference.py 或 build_dataset.py。")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _num(x: float) -> str:
    """數值格式化。像 order_1x_dominance 這種 1e-4 量級的特徵，
    固定四位小數會讓「實測 0.0001 / 基準 0.0001 / 2.2 倍」看起來自相矛盾，
    所以小數值改用有效位數表示。"""
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 0.01:
        return f"{x:.3g}"
    if ax < 1:
        return f"{x:.4f}"
    return f"{x:.2f}"


def _split(col: str):
    """'ch0_env_BPFO' -> (0, 'env_BPFO')"""
    if not col.startswith("ch") or "_" not in col:
        return None, col
    ch, _, rest = col[2:].partition("_")
    try:
        return int(ch), rest
    except ValueError:
        return None, col


# 哪些特徵支持哪個故障類別。用來標註「與診斷結論不一致」的證據項。
FEATURE_SUPPORTS = {
    "env_BPFO": "BPFO", "env_BPFO_h2": "BPFO", "env_BPFO_sb_ratio": "BPFO",
    "env_BPFI": "BPFI", "env_BPFI_h2": "BPFI", "env_BPFI_sb_ratio": "BPFI",
    "order_3x": "Misalign", "ratio_3x_1x": "Misalign",
    "order_1x": "Unbalance", "order_1x_dominance": "Unbalance",
}


def extract(feature_row: dict, load_nm: float, top: int = 5,
            reference: dict | None = None, fault_type: str | None = None,
            dedup_by_feature: bool = True) -> list[dict]:
    """feature_row: 單一設備的特徵（dict 或 pandas Series 都可）
    load_nm: 這台設備目前的負載，用來挑對應的正常基準
    fault_type: 模型判定的故障類別。有給的話會標註哪些證據與結論不一致。
    dedup_by_feature: 同一種量測項目只保留偏離最大的那個通道。

    回傳依「偏離正常的程度」排序的證據清單。

    兩個設計決定：

    1. 支援「低值代表異常」的特徵（direction="low"）。
       env_*_sb_ratio 就是這種——它是主譜線突出程度的倒數，越低越異常。
       偏離程度統一換算成「倍」，低值型的算法是 基準/實測。

    2. 同一種量測項目跨四個通道會產生四筆幾乎一樣的證據，
       塞滿 top 5 之後畫面上只剩兩種物理量，資訊量很低。
       改成每種量測項目只留偏離最大的通道。
    """
    ref = reference or load_reference()
    key = str(int(load_nm))
    if key not in ref["baseline"]:
        key = sorted(ref["baseline"])[0]
    base = ref["baseline"][key]

    rows = []
    for col, value in dict(feature_row).items():
        if col not in base:
            continue
        ch, feat = _split(col)
        if feat not in FEATURE_PHYSICS:
            continue
        b = base[col]["median"]
        if abs(b) < 1e-12:
            continue

        name, why, direction = FEATURE_PHYSICS[feat]
        value = float(value)
        if direction == "low":
            if value <= 1e-12:
                continue
            deviation = b / value          # 比正常低幾倍
            trend = "低於正常"
        else:
            deviation = value / b
            trend = "高於正常"
        if deviation < MIN_RATIO:
            continue

        supports = FEATURE_SUPPORTS.get(feat)
        rows.append({
            "feature": col,
            "特徵種類": feat,
            "名稱": name,
            "位置": CHANNEL_NAMES.get(ch, f"ch{ch}"),
            "實測值": round(value, 4),
            "正常基準": round(b, 4),
            "倍數": round(deviation, 1),
            "方向": trend,
            "支持類別": supports,
            # 這一項指向的故障，與模型的結論不同 -> 需要在畫面與 LLM 提示裡標明，
            # 否則會出現「診斷：外環故障」旁邊列著「內環頻率異常 8 倍」而無人解釋。
            "與結論一致": (supports is None or fault_type is None
                          or supports == fault_type),
            "物理意義": why,
        })

    rows.sort(key=lambda r: (not r["與結論一致"], -r["倍數"]))

    if dedup_by_feature:
        seen, kept = set(), []
        for r in rows:
            if r["特徵種類"] in seen:
                continue
            seen.add(r["特徵種類"])
            kept.append(r)
        rows = kept
    return rows[:top]


def severity_series(feature_frame, load_nm: float, fault_type: str | None = None,
                    reference: dict | None = None) -> dict | None:
    """逐窗的「物理嚴重度」：每個時間窗實測值相對同負載正常基準的倍數。

    為什麼需要這個（原本的趨勢圖畫錯東西了）
    ----------------------------------------
    「異常程度隨時間變化」原本畫的是分類器的 1 - P(正常)。那是**決策**不是**量測**，
    而且樹模型在明確樣本上機率會飽和，畫出來是一條水平線。實測 4Nm_BPFO_30：

        分類器異常分數   17 個窗全是 0.9999339，全距 0.000002（換成 log-odds 也只有 0.03）
        包絡譜 BPFO 譜線 13.77× → 15.97×，變異係數 3.8%

    同一批訊號，模型機率動也不動，物理量卻有真實起伏。設備健康監測要看的是
    後者：它會隨負載、轉速、缺陷擴展而變化，而且不必相信模型就能自己核對。

    挑哪一個特徵畫
    --------------
    優先選「支持模型結論、且偏離最大」的那個通道（例如判外環故障就畫 BPFO 譜線）。
    判正常或找不到支持項時，退而畫偏離最大的那一項——正常機台看的就是
    「所有物理量都貼著 1.0 倍」，那條線本身就是證據。

    回傳 None 代表算不出來（沒有基準檔、或該負載沒有對應基準），呼叫端要能接受。
    """
    try:
        ref = reference or load_reference()
    except FileNotFoundError:
        return None

    key = str(int(load_nm))
    if key not in ref["baseline"]:
        key = sorted(ref["baseline"])[0]
    base = ref["baseline"][key]

    cols = [c for c in feature_frame.columns if c in base]
    if not cols:
        return None

    median_row = feature_frame[cols].median(numeric_only=True).to_dict()
    ranked = extract(median_row, load_nm=load_nm, top=8, reference=ref,
                     fault_type=fault_type, dedup_by_feature=True)
    if not ranked:
        # 完全沒有超過 MIN_RATIO 的項目 -> 這台看起來正常。
        # 挑一個代表性的軸承特徵來畫，讓「正常」也有一條線可以看。
        for cand in ("ch0_env_BPFO", "ch1_env_BPFO", "ch0_order_1x"):
            if cand in cols:
                pick = {"feature": cand, "特徵種類": _split(cand)[1],
                        "名稱": FEATURE_PHYSICS[_split(cand)[1]][0],
                        "位置": CHANNEL_NAMES.get(_split(cand)[0], cand),
                        "方向": "高於正常",
                        "物理意義": FEATURE_PHYSICS[_split(cand)[1]][1]}
                break
        else:
            return None
    else:
        supporting = [e for e in ranked if e.get("與結論一致", True)]
        pick = (supporting or ranked)[0]

    col = pick["feature"]
    b = base[col]["median"]
    if abs(b) < 1e-12:
        return None

    values = feature_frame[col].astype(float).to_numpy()
    direction = FEATURE_PHYSICS[pick["特徵種類"]][2]
    if direction == "low":
        safe = values.copy()
        safe[abs(safe) < 1e-12] = 1e-12
        ratios = b / safe
    else:
        ratios = values / b

    return {
        "feature": col,
        "名稱": pick["名稱"],
        "位置": pick["位置"],
        "方向": direction,
        "物理意義": pick["物理意義"],
        "正常基準": float(b),
        "ratios": [float(v) for v in ratios],
        "實測值": [float(v) for v in values],
    }


def format_for_llm(evidence: list[dict], shaft_freq: float = 50.15,
                   orders: dict | None = None) -> str:
    """把證據清單排版成注入 system prompt 的文字區塊。

    刻意把頻率換算成 Hz 一起寫出來，這樣 LLM 生成的句子裡會出現具體頻率，
    使用者可以拿去跟頻譜圖對照 —— 這是「可驗證的解釋」與「聽起來很專業的
    空話」之間的差別。
    """
    if not evidence:
        return "  （本次量測未發現明顯偏離正常基準的特徵，各項指標皆在正常範圍內。）"

    orders = orders or {"BPFO": 3.66, "BPFI": 5.36}
    lines = []
    for i, e in enumerate(evidence, 1):
        feat = e["feature"]
        hz = ""
        for name, o in orders.items():
            if name in feat:
                # _h2 是二次諧波，頻率要乘 2，否則會印出跟基頻一樣的數字
                mult = 2 if feat.endswith("_h2") else 1
                hz = f"（{shaft_freq * o * mult:.1f} Hz）"
                break
        if "order_1x" in feat:
            hz = f"（{shaft_freq:.1f} Hz）"
        elif "order_2x" in feat or "2x_1x" in feat:
            hz = f"（{shaft_freq * 2:.1f} Hz）"
        elif "order_3x" in feat or "3x_1x" in feat:
            hz = f"（{shaft_freq * 3:.1f} Hz）"

        trend = e.get("方向", "高於正常")
        rel = (f"為正常值的 {e['倍數']} 倍" if trend == "高於正常"
               else f"僅為正常值的 1/{e['倍數']}（{trend}）")
        flag = "" if e.get("與結論一致", True) else \
            "\n     ⚠️ 此項指向的故障類別與模型結論不同，屬旁證，不可據此推翻結論。"
        lines.append(
            f"  {i}. {e['名稱']}{hz} @ {e['位置']}\n"
            f"     實測 {_num(e['實測值'])}，同負載正常基準 {_num(e['正常基準'])}，{rel}\n"
            f"     物理意義：{e['物理意義']}{flag}"
        )
    return "\n".join(lines)


def fallback_summary(diag: dict, evidence: list[dict]) -> str:
    """LLM 不可用時的離線摘要。

    demo 當天沒有網路、API 額度用完、或 Gemini 模型下架時，
    畫面不能開天窗。這個函式不呼叫任何外部服務，
    純粹用診斷結果 + 證據清單組出一份可讀的報告。
    內容會比 LLM 生成的生硬，但每個數字都是真的，該講的都有講到。
    """
    fault = (diag.get("fault_label") or diag.get("probable_fault_display")
             or diag.get("fault_type") or "未知")
    risk = diag.get("risk_level_display", diag.get("risk_level", "未知"))
    conf = diag.get("confidence")
    conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "未知"
    stop = "建議盡快安排停機檢修" if diag.get("should_stop") else "可繼續運轉，但需持續監測"

    parts = [
        f"【診斷結論】研判為「{fault}」，風險等級 {risk}，逐窗判斷一致率 {conf_s}。{stop}。",
        "",
        "【量測證據】",
    ]
    if evidence:
        for e in evidence:
            parts.append(
                f"  ・{e['名稱']} @ {e['位置']}："
                f"實測 {_num(e['實測值'])}，"
                f"為同負載正常基準（{_num(e['正常基準'])}）的 {e['倍數']} 倍。"
            )
    else:
        parts.append("  ・各項特徵皆在正常範圍內，未偵測到明顯異常證據。")

    rules = diag.get("triggered_rules") or []
    if rules:
        parts += ["", "【觸發的判斷規則】"] + [f"  ・{r}" for r in rules]

    for title, key in [("可能原因", "causes"), ("建議檢查", "checks"),
                       ("建議行動", "actions"), ("需要的工具", "tools"),
                       ("可能需要的備品", "parts"), ("安全注意事項", "safety")]:
        items = diag.get(key) or []
        if items:
            parts += ["", f"【{title}】"] + [f"  ・{c}" for c in items]

    if diag.get("effort"):
        parts += ["", f"【預估工時】{diag['effort']}"]

    parts += ["", "（本則為離線摘要：LLM 服務目前無法連線，"
                  "以上內容由診斷結果與知識庫直接產生，未經語言模型潤飾。）"]
    return "\n".join(parts)


if __name__ == "__main__":
    import pandas as pd

    ref = load_reference()
    print("基準檔負載:", list(ref["baseline"]))
    print(f"轉頻 {ref['shaft_freq_hz']} Hz，"
          f"BPFO={ref['bearing_orders']['BPFO']} BPFI={ref['bearing_orders']['BPFI']}\n")

    # 用真實特徵檔跑一次，確認每個故障類別點亮的是對的證據
    v = pd.concat([pd.read_csv(f"processed/vibration_v2_features_{s}.csv")
                   for s in ["train", "val", "test"]], ignore_index=True)
    for cond in ["Normal", "BPFO", "BPFI", "Misalign", "Unbalance"]:
        g = v[(v.condition == cond) & (v.load_nm == 4)]
        if g.empty:
            continue
        row = g.median(numeric_only=True).to_dict()
        ev = extract(row, load_nm=4, top=3, reference=ref)
        print(f"=== {cond}（4Nm，取中位數窗）===")
        print(format_for_llm(ev, ref["shaft_freq_hz"], ref["bearing_orders"]))
        print()
