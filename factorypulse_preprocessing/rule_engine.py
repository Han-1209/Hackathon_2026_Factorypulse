"""
FactoryPulse 規則引擎：把各模態的模型輸出融合成一份結構化診斷。

設計原則
--------
1. 分類器只回答「是什麼故障、多確定」，不回答「多嚴重、要不要停機」。
2. 規則引擎負責跨模態交叉驗證、決定風險等級、判斷是否停機。
3. 溫度在這裡才發揮作用：它不參與故障分類（樣本太少會造成標籤洩漏），
   但負責回答「異常是否在惡化」「有沒有過熱風險」。
4. 輸出是結構化 dict，之後交給 LLM 轉自然語言。LLM 不得自行編造故障。

門檻怎麼訂的
------------
以下數值是從這份資料集 45 個檔案的實際分布算出來的，不是憑感覺：

  A_temp_delta_from_baseline_mean（相對同負載 Normal 基準的溫差）
      Normal     -0.12°C     Unbalance   0.42°C
      BPFI        1.40°C     BPFO        3.18°C     Misalign  3.79°C
      全體：中位數 1.52、75 分位 3.28、最大 5.25

  A_temp_trend_slope（°C/秒）
      全體：中位數 0.0061、最大 0.0205
      換算成 10 分鐘：中位數約 +3.7°C、最大約 +12°C

⚠️ 重要限制：這份資料集每段錄音只有 60~300 秒，溫升幅度本來就小，
   而且不同檔案是不同日期錄的（環境溫度不同）。這些門檻適合用於本專案 demo，
   實際部署到工廠時必須用該設備自己的長期資料重新校正。
   簡報時建議主動說明這點。
"""

from dataclasses import dataclass, field, asdict

from knowledge_base import get_fault_info, RISK_ACTIONS

# ---------------------------------------------------------------- 門檻設定
TEMP_DELTA_SLIGHT = 1.0    # °C，高於基準多少算「輕微升溫」
TEMP_DELTA_CLEAR = 3.0     # 「明顯升溫」
TEMP_DELTA_OVERHEAT = 5.0  # 「過熱」

TEMP_SLOPE_RISING = 0.005   # °C/秒，約 +3°C/10分鐘 -> 緩慢上升
TEMP_SLOPE_FAST = 0.010     # 約 +6°C/10分鐘 -> 快速上升

ANOMALY_HIGH = 0.80        # 異常機率高於此視為「明確異常」
ANOMALY_MID = 0.50         # 高於此視為「可疑」

# 融合權重（聲音模態因資料不足已移除，權重重新分配）
WEIGHT_VIBRATION = 0.65
WEIGHT_CURRENT = 0.25
WEIGHT_TEMPERATURE = 0.10


@dataclass
class ModalityInput:
    """單一模態的模型輸出。"""
    anomaly_prob: float          # 異常機率 0~1
    fault_type: str = "Normal"   # 分類結果
    confidence: float = 1.0      # 模型自評可信度 0~1


@dataclass
class TemperatureInput:
    """溫度不做分類，只提供趨勢數值。"""
    delta_from_baseline: float   # °C，相對該設備同負載正常基準
    trend_slope: float           # °C/秒


@dataclass
class Diagnosis:
    risk_score: int
    risk_level: str
    risk_level_display: str
    action_window: str
    should_stop: bool
    probable_fault: str
    probable_fault_display: str
    health_scores: dict
    temperature_status: str
    evidence: list = field(default_factory=list)
    triggered_rules: list = field(default_factory=list)
    causes: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    # 知識庫在 2026/08 擴充了工具 / 備品 / 安全 / 工時四個欄位。
    # 這裡一併帶出去，讓「設備診斷」頁與離線摘要拿得到同樣的內容，
    # 不必各自再去查一次知識庫（查兩次就會有兩份可能不同步的邏輯）。
    tools: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    safety: list = field(default_factory=list)
    effort: str = ""
    signature: str = ""
    known_confusion: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 溫度判讀
def assess_temperature(temp: TemperatureInput) -> tuple[str, float]:
    """回傳 (狀態文字, 異常分數 0~1)。"""
    d, s = temp.delta_from_baseline, temp.trend_slope

    if d >= TEMP_DELTA_OVERHEAT:
        status, score = "過熱", 0.95
    elif d >= TEMP_DELTA_CLEAR:
        status, score = "明顯升溫", 0.65
    elif d >= TEMP_DELTA_SLIGHT:
        status, score = "輕微升溫", 0.35
    elif d <= -TEMP_DELTA_SLIGHT:
        status, score = "低於基準", 0.10
    else:
        status, score = "正常", 0.05

    # 上升速率會加重判斷：即使目前溫差不大，快速上升代表正在惡化
    if s >= TEMP_SLOPE_FAST:
        status += "（快速上升中）"
        score = min(1.0, score + 0.25)
    elif s >= TEMP_SLOPE_RISING:
        status += "（緩慢上升中）"
        score = min(1.0, score + 0.10)

    return status, score


# ---------------------------------------------------------------- 主流程
def diagnose(vibration: ModalityInput,
             current: ModalityInput,
             temperature: TemperatureInput,
             load_nm: float | None = None,
             use_current: bool = True) -> Diagnosis:
    """use_current=False：電流不參與判斷。

    ⚠️ 這個開關不是可有可無的。呼叫端在電流不可用時，為了湊滿介面會把振動的
    結果同時當成電流傳進來；如果照常跑跨感測規則，R1「振動與電流同時異常」
    就會對每一台都成立 —— 畫面上出現「兩個獨立來源互相驗證」，
    但實際上只有一個來源，講了兩次。那是最糟的一種假證據。

    關掉之後：電流的權重併回振動，所有跨感測規則（R1/R2/R3/R6）一律不評估。
    """
    temp_status, temp_score = assess_temperature(temperature)

    # ---- 基礎風險分數（加權平均）----
    w_vib = WEIGHT_VIBRATION + (0.0 if use_current else WEIGHT_CURRENT)
    w_cur = WEIGHT_CURRENT if use_current else 0.0
    base = (vibration.anomaly_prob * w_vib
            + current.anomaly_prob * w_cur
            + temp_score * WEIGHT_TEMPERATURE)
    risk = base * 100

    rules, evidence = [], []
    should_stop = False

    def bump(amount: float):
        """規則加分採「剩餘空間的比例」而不是直接相加。

        直接相加會讓多條規則同時觸發時輕易衝到 100 分、綜合健康變成 0，
        看起來不像真實系統。改成往上限逼近的方式，分數會自然收斂在 90 幾分，
        保留「還有更糟的狀況」的空間。
        """
        nonlocal risk
        risk += amount * (100 - risk) / 100

    vib_abnormal = vibration.anomaly_prob >= ANOMALY_HIGH
    vib_suspect = vibration.anomaly_prob >= ANOMALY_MID
    # 電流不可用時一律當成「沒有意見」，跨感測規則就不會誤觸發
    cur_abnormal = use_current and current.anomaly_prob >= ANOMALY_HIGH
    cur_suspect = use_current and current.anomaly_prob >= ANOMALY_MID
    temp_rising = temperature.trend_slope >= TEMP_SLOPE_RISING
    temp_hot = temperature.delta_from_baseline >= TEMP_DELTA_CLEAR

    # 規則 1：振動與電流同時異常 -> 機械故障可信度提高
    if vib_abnormal and cur_abnormal:
        bump(12)
        rules.append("R1 振動與電流同時異常，機械故障可信度提高")

    # 規則 2：電流異常且溫度上升 -> 過載或摩擦
    if cur_suspect and temp_rising:
        bump(10)
        rules.append("R2 電流異常且溫度持續上升，研判為過載或摩擦阻力增加")

    # 規則 3：只有電流異常、振動正常 -> 可能是負載變動而非機械故障
    if cur_abnormal and not vib_suspect:
        risk -= 8
        rules.append("R3 僅電流異常而振動正常，可能為負載變動，建議重新採樣確認")

    # 規則 4：振動異常且溫度明顯升高 -> 異常正在惡化
    if vib_abnormal and temp_hot:
        bump(15)
        rules.append("R4 振動異常且溫度明顯高於基準，異常正在惡化")

    # 規則 5：過熱 -> 直接進入停機判斷
    if temperature.delta_from_baseline >= TEMP_DELTA_OVERHEAT:
        risk = max(risk, 85)
        should_stop = True
        rules.append("R5 溫度超過基準 5°C，過熱風險，建議立即停機")

    # 規則 6：三個訊號全部異常 -> 至少高風險
    if vib_abnormal and cur_abnormal and temp_score >= 0.5:
        risk = max(risk, 75)
        rules.append("R6 三個感測來源同時異常，風險等級至少為高風險")

    # 規則 7：模型可信度低 -> 風險打折並標記需複檢
    if vibration.confidence < 0.5:
        risk *= 0.85
        rules.append("R7 振動模型可信度偏低，建議重新採樣複檢")

    # 規則 8（下限保護）：只要有任何一個模態明確異常，就不得回報「健康」。
    # 沒有這條的話，R3 的扣分會把「電流明確異常但振動正常」壓到 27 分 = 健康，
    # 等於叫使用者忽略一個確實測到的異常，這在預測維護是不可接受的漏報。
    if (vib_abnormal or cur_abnormal) and risk < 30:
        risk = 30
        rules.append("R8 有感測來源明確異常，風險等級不低於「注意」，需人工複檢")

    risk = int(max(0, min(100, round(risk))))

    # ---- 風險等級 ----
    if risk >= 85:
        level = "critical"
        should_stop = True
    elif risk >= 70:
        level = "high"
    elif risk >= 50:
        level = "warning"
    elif risk >= 30:
        level = "attention"
    else:
        level = "healthy"

    # ---- 四面向健康分數 ----
    health = {
        "機械健康": int(round((1 - vibration.anomaly_prob) * 100)),
        "負載健康": int(round((1 - current.anomaly_prob) * 100)) if use_current else None,
        "熱狀態": int(round((1 - temp_score) * 100)),
        "綜合健康": 100 - risk,
    }
    health = {k: v for k, v in health.items() if v is not None}

    # ---- 決定最可能的故障 ----
    # 振動是主要診斷來源；振動正常時才考慮電流的判斷
    if vib_suspect and vibration.fault_type != "Normal":
        fault = vibration.fault_type
    elif use_current and cur_suspect and current.fault_type != "Normal":
        fault = current.fault_type
    else:
        fault = "Normal"

    info = get_fault_info(fault)

    # ---- 證據列表（給「多感測證據頁」用）----
    evidence.append({
        "來源": "振動", "狀態": _level_text(vibration.anomaly_prob),
        "數值": f"異常機率 {vibration.anomaly_prob:.2f}",
        # 這裡以前寫「可信度 0.85」。那個數字其實是逐窗多數決的一致率，
        # 叫「可信度」會被讀成「這個診斷有 85% 是對的」，是兩回事。
        "說明": f"分類為 {get_fault_info(vibration.fault_type)['display']}"
                f"（逐窗一致率 {vibration.confidence:.0%}）",
    })
    evidence.append({
        "來源": "電流", "狀態": _level_text(current.anomaly_prob),
        "數值": f"異常機率 {current.anomaly_prob:.2f}",
        "說明": f"分類為 {get_fault_info(current.fault_type)['display']}",
    })
    evidence.append({
        "來源": "溫度", "狀態": temp_status,
        "數值": f"高於基準 {temperature.delta_from_baseline:+.2f}°C，"
                f"趨勢 {temperature.trend_slope * 600:+.2f}°C/10分鐘",
        "說明": "溫度用於判斷異常是否惡化與過熱風險，不參與故障分類",
    })

    risk_info = RISK_ACTIONS[level]
    return Diagnosis(
        risk_score=risk,
        risk_level=level,
        risk_level_display=risk_info["display"],
        action_window=risk_info["window"],
        should_stop=should_stop,
        probable_fault=fault,
        probable_fault_display=info["display"],
        health_scores=health,
        temperature_status=temp_status,
        evidence=evidence,
        triggered_rules=rules,
        causes=info["causes"],
        checks=info["checks"],
        actions=info["actions"],
        tools=info["tools"],
        parts=info["parts"],
        safety=info["safety"],
        effort=info["effort"],
        signature=info["signature"],
        known_confusion=info["known_confusion"],
    )


def _level_text(p: float) -> str:
    if p >= ANOMALY_HIGH:
        return "高異常"
    if p >= ANOMALY_MID:
        return "中度異常"
    if p >= 0.3:
        return "輕微偏離"
    return "正常"


# ---------------------------------------------------------------- 自我測試
if __name__ == "__main__":
    scenarios = [
        ("① 全部正常",
         ModalityInput(0.05, "Normal", 0.95), ModalityInput(0.08, "Normal", 0.9),
         TemperatureInput(-0.1, 0.004)),
        ("② 早期軸承異常（僅振動）",
         ModalityInput(0.88, "BPFI", 0.85), ModalityInput(0.20, "Normal", 0.9),
         TemperatureInput(0.5, 0.005)),
        ("③ 振動+電流異常，溫度上升",
         ModalityInput(0.91, "BPFO", 0.88), ModalityInput(0.85, "Normal", 0.86),
         TemperatureInput(3.4, 0.012)),
        ("④ 僅電流異常（疑似負載變動）",
         ModalityInput(0.20, "Normal", 0.9), ModalityInput(0.86, "Normal", 0.88),
         TemperatureInput(0.2, 0.003)),
        ("⑤ 過熱（應觸發立即停機）",
         ModalityInput(0.75, "Misalign", 0.8), ModalityInput(0.6, "Normal", 0.85),
         TemperatureInput(5.6, 0.018)),
        ("⑥ 振動異常但模型沒把握",
         ModalityInput(0.82, "BPFI", 0.35), ModalityInput(0.15, "Normal", 0.9),
         TemperatureInput(0.3, 0.004)),
    ]

    for name, vib, cur, tmp in scenarios:
        d = diagnose(vib, cur, tmp)
        print("=" * 68)
        print(f"{name}")
        print(f"  風險 {d.risk_score}/100 → {d.risk_level_display}"
              f"　處理時限：{d.action_window}　立即停機：{'是' if d.should_stop else '否'}")
        print(f"  研判故障：{d.probable_fault_display}")
        print(f"  健康分數：{d.health_scores}")
        print(f"  溫度狀態：{d.temperature_status}")
        if d.triggered_rules:
            print("  觸發規則：")
            for r in d.triggered_rules:
                print(f"    - {r}")
        if d.actions:
            print(f"  建議行動：{'、'.join(d.actions)}")
        print()
