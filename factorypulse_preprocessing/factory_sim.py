"""
廠房設備模擬層：把「錄音檔」對應成「CNC 機台」。

為什麼需要這層
--------------
公開資料集是同一台實驗機台在不同故障狀態下的 45 段錄音。直接把 45 段錄音
當成 45 台設備，概念上不誠實（0Nm_BPFI_03 與 0Nm_BPFI_10 其實是同一台機器
裝了不同嚴重度的軸承）。

這裡改成比較貼近實際場景的敘事：
  - 廠內有 10 台**同型號** CNC（FP-01 ~ FP-10），每台上面都是同一款主軸馬達
  - 每台機器對應一段真實錄音，代表它「目前的健康狀態」
  - 全部同型號是刻意的：資料本來就來自同一台機台，宣稱不同機型等於暗示
    模型能跨機型使用，但我們沒有資料支持那個說法。而「一套模型管理整廠
    同型設備」本來就是這類系統的典型部署方式。

數值為什麼會變動
----------------
每分鐘取一段**不同時間區間**的訊號窗來計算。這不是亂數模擬——真實感測訊號
本來就會逐窗波動，我們只是把那個波動呈現出來。相鄰兩分鐘的觀測窗大幅重疊，
所以分數會平滑變化，不會出現「溫度從 50 度跳到 10000 度」那種假數字。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import dashboard_data as dd

# 每台機器每次觀測用幾個時間窗；相鄰分鐘之間的滑動步長
WINDOWS_PER_OBSERVATION = 12
SLIDE_PER_MINUTE = 2

CNC_MODEL = "VMC-850 立式綜合加工中心機"
MOTOR_MODEL = "3ϕ induction motor・3010 RPM 額定"


@dataclass
class Machine:
    machine_id: str      # FP-01
    name: str            # 顯示名稱
    source: str          # 對應的錄音 file_id
    station: str         # 廠房區域，純粹讓畫面有廠區感


# 挑選原則：讓 demo 畫面同時看得到健康、輕微、中度、嚴重四種狀態，
# 這樣總覽頁才有東西可看，也才能展示分級告警的價值。
FLEET_LOAD = [
    Machine("FP-01", "粗銑加工站",   "0Nm_Normal",            "A 線"),
    Machine("FP-02", "精銑加工站",   "2Nm_Normal",            "A 線"),
    Machine("FP-03", "鑽孔站",       "4Nm_Normal",            "A 線"),
    Machine("FP-04", "攻牙站",       "0Nm_Unbalance_0583mg",  "A 線"),
    Machine("FP-05", "側銑站",       "2Nm_Misalign_01",       "A 線"),
    Machine("FP-06", "重切削站",     "4Nm_Unbalance_1751mg",  "B 線"),
    Machine("FP-07", "模具加工站",   "0Nm_BPFI_03",           "B 線"),
    Machine("FP-08", "曲面加工站",   "2Nm_BPFO_10",           "B 線"),
    Machine("FP-09", "深孔加工站",   "4Nm_BPFI_30",           "B 線"),
    Machine("FP-10", "多軸加工站",   "4Nm_BPFO_30",           "B 線"),
]

FLEET_SPEED = [
    Machine("FP-01", "粗銑加工站",   "normal_0", "A 線"),
    Machine("FP-02", "精銑加工站",   "normal_3", "A 線"),
    Machine("FP-03", "鑽孔站",       "normal_5", "A 線"),
    Machine("FP-04", "攻牙站",       "ball_1",   "A 線"),
    Machine("FP-05", "側銑站",       "ball_4",   "A 線"),
    Machine("FP-06", "重切削站",     "inner_2",  "B 線"),
    Machine("FP-07", "模具加工站",   "inner_6",  "B 線"),
    Machine("FP-08", "曲面加工站",   "outer_0",  "B 線"),
    Machine("FP-09", "深孔加工站",   "outer_3",  "B 線"),
    Machine("FP-10", "多軸加工站",   "outer_6",  "B 線"),
]


def fleet(mode: str) -> list[Machine]:
    return FLEET_LOAD if mode == "load" else FLEET_SPEED


def machine_by_id(mode: str, mid: str) -> Machine:
    for m in fleet(mode):
        if m.machine_id == mid:
            return m
    raise KeyError(mid)


def current_minute() -> int:
    """目前是第幾分鐘（自 epoch 起算）。用來決定要看訊號的哪一段。"""
    return int(time.time() // 60)


def observation_slice(n_total: int, minute: int,
                      size: int = WINDOWS_PER_OBSERVATION,
                      slide: int = SLIDE_PER_MINUTE) -> np.ndarray:
    """決定這一分鐘要看哪些時間窗。

    採「連續區間 + 每分鐘往前滑動」而不是隨機抽樣：
      - 隨機抽樣會讓相鄰兩分鐘的分數毫無關聯，看起來像亂跳的假數字
      - 連續區間滑動時，相鄰分鐘共用大部分的窗，分數自然平滑變化，
        也符合真實監測系統「持續觀察一段時間」的行為

    區間走到底就繞回開頭，讓 demo 可以一直跑下去。
    """
    size = min(size, n_total)
    span = max(1, n_total - size + 1)
    start = (minute * slide) % span
    return np.arange(start, start + size)


def snapshot(mode: str, minute: int | None = None) -> list[dict]:
    """回傳這一分鐘全廠 10 台機器的狀態。"""
    if minute is None:
        minute = current_minute()

    out = []
    for m in fleet(mode):
        r = dd.diagnose(mode, m.source, window_minute=minute)
        health = r["health_scores"].get("綜合健康", 100 - r["risk_score"])
        out.append({
            "machine_id": m.machine_id,
            "name": m.name,
            "station": m.station,
            "source": m.source,
            "health": health,
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level_display"],
            "fault": r["fault_label"],          # 已依嚴重度調整措辭
            "fault_note": r["fault_note"],
            # 主模型在兩種模式下 key 不同（vibration / fused），用 primary_key 取
            "confidence": r["modality_detail"][r["primary_key"]]["confidence"],
            "should_stop": r["should_stop"],
            "action_window": r["action_window"],
            "diagnosis": r,
        })
    return out
