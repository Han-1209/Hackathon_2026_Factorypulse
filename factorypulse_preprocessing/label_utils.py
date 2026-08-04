"""
從檔名解析標籤：load(負載) / condition(故障類型) / severity(嚴重度代碼) / severity_level(序數等級)

已知的原始檔名拼字問題（不用去改資料夾裡的檔案，這裡直接處理）：
  - 2Nm 振動資料夾裡 Unbalance 被打成 "Unbalalnce"
"""

import re
from pathlib import Path
from config import SEVERITY_ORDER, FAULT_CLASSES

_CONDITION_ALIASES = {
    "unbalalnce": "Unbalance",  # 原始資料集拼字錯誤
    "unbalance": "Unbalance",
    "normal": "Normal",
    "bpfi": "BPFI",
    "bpfo": "BPFO",
    "misalign": "Misalign",
}

_FILENAME_RE = re.compile(
    r"^(?P<load>\d+)Nm_(?P<condition>[A-Za-z]+)(?:_(?P<severity>[\w]+))?$"
)


class LabelParseError(ValueError):
    pass


def parse_filename(path: Path) -> dict:
    """回傳單一檔案的標籤 metadata。

    範例：
        parse_filename(Path("4Nm_BPFI_10.mat"))
        -> {
            "load_nm": 4,
            "condition": "BPFI",
            "severity": "10",
            "severity_level": 2,
            "file_id": "4Nm_BPFI_10",
        }
    """
    stem = path.stem
    m = _FILENAME_RE.match(stem)
    if not m:
        raise LabelParseError(f"無法解析檔名: {path.name}")

    load_nm = int(m.group("load"))
    raw_condition = m.group("condition")
    condition = _CONDITION_ALIASES.get(raw_condition.lower())
    if condition is None:
        raise LabelParseError(f"未知的故障類型 '{raw_condition}' in {path.name}")

    severity = m.group("severity")  # Normal 時會是 None

    severity_map = SEVERITY_ORDER.get(condition, {})
    if condition == "Normal":
        severity_level = 0
    else:
        if severity not in severity_map:
            raise LabelParseError(
                f"未知的嚴重度代碼 '{severity}' for condition={condition} in {path.name}"
            )
        severity_level = severity_map[severity]

    return {
        "load_nm": load_nm,
        "condition": condition,
        "severity": severity,
        "severity_level": severity_level,
        "is_fault": condition != "Normal",
        "file_id": stem,
    }


def validate_condition(condition: str) -> None:
    if condition not in FAULT_CLASSES:
        raise LabelParseError(f"'{condition}' 不在定義好的類別 {FAULT_CLASSES} 之中")


if __name__ == "__main__":
    # 簡單自我測試：涵蓋你資料夾裡實際出現過的檔名型態，包含那個拼字錯誤的 case
    samples = [
        "0Nm_Normal.mat",
        "0Nm_BPFI_03.mat",
        "4Nm_BPFO_30.tdms",
        "0Nm_Misalign_05.mat",
        "4Nm_Unbalance_3318mg.mat",
        "2Nm_Unbalalnce_0583mg.mat",  # 拼字錯誤版本
    ]
    for s in samples:
        print(s, "->", parse_filename(Path(s)))
