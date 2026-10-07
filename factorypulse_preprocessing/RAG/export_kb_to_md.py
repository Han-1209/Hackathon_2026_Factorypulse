"""
把 knowledge_base.py 匯出成 RAG 用的 Markdown 文件。

為什麼用程式匯出、不手動另寫一份 MD：
    knowledge_base.py 仍是唯一真相來源（規則引擎、LLM、儀表板都讀它）。
    MD 只是它的「RAG 版本」，每次改完知識庫重跑一次本程式即可，
    避免兩份內容各改各的、越差越遠。

用法：
    python export_kb_to_md.py            # 輸出到 kb_docs/

輸出結構（一個故障一個檔，方便依檔案或 metadata 過濾）：
    kb_docs/README.md           說明與切塊（chunking）建議
    kb_docs/fault_<代碼>.md      六種馬達狀況
    kb_docs/risk_levels.md      風險等級與處置時限

切塊設計：
    每個「## 」標題是一個 chunk，標題本身帶故障名稱
    （例如「## 軸承內圈故障｜需要工具」），且每段開頭重述適用狀況。
    這樣 chunk 單獨被檢索出來時，LLM 仍知道它屬於哪一種故障，
    不會把 A 故障的工具清單拿去回答 B 故障。
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# knowledge_base.py 在上一層（主程式資料夾），加進搜尋路徑才 import 得到
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_base import FAULT_INFO, RISK_ACTIONS

OUT_DIR = Path(__file__).parent / "kb_docs"

# 每類故障出現在哪個資料集（與 knowledge_base.py 檔頭說明一致）
DATASETS = {
    "Normal": ["變負載", "變轉速"],
    "BPFI": ["變負載", "變轉速"],
    "BPFO": ["變負載", "變轉速"],
    "Ball": ["變轉速"],
    "Misalign": ["變負載"],
    "Unbalance": ["變負載"],
}
URGENCY_ZH = {"none": "無", "medium": "中", "high": "高"}

# (欄位, 小節名稱)；順序即文件中的順序
SECTIONS = [
    ("description", "說明"),
    ("signature", "判斷依據（本系統實測訊號特徵）"),
    ("causes", "可能原因"),
    ("checks", "建議檢查項目"),
    ("actions", "待修事項"),
    ("tools", "需要工具"),
    ("parts", "備品耗材"),
    ("safety", "安全注意事項"),
    ("effort", "工時與人力"),
    ("known_confusion", "已知易混淆情況"),
]


def _body(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {v}" for v in value)
    return str(value)


def fault_md(code: str, info: dict, today: str) -> str:
    name = info["display"]
    urgency = URGENCY_ZH.get(info.get("urgency", ""), info.get("urgency", ""))
    datasets = DATASETS.get(code, [])
    lines = [
        "---",
        "doc_type: fault",
        f"fault_code: {code}",
        f"display: {name}",
        f"urgency: {info.get('urgency', '')}",
        f"datasets: [{', '.join(datasets)}]",
        "source: knowledge_base.py",
        f"exported: {today}",
        "---",
        "",
        f"# {name}（{code}）",
        "",
        f"> 緊急程度：{urgency}　｜　適用資料集：{'、'.join(datasets)}",
        "> 是否需要「立即」停機由規則引擎依風險等級判定，本文件不決定。",
        "",
    ]
    for key, title in SECTIONS:
        value = info.get(key)
        if not value:          # 空清單 / 空字串 / 欄位不存在 → 不輸出空章節
            continue
        lines += [
            f"## {name}｜{title}",
            "",
            f"適用狀況：{name}（{code}）",
            "",
            _body(value),
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def risk_md(today: str) -> str:
    lines = [
        "---",
        "doc_type: risk_levels",
        "source: knowledge_base.py",
        f"exported: {today}",
        "---",
        "",
        "# 風險等級與處置時限",
        "",
        "系統依多項感測證據綜合判定風險等級，再依等級決定處置時限與是否需要立即停機。",
        "",
    ]
    for code, r in RISK_ACTIONS.items():
        lines += [
            f"## 風險等級｜{r['display']}（{code}）",
            "",
            f"- 建議處置時限：{r['window']}",
            f"- 說明：{r['note']}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


README = """# FactoryPulse 維修知識庫（RAG 版）

> ⚠️ 本資料夾由 `export_kb_to_md.py` 自動產生，請勿直接修改。
> 要改內容請改 `knowledge_base.py`，再執行 `python export_kb_to_md.py`。

## 檔案

| 檔案 | 內容 |
|---|---|
{rows}
| `risk_levels.md` | 風險等級與處置時限 |

## 切塊（chunking）建議

- 以 `## ` 標題切塊，一個小節一個 chunk（每段約 50~300 字，不需要再用固定字數切）。
- 每個 chunk 的標題與第一行都帶有故障名稱，單獨被檢索時仍可辨識所屬故障。
- 檔頭 YAML 的 `fault_code`、`urgency`、`datasets` 建議存成 chunk 的 metadata。

## 檢索建議

- **已知診斷結果時**：用 `fault_code` 做 metadata 過濾，只在該故障的 chunk 裡做語意搜尋，
  避免把其他故障的工具或步驟混進回答。
- **使用者自由提問、尚無診斷時**：才對全部 chunk 做語意搜尋，並在回答中標示來源檔案與小節。
- 檢索不到相關內容時，應回答「知識庫中沒有這項資訊」，不得自行補充。

匯出日期：{today}
"""


def main() -> None:
    today = dt.date.today().isoformat()
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for code, info in FAULT_INFO.items():
        fname = f"fault_{code}.md"
        (OUT_DIR / fname).write_text(fault_md(code, info, today), encoding="utf-8")
        rows.append(f"| `{fname}` | {info['display']}（{code}） |")
    (OUT_DIR / "risk_levels.md").write_text(risk_md(today), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        README.format(rows="\n".join(rows), today=today), encoding="utf-8")
    print(f"已輸出 {len(FAULT_INFO) + 2} 個檔案到 {OUT_DIR}")


if __name__ == "__main__":
    main()
