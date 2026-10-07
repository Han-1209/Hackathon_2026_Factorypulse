# FactoryPulse 維修知識庫（RAG 版）

> ⚠️ 本資料夾由 `export_kb_to_md.py` 自動產生，請勿直接修改。
> 要改內容請改 `knowledge_base.py`，再執行 `python export_kb_to_md.py`。

## 檔案

| 檔案 | 內容 |
|---|---|
| `fault_Normal.md` | 正常（Normal） |
| `fault_BPFI.md` | 軸承內圈故障（BPFI） |
| `fault_BPFO.md` | 軸承外圈故障（BPFO） |
| `fault_Ball.md` | 滾動體故障（Ball） |
| `fault_Misalign.md` | 軸不對心（Misalign） |
| `fault_Unbalance.md` | 轉子不平衡（Unbalance） |
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

匯出日期：2026-10-06
