"""
RAG 檢索測試：用一組「已知正確答案」的問題，檢查檢索找得準不準，並建議相似度門檻。

用法（需要網路與 Gemini API Key）：
    python rag_eval.py

為什麼需要這支程式：
    Gemini embedding 的相似度「基準線」很高 —— 知識庫裡任兩段文字的相似度就有 0.68~0.97，
    所以無關的問題也可能拿到 0.7 分。門檻要用真實問題量出來，不能憑感覺定。
    之後知識庫改版、換 embedding 模型，都重跑一次本程式確認沒有退步。
"""
from __future__ import annotations

import rag_store as r

# (問題, 本次研判故障, 應該排第一的段落標題；None = 知識庫沒有答案，不應該回傳任何段落)
CASES = [
    ("要準備什麼工具", "BPFO", "軸承外圈故障｜需要工具"),
    ("大概要修多久", "BPFO", "軸承外圈故障｜工時與人力"),
    ("維修時要注意什麼安全事項", "BPFI", "軸承內圈故障｜安全注意事項"),
    ("需要先準備哪些零件", "BPFI", "軸承內圈故障｜備品耗材"),
    ("如果其實是不平衡，要準備什麼工具", "BPFO", "轉子不平衡｜需要工具"),
    ("對心要用什麼儀器", "Misalign", "軸不對心｜需要工具"),
    ("為什麼會不對心", "Misalign", "軸不對心｜可能原因"),
    ("不對心跟不平衡會不會搞混", "Misalign", "軸不對心｜已知易混淆情況"),
    ("滾珠壞掉的話怎麼判斷", "Normal", "滾動體故障｜判斷依據（本系統實測訊號特徵）"),
    ("高風險要多快處理", "BPFO", "風險等級｜高風險（high）"),
    ("危急的時候要立刻停機嗎", "BPFO", "風險等級｜危急（critical）"),
    # 知識庫沒有答案的問題 —— 用來量「不相關問題」最高會拿到幾分
    ("主軸馬達的額定電流是多少", "BPFO", None),
    ("這台機器是哪一年出廠的", "BPFO", None),
    ("潤滑油要用哪個品牌", "BPFO", None),
    ("公司的請假流程是什麼", "BPFO", None),
]


def main():
    ret = r.KnowledgeRetriever()
    print(f"檢索模式：{ret.status}  {ret.status_note}")
    if ret.status != "semantic":
        print("⚠️ 不是語意模式，請先執行 python rag_store.py build，並確認網路與 API Key")
        return
    # 測試時把門檻降到最低，看清楚原始分數
    saved = r.MIN_SCORE_SEMANTIC
    r.MIN_SCORE_SEMANTIC = -1

    # 判定標準：正確段落出現在前 3 名（LLM 實際上會拿到前 3 段，不只第 1 名）
    # 分數一律是「原始相似度」，不含本次故障的排名加分
    correct_scores, unanswerable_scores, n_top1, n_top3, n_ans = [], [], 0, 0, 0
    print(f"\n{'結果':<6} {'分數':<7} 問題  →  檢索到的第一名")
    for q, fault, expect in CASES:
        hits = ret.search(q, prefer_fault=fault, k=3)
        top = hits[0] if hits else None
        if expect is None:
            score = max((h["score"] for h in hits), default=0.0)
            unanswerable_scores.append(score)
            print(f"{'（無解）':<6} {score:<7.3f} {q}  →  {top['title'] if top else '—'}")
            continue
        n_ans += 1
        titles = [h["title"] for h in hits]
        rank = titles.index(expect) + 1 if expect in titles else None
        n_top1 += rank == 1
        n_top3 += rank is not None
        mark = {1: "✅", 2: "🟡第2", 3: "🟡第3"}.get(rank, "❌")
        score = hits[rank - 1]["score"] if rank else 0.0
        if rank:
            correct_scores.append(score)
        print(f"{mark:<6} {score:<7.3f} {q}  →  {top['title'] if top else '—'}")
        if rank != 1:
            print(f"{'':16}正確段落：{expect}")

    r.MIN_SCORE_SEMANTIC = saved
    print(f"\n第 1 名命中：{n_top1}/{n_ans}　　前 3 名命中：{n_top3}/{n_ans}")
    if correct_scores and unanswerable_scores:
        lo, hi = min(correct_scores), max(unanswerable_scores)
        print(f"正確段落的最低分：{lo:.3f}")
        print(f"無解問題的最高分：{hi:.3f}　　（兩者差距 {lo - hi:+.3f}）")
        shown_ok = sum(x >= r.DISPLAY_MIN_SEMANTIC for x in correct_scores)
        shown_bad = sum(x >= r.DISPLAY_MIN_SEMANTIC for x in unanswerable_scores)
        print(f"顯示門檻 {r.DISPLAY_MIN_SEMANTIC}：正確段落會顯示 {shown_ok}/{len(correct_scores)}，"
              f"無解問題誤顯示 {shown_bad}/{len(unanswerable_scores)}（應為 0）")
        if lo > hi:
            print(f"參考：兩組分數的中點 = {(lo + hi) / 2:.3f}（目前交給 LLM 的門檻 {saved}）")
            if lo - hi < 0.05:
                print("⚠️ 差距很小，門檻只能擋掉一部分無關問題，"
                      "主要防線仍是 system prompt 的「知識庫未收錄就說不知道」規則。")
        else:
            print("⚠️ 兩組分數重疊，單靠門檻分不開。保留目前門檻，"
                  "並依賴 system prompt 的「參考資料無關就忽略」規則。")


if __name__ == "__main__":
    main()
