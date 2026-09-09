"""
信心度：從「模型有多確定」改成「有多少條獨立證據支持這個結論」。

為什麼不用模型機率
------------------
原本畫面上的「可信度」是逐窗多數決的一致率。它在這個資料集上永遠是 100%，
看起來像寫死的。換成 top-1 機率也沒用 —— 實測 10 台機台的 top-1 機率中位數
全部落在 0.9996~0.9998，故障的跟正常的一樣高。

更關鍵的是：在跨負載（模型沒看過該負載）的實驗裡，模型機率**分不出對錯**。

    4Nm_Misalign_01       真實 Misalign   判 Misalign   ✓   一致率 76%   top1 0.55
    4Nm_Unbalance_0583mg  真實 Unbalance  判 Misalign   ✗   一致率 96%   top1 0.84

唯一判錯的那支，一致率比判對的那支還高。任何從模型機率算出來的信心度，
都會在那支上「很有信心地講錯話」—— 那比顯示 100% 更難跟評審交代。

而且檔案層級只有 45 支錄音、每個故障條件僅一支，用這種樣本數去做機率校準
（Platt / isotonic）算出「校準後 87% 信心」，是比 100% 更大的過度宣稱。

所以這裡的設計
--------------
信心度 = 四條**可被人工檢核**的條件，輸出分級（高／中／低），不輸出百分比。
每一條都攤在畫面上，讓看的人自己判斷要不要相信，而不是要他們相信一個數字。

    1. 物理證據支持    包絡譜對應頻率的譜線是否明顯偏離同負載正常基準（不經模型）
    2. 跨感測一致      振動與電流是否指向同一結論
    3. 逐窗穩定        模型在不同時間窗是否前後一致
    4. 非已知混淆對    結論是否落在已知會互相混淆的類別（不對心 ↔ 不平衡）

條件 1 是唯一不經模型的一條，所以它有否決權：物理證據不支持，等級不可能是「高」。
這也接得上專案本來的敘事 ——「就算不相信 AI，也可以自己拿頻譜圖核對」。
"""

from __future__ import annotations

# 證據要幾倍於正常基準，才算「明確支持」。
# evidence.MIN_RATIO 是 1.5（列不列出來的門檻），這裡要求更嚴：
# 1.5 倍在單次量測裡還可能是雜訊，3 倍才適合拿來當「高信心」的依據。
EVIDENCE_STRONG = 3.0

# 判「正常」時反過來：任何一項超過這個倍數，就不該給「正常＋高信心」。
EVIDENCE_QUIET = 2.0

AGREEMENT_STABLE = 0.90    # 逐窗一致率高於此算穩定
AGREEMENT_SHAKY = 0.60     # 低於此視為判斷搖擺，直接壓到「低」

LEVEL_COLORS = {"高": "OK", "中": "WARN", "低": "BAD"}

LEVEL_NOTE = {
    "高": "四項條件都成立，其中物理證據可以自行用頻譜核對",
    "中": "部分條件成立，結論方向可信但建議一併複檢",
    "低": "支持條件不足，請轉人工複檢後再處置",
}

METHOD_NOTE = (
    "信心度不是模型輸出的機率，而是四條可檢核條件的通過情形。"
    "本資料集共 45 支錄音、每個故障條件僅一支，樣本數不足以校準出可靠的百分比，"
    "因此只給分級，不給百分比。"
)


def _check_physical(evidence: list[dict], fault_type: str) -> dict:
    """條件 1：物理證據支持結論。唯一不經模型的一條，因此有否決權。"""
    if not evidence:
        if fault_type == "Normal":
            return dict(key="physical", label="物理證據支持", passed=True,
                        detail="各項頻譜特徵皆在同負載正常基準範圍內，沒有需要解釋的偏離")
        return dict(key="physical", label="物理證據支持", passed=False,
                    detail=f"頻譜上找不到支持「{fault_type}」的譜線偏離，結論只有模型輸出撐著")

    if fault_type == "Normal":
        worst = max(evidence, key=lambda e: e["倍數"])
        ok = worst["倍數"] < EVIDENCE_QUIET
        return dict(
            key="physical", label="物理證據支持", passed=ok,
            detail=(f"最大偏離僅 {worst['倍數']}×（{worst['名稱']}），"
                    f"在正常基準的容許範圍內"
                    if ok else
                    f"判為正常，但 {worst['名稱']} 已達正常基準的 {worst['倍數']}×，"
                    f"兩者不一致"),
        )

    support = [e for e in evidence
               if e.get("支持類別") == fault_type and e["倍數"] >= EVIDENCE_STRONG]
    if support:
        best = max(support, key=lambda e: e["倍數"])
        return dict(key="physical", label="物理證據支持", passed=True,
                    detail=(f"{best['名稱']} @ {best['位置']} 為同負載正常基準的 "
                            f"{best['倍數']}×，與「{fault_type}」的已知機制相符"))

    near = [e for e in evidence if e.get("支持類別") == fault_type]
    if near:
        best = max(near, key=lambda e: e["倍數"])
        return dict(key="physical", label="物理證據支持", passed=False,
                    detail=(f"{best['名稱']} 只有 {best['倍數']}×，"
                            f"未達判定門檻 {EVIDENCE_STRONG:.0f}×"))
    other = max(evidence, key=lambda e: e["倍數"])
    return dict(key="physical", label="物理證據支持", passed=False,
                detail=(f"偏離最大的是 {other['名稱']}（{other['倍數']}×），"
                        f"但它指向的是「{other.get('支持類別') or '其他'}」而不是本次結論"))


def _check_cross_sensor(cross_sensor) -> dict:
    """條件 2：跨感測一致。

    cross_sensor 期望 (振動異常機率, 電流異常機率)；沒有獨立電流模型時傳 None，
    這一條會標成「不適用」並退出計分，而不是當成失敗 ——
    變轉速模式用的是單一融合模型，本來就拆不出兩個獨立意見，
    把它算成扣分等於因為架構不同而懲罰它。
    """
    if cross_sensor is None:
        return dict(key="cross_sensor", label="跨感測一致", passed=None,
                    detail="此模式為單一融合模型，拆不出各感測器的獨立意見，不列入計分")
    vib, cur = cross_sensor
    both_abnormal = vib >= 0.5 and cur >= 0.5
    both_normal = vib < 0.5 and cur < 0.5
    ok = both_abnormal or both_normal
    return dict(
        key="cross_sensor", label="跨感測一致", passed=ok,
        detail=(f"振動異常機率 {vib:.2f}、電流異常機率 {cur:.2f}，兩者指向同一結論"
                if ok else
                f"振動 {vib:.2f} 與電流 {cur:.2f} 不一致，只有單一來源支持"),
    )


def _check_stability(agreement: float, n_windows: int) -> dict:
    """條件 3：逐窗穩定。

    ⚠️ 這一條刻意只當「四分之一」。相鄰 1 秒時間窗高度相關，
    不是 N 次獨立投票，而是接近同一次投票重複 N 次 ——
    一致率高本來就是常態，它能證明的事情比直覺以為的少很多。
    """
    ok = agreement >= AGREEMENT_STABLE and n_windows >= 10
    if n_windows < 10:
        detail = f"只有 {n_windows} 個時間窗，樣本太短，穩定性無法判定"
    elif ok:
        detail = f"{n_windows} 個時間窗中有 {agreement:.0%} 給出相同結論"
    else:
        detail = f"一致率僅 {agreement:.0%}，模型在不同時間窗之間搖擺"
    return dict(key="stability", label="逐窗穩定", passed=ok, detail=detail)


def _check_confusion(known_confusion: str) -> dict:
    """條件 4：非已知混淆對。"""
    if known_confusion:
        return dict(key="no_confusion", label="非已知混淆對", passed=False,
                    detail=f"此類別存在已知辨識限制：{known_confusion}")
    return dict(key="no_confusion", label="非已知混淆對", passed=True,
                detail="此類別在跨負載測試中沒有與其他類別互相混淆的紀錄")


def assess(*, evidence: list[dict] | None, fault_type: str,
           agreement: float, n_windows: int,
           cross_sensor: tuple[float, float] | None,
           known_confusion: str = "", overrode: str | None = None) -> dict:
    """回傳 {level, checks, passed, applicable, summary, note}。

    overrode: 物理閘門推翻分類器結論時，傳入分類器原本的答案。
        兩層意見不一致就不該給「高」——實測有一支最輕微的不平衡
        （4Nm_Unbalance_0583mg）物理偏離只有 1.0 倍，被閘門判成正常。
        那是漏報，而「高信心的漏報」是預測維護裡最貴的一種錯誤。
        這種情況一律壓到「中」，並在畫面上標明兩層不一致。
    """
    checks = [
        _check_physical(evidence or [], fault_type),
        _check_cross_sensor(cross_sensor),
        _check_stability(agreement, n_windows),
        _check_confusion(known_confusion),
    ]
    if overrode:
        checks[0] = dict(
            key="physical", label="物理證據支持", passed=None,
            detail=(f"分類器判為「{overrode}」，但頻譜上各項特徵都在同負載正常基準"
                    f"範圍內（最大偏離未達 1.5 倍）。兩層意見不一致，"
                    f"依基準判為正常，但不給高信心 —— 最輕微的故障本來就"
                    f"可能低於物理偵測門檻，建議延長取樣後複檢。"),
        )

    applicable = [c for c in checks if c["passed"] is not None]
    passed = [c for c in applicable if c["passed"]]
    physical_ok = checks[0]["passed"]

    # 分級規則：
    #   高 —— 物理證據成立，且其餘適用條件全數成立
    #   低 —— 物理證據不成立，或模型判斷搖擺（一致率 < 60%）
    #   中 —— 其餘
    if agreement < AGREEMENT_SHAKY:
        level = "低"
    elif overrode:
        # 兩層不一致：不可能是「高」，也不該是「低」（物理層有明確依據）
        level = "中"
    elif physical_ok and len(passed) == len(applicable):
        level = "高"
    elif not physical_ok:
        level = "低" if len(passed) <= 1 else "中"
    else:
        level = "中"

    return {
        "level": level,
        "checks": checks,
        "passed": len(passed),
        "applicable": len(applicable),
        "summary": f"{len(passed)}/{len(applicable)} 項條件成立",
        "note": LEVEL_NOTE[level],
        "method_note": METHOD_NOTE,
    }


def from_diagnosis(diag: dict, evidence: list[dict] | None = None) -> dict:
    """從 dashboard_data.diagnose() / build_llm_diag() 的輸出直接算信心度。"""
    detail = diag.get("modality_detail") or {}
    primary = diag.get("primary_key")
    main = detail.get(primary, {}) if primary else {}

    cross = None
    if "vibration" in detail and "current" in detail:
        cross = (float(detail["vibration"]["anomaly_prob"]),
                 float(detail["current"]["anomaly_prob"]))

    return assess(
        evidence=evidence if evidence is not None else diag.get("physical_evidence"),
        fault_type=diag.get("probable_fault") or diag.get("fault_type") or "Normal",
        agreement=float(main.get("confidence", diag.get("confidence", 0.0))),
        n_windows=int(main.get("n_windows", diag.get("n_windows", 0))),
        cross_sensor=cross,
        known_confusion=diag.get("known_confusion", ""),
    )


def format_for_llm(verdict: dict) -> str:
    """注入 LLM system prompt 用的文字區塊。"""
    lines = [f"  等級：{verdict['level']}（{verdict['summary']}）"]
    for c in verdict["checks"]:
        mark = "—" if c["passed"] is None else ("通過" if c["passed"] else "未通過")
        lines.append(f"  [{mark}] {c['label']}：{c['detail']}")
    lines.append(f"  {METHOD_NOTE}")
    return "\n".join(lines)
