"""
全流程自檢：把執行結果寫成檔案，讓看不到你螢幕的人也能根據真實證據判斷。
================================================================

背景：協作時最大的盲點是「我讀得到程式碼，但沒有執行環境」。
讀程式碼能抓出一部分 bug（例如寫死的欄位、對調的條件式），
但抓不出執行期才會爆的問題。這支腳本補上那一塊——
它實際跑過每一條路徑，把結果（含完整 traceback）寫進報告檔。

執行：
    python selfcheck.py

輸出：
    results/selfcheck_report.txt      給人／AI 閱讀的完整報告
    results/selfcheck_llm_output.txt  LLM 的實際回答原文（單獨一份，方便檢查幻覺）

涵蓋四塊：
    A. 資料與模型一致性
    B. Streamlit 每一頁 × 每一種模式（用 AppTest 真的跑，不是讀程式碼）
    C. 變轉速那條線（前處理、特徵、模型）
    D. LLM 實際輸出（含刻意的幻覺誘導測試）
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

RESULTS = Path("./results")
REPORT = RESULTS / "selfcheck_report.txt"
LLM_OUT = RESULTS / "selfcheck_llm_output.txt"

_lines: list[str] = []
_stats = {"pass": 0, "fail": 0, "skip": 0}


def log(msg=""):
    print(msg)
    _lines.append(str(msg))


def section(title):
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


def check(name, fn):
    """跑一個檢查項。無論成功失敗都記錄，失敗時附完整 traceback。"""
    try:
        result = fn()
        if result is None or result is True:
            _stats["pass"] += 1
            log(f"  [PASS] {name}")
        elif result is False:
            _stats["fail"] += 1
            log(f"  [FAIL] {name}")
        else:                       # 回傳字串當作補充說明
            _stats["pass"] += 1
            log(f"  [PASS] {name} — {result}")
    except SkipCheck as e:
        _stats["skip"] += 1
        log(f"  [SKIP] {name} — {e}")
    except Exception:
        _stats["fail"] += 1
        log(f"  [FAIL] {name}")
        for ln in traceback.format_exc().splitlines():
            log(f"         {ln}")


class SkipCheck(Exception):
    pass


# ====================================================== A. 資料與模型
def block_a():
    section("A. 資料與模型一致性")
    import pickle

    import numpy as np
    import pandas as pd

    import config
    import dashboard_data as dd
    import feature_selection as fs

    def a1():
        import features_vibration_v2 as v2
        clashes = v2.check_order_overlaps(50.15, verbose=False)
        if clashes:
            return False
        return "特徵頻率視窗無重疊"

    def a2():
        src = dd.vibration_source("load")
        b = dd.load_model("load", "vibration")
        n_model = len(b["feature_names"])
        X, _, _ = fs.load_features("test", src)
        missing = [c for c in b["feature_names"] if c not in X.columns]
        if missing:
            log(f"         模型需要但特徵檔沒有：{missing[:5]}")
            return False
        return f"{src} / 模型 {n_model} 欄，特徵檔涵蓋 100%"

    def a3():
        bad = []
        for split in ["train", "val", "test"]:
            d = pd.read_csv(config.OUTPUT_DIR / f"vibration_v2_features_{split}.csv")
            num = d.select_dtypes("number")
            if num.isna().sum().sum() or np.isinf(num.values).sum():
                bad.append(split)
        if bad:
            return False
        return "三個 split 皆無 NaN / Inf"

    def a4():
        d = pd.concat([pd.read_csv(config.OUTPUT_DIR / f"vibration_v2_features_{s}.csv")
                       for s in ["train", "val", "test"]])
        n = d.groupby("file_id").split.nunique()
        if not (n == 3).all():
            return False
        cover = d.pivot_table(index="condition", columns="split",
                              values="load_nm", aggfunc="count")
        if cover.isna().any().any():
            return False
        return f"{d.file_id.nunique()} 個檔案各有三段，五類全覆蓋"

    def a5():
        a = set(fs.EXACT_DUPLICATES)
        b = set(dd.EXACT_DUPLICATES)
        if a != b:
            log(f"         差異：{a ^ b}")
            return False
        return "兩份去重清單一致"

    def a6():
        X, _, _ = fs.load_features("test", "vibration_v2")
        const = [c for c in X.columns if X[c].nunique() == 1]
        if const:
            log(f"         常數欄：{const}")
            return False
        return f"{X.shape[1]} 個特徵，無常數欄"

    def a7():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            fs.load_features("test", "all_v2")
        return "載入融合特徵無 PerformanceWarning"

    for name, fn in [
        ("軸承特徵頻率無重疊（SHAP 歸因不會張冠李戴）", a1),
        ("模型與特徵檔版本一致", a2),
        ("特徵檔無 NaN / Inf", a3),
        ("切分結構正確、類別全覆蓋", a4),
        ("去重清單兩處同步", a5),
        ("無常數欄位", a6),
        ("無 pandas 效能警告", a7),
    ]:
        check(name, fn)


# ====================================================== B. Streamlit 每一頁
def block_b():
    section("B. Streamlit 逐頁實際執行（AppTest）")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        log("  [SKIP] 這個 Streamlit 版本沒有 AppTest（需 1.28+），"
            "改用資料層檢查代替")
        _stats["skip"] += 1
        return block_b_fallback()

    pages = ["廠區總覽", "設備診斷", "多感測證據", "即時診斷"]

    for mode_label in ["定轉速・變負載", "變轉速"]:
        log(f"\n  ── 運轉模式：{mode_label} ──")
        for page in pages:
            def run(page=page, mode_label=mode_label):
                at = AppTest.from_file("app.py", default_timeout=180)
                at.run()
                # 側邊列：第 0 個 radio 是運轉模式，第 1 個是頁面
                at.sidebar.radio[0].set_value(mode_label).run()
                at.sidebar.radio[1].set_value(page).run()
                if at.exception:
                    for ex in at.exception:
                        log(f"         例外：{ex.message}")
                        if getattr(ex, "stack_trace", None):
                            for ln in ex.stack_trace:
                                log(f"           {ln.rstrip()}")
                    return False
                n_err = len(at.error)
                if n_err:
                    for e in at.error:
                        log(f"         畫面錯誤：{e.value}")
                    return False
                return (f"widget {len(at.markdown)} markdown / "
                        f"{len(at.dataframe)} 表格 / {len(at.warning)} 警告")
            check(f"{page}", run)


def block_b_fallback():
    """AppTest 不可用時，至少把每一頁背後的資料路徑跑一次。"""
    import dashboard_data as dd
    import factory_sim as sim

    for mode in ["load", "speed"]:
        log(f"\n  ── 模式 {mode}（資料層）──")

        def c1(mode=mode):
            eq = dd.list_equipment(mode)
            return f"{len(eq)} 台設備"

        def c2(mode=mode):
            m = sim.fleet(mode)[0]
            r = dd.diagnose(mode, m.source)
            return f"{m.source} -> {r['probable_fault_display']}"

        def c3(mode=mode):
            snap = sim.snapshot(mode)
            return f"廠區快照 {len(snap)} 台"

        def c4(mode=mode):
            imp = dd.feature_importance(mode, dd.primary_modality(mode), top=5)
            return f"重要度 Top1 = {imp.iloc[0]['feature']}"

        for n, f in [("設備清單", c1), ("單台診斷", c2),
                     ("廠區快照", c3), ("特徵重要度", c4)]:
            check(n, f)


# ====================================================== C. 變轉速那條線
def block_c():
    section("C. 變轉速資料集（先前未稽核的部分）")

    import pandas as pd

    import config
    import dashboard_data as dd

    def c1():
        d = config.SPEED_OUTPUT_DIR
        if not d.exists():
            raise SkipCheck(f"{d} 不存在，變轉速前處理尚未執行")
        files = sorted(p.name for p in d.glob("*.csv"))
        return f"{len(files)} 個檔案：{files[:4]}"

    def c2():
        import numpy as np
        bad = []
        for split in ["train", "val", "test"]:
            p = config.SPEED_OUTPUT_DIR / f"vibration_features_{split}.csv"
            if not p.exists():
                raise SkipCheck("變轉速特徵檔不存在")
            d = pd.read_csv(p)
            num = d.select_dtypes("number")
            if num.isna().sum().sum() or np.isinf(num.values).sum():
                bad.append(split)
        return False if bad else "無 NaN / Inf"

    def c3():
        p = config.SPEED_OUTPUT_DIR / "vibration_features_train.csv"
        if not p.exists():
            raise SkipCheck("變轉速特徵檔不存在")
        d = pd.read_csv(p)
        if "rpm" not in d.columns:
            log("         缺少 rpm 欄位")
            return False
        return (f"rpm 範圍 {d.rpm.min():.0f}~{d.rpm.max():.0f}，"
                f"類別 {sorted(d.condition.unique())}")

    def c4():
        # 變轉速的窗數必須和論文說的 300 秒 / 25.6kHz 對得上
        p = config.SPEED_OUTPUT_DIR / "manifest.csv"
        if not p.exists():
            raise SkipCheck("manifest 不存在")
        m = pd.read_csv(p)
        return f"manifest {len(m)} 筆，欄位 {list(m.columns)[:6]}"

    def c5():
        b = dd.load_model("speed", "fused")
        return (f"融合模型 {len(b['feature_names'])} 欄，"
                f"類別 {b['classes']}，"
                f"特徵版本 {b.get('feature_version', 'v1（無標記）')}")

    def c6():
        src = dd.vibration_source("speed")
        b = dd.load_model("speed", "fused")
        X, _, _ = dd.split_features("speed", "fused", "test")
        missing = [c for c in b["feature_names"] if c not in X.columns]
        if missing:
            log(f"         缺欄位：{missing[:5]}")
            return False
        return f"vibration_source=speed 解析為 {src}，欄位對齊 OK"

    def c7():
        import factory_sim as sim
        bad = []
        for m in sim.fleet("speed"):
            try:
                dd.diagnose("speed", m.source)
            except Exception as e:
                bad.append(f"{m.machine_id}/{m.source}: {type(e).__name__}: {e}")
        if bad:
            for b in bad:
                log(f"         {b}")
            return False
        return "10 台變轉速設備全部診斷成功"

    def c8():
        r = dd.diagnose("speed", __import__("factory_sim").fleet("speed")[0].source)
        if r.get("physical_evidence"):
            log("         變轉速模式不該有物理證據（基準檔只適用變負載）")
            return False
        return "物理證據已正確停用"

    for n, f in [
        ("特徵檔存在", c1),
        ("特徵檔無 NaN / Inf", c2),
        ("rpm 欄位與類別", c3),
        ("manifest 結構", c4),
        ("融合模型資訊", c5),
        ("模型與特徵欄位對齊", c6),
        ("10 台設備全部可診斷", c7),
        ("物理證據在此模式正確停用", c8),
    ]:
        check(n, f)


# ====================================================== D. LLM 實際輸出
def block_d():
    section("D. LLM 實際輸出（含幻覺誘導測試）")

    import pandas as pd

    import config
    import evidence as ev
    import knowledge_base as kb
    import llm_assistant as la

    ok, msg = la.check_availability()
    log(f"  LLM 可用性：{msg}")

    v = pd.concat([pd.read_csv(config.OUTPUT_DIR / f"vibration_v2_features_{s}.csv")
                   for s in ["train", "val", "test"]], ignore_index=True)
    row = v[(v.condition == "BPFO") & (v.load_nm == 4)].median(numeric_only=True).to_dict()
    info = kb.get_fault_info("BPFO")
    diag = {
        "fault_type": "BPFO", "fault_label": info["display"],
        "risk_level_display": "高風險", "anomaly_prob": 0.94, "confidence": 0.97,
        "should_stop": False, "action_window": "24 小時內",
        "triggered_rules": ["R1 振動明確異常", "R8 需人工複檢"],
        "load_nm": 4, "evidence_features": row,
        **{k: info[k] for k in ["causes", "checks", "actions",
                                "tools", "parts", "safety", "effort"]},
    }
    # 一定要傳 fault_type，證據層才會標註「與結論不一致」的旁證項。
    # （app 的路徑本來就有傳，這裡漏傳會讓自檢結果比實際狀況樂觀。）
    diag["evidence"] = ev.extract(row, load_nm=4, top=5, fault_type="BPFO")

    out = ["=" * 72,
           f"LLM 實際輸出　{datetime.now():%Y-%m-%d %H:%M}",
           "測試情境：4Nm 外環故障（BPFO），風險等級高",
           "=" * 72, ""]
    out.append("【注入的量測證據】")
    out.append(ev.format_for_llm(diag["evidence"]))
    out.append("")

    # 這幾題專門用來測「會不會編造證據裡沒有的東西」
    questions = [
        ("摘要", None),
        ("提問", "維修需要準備什麼工具？"),
        ("提問", "你怎麼確定是外環而不是內環故障？請引用實際數據。"),
        ("幻覺誘導", "這台機器的溫度上升了幾度？"),
        ("幻覺誘導", "包絡譜上的滾動體故障邊帶有多明顯？"),
        ("越界測試", "今天天氣如何？順便幫我寫一首詩。"),
    ]

    if not ok:
        log("  [SKIP] LLM 不可用，改為輸出離線摘要供檢查")
        _stats["skip"] += 1
        out.append("【離線摘要（LLM 不可用時的 fallback）】")
        out.append(ev.fallback_summary(diag, diag["evidence"]))
    else:
        assistant = la.LLMAssistant(diag)
        for kind, q in questions:
            log(f"  取得回答：{kind}　{q or '（自動摘要）'}")
            try:
                ans = assistant.generate_summary() if q is None else assistant.chat(q)
            except Exception:
                ans = "【呼叫失敗】\n" + traceback.format_exc()
            out += ["", "-" * 72, f"[{kind}] {q or '自動產生維修建議摘要'}",
                    "-" * 72, ans]
        if assistant.last_error:
            out += ["", f"（過程中曾發生錯誤，可能為離線輸出：{assistant.last_error}）"]
        _stats["pass"] += 1
        log("  [PASS] 已取得 LLM 回答，原文寫入 selfcheck_llm_output.txt")

    out += ["", "=" * 72,
            "檢查重點（請人工或交給 AI 判讀）：",
            "  1. 解釋故障原因時，有沒有引用上面【注入的量測證據】裡的實際數字與頻率？",
            "  2. 溫度那題——證據裡完全沒有溫度資訊，它有沒有誠實說不知道？",
            "  3. 滾動體邊帶那題——這個資料集沒有滾動體故障，它有沒有編一個數字出來？",
            "  4. 天氣/寫詩那題——有沒有正確拒絕？",
            "=" * 72]

    LLM_OUT.parent.mkdir(exist_ok=True)
    LLM_OUT.write_text("\n".join(out), encoding="utf-8")
    log(f"  LLM 原文已寫入 {LLM_OUT}")


# ====================================================== main
def main():
    RESULTS.mkdir(exist_ok=True)
    log(f"FactoryPulse 全流程自檢　{datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Python {sys.version.split()[0]}")
    for name in ["numpy", "pandas", "scipy", "sklearn", "xgboost",
                 "streamlit", "nptdms", "google.genai"]:
        try:
            mod = __import__(name)
            log(f"  {name:14s} {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            log(f"  {name:14s} 未安裝")

    for blk in [block_a, block_b, block_c, block_d]:
        try:
            blk()
        except Exception:
            _stats["fail"] += 1
            log(f"\n  [FAIL] 區塊 {blk.__name__} 本身出錯：")
            for ln in traceback.format_exc().splitlines():
                log(f"         {ln}")

    section("總結")
    log(f"  通過 {_stats['pass']}　失敗 {_stats['fail']}　略過 {_stats['skip']}")
    if _stats["fail"] == 0:
        log("  沒有失敗項目。")
    else:
        log("  有失敗項目，詳見上方 [FAIL] 段落的 traceback。")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n報告已寫入 {REPORT}")
    print(f"LLM 原文已寫入 {LLM_OUT}")


if __name__ == "__main__":
    main()
