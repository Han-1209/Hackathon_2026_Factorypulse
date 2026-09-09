"""
FactoryPulse 儀表板

啟動：
    pip install -r requirements.txt
    streamlit run app.py

四個頁面：
  廠區總覽    ── 10 台 CNC 的廠房佈局與健康狀態，每分鐘更新
  設備診斷    ── 點進單一機台，看完整診斷與建議行動
  多感測證據  ── 展示 AI 的判斷依據（可解釋性，簡報重點頁）
  即時診斷    ── 上傳原始感測檔（.mat / .tdms / .csv），系統自動轉檔並診斷

側邊列可切換「運轉模式」，對應兩個資料集與各自的模型。
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import llm_assistant as llm_mod

import pandas as pd
import streamlit as st

import charts as ch
import dashboard_data as dd
import factory_sim as sim
import factory_view as fv
import ui_theme as T

st.set_page_config(page_title="FactoryPulse 戰情室", page_icon="⚙️", layout="wide")
T.inject()

# 風險等級 -> 顏色。
# 刻意與 ui_theme.tone_color 的三階刻度對齊（見那裡的說明）：
#   健康         -> 綠（風險 <30）
#   注意 / 警告  -> 黃（風險 30~70）
#   高風險 / 危急 -> 紅（風險 >=70）
# 徽章負責顯示「精確是哪一級」，顏色負責傳達「要不要動作」，兩者不會互相矛盾。
RISK_COLOR = {
    "健康": T.OK,
    "注意": T.WARN, "警告": T.WARN,
    "高風險": T.BAD, "危急": T.CRIT,
}

score_color = T.tone_color
score_bar = T.score_bar


def risk_badge(level: str) -> str:
    return T.tag(level, RISK_COLOR.get(level, T.MUTED), filled=True)


# ---------------------------------------------------------------- 側邊列
st.sidebar.title("⚙️ FactoryPulse")
st.sidebar.caption("旋轉設備健康診斷與維修決策平台")

mode = st.sidebar.radio(
    "運轉模式",
    options=["load", "speed"],
    format_func=lambda m: dd.MODE_LABELS[m],
    help="不同運轉模式使用各自訓練的模型。定轉速模式有溫度資料，可做完整三模態診斷。",
)
spec = dd.MODE_SPEC[mode]

st.sidebar.markdown("---")

try:
    _primary = dd.primary_modality(mode)
    _classes = dd.load_model(mode, _primary)["classes"]
    _n_class = f"{len(_classes)} 種狀態（{'、'.join(dd.fault_display(c) for c in _classes)}）"
    _models_ready = True
except FileNotFoundError:
    _n_class = "尚未訓練"
    _models_ready = False

_sensors = dd.sensor_list(mode)
_arch = ("振動與電流各自獨立的模型，可跨感測交叉驗證"
         if "current" in spec["models"]
         else "振動與電流合併輸入的單一融合模型")

st.sidebar.markdown(
    "**這個模式的能力**\n\n"
    f"- 可辨識：{_n_class}\n"
    f"- 感測來源：{'、'.join(_sensors)}\n"
    f"- 模型架構：{_arch}"
)
if not spec["has_temperature"]:
    st.sidebar.info("變轉速資料集沒有溫度感測，因此不做過熱風險判斷。")
if not _models_ready:
    st.sidebar.error(
        "找不到模型檔。請先執行：\n\n"
        f"`python {'train_baseline.py' if mode == 'load' else 'train_speed.py'}`"
    )

page = st.sidebar.radio("頁面", ["廠區總覽", "設備診斷", "多感測證據", "即時診斷"])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"廠內 {len(sim.fleet(mode))} 台同型 CNC\n\n"
    f"機型：{sim.CNC_MODEL}\n\n"
    f"主軸馬達：{sim.MOTOR_MODEL}"
)
st.sidebar.caption(
    "資料來源：KAIST 公開資料集（CC BY 4.0）。"
    "本系統以公開實驗資料驗證診斷引擎，實際部署時改接現場感測節點。"
)


def require_models():
    if not _models_ready:
        st.error("模型尚未訓練，無法顯示診斷結果。")
        st.stop()


@st.cache_data(show_spinner=False, ttl=70)
def cached_snapshot(mode: str, minute: int):
    return sim.snapshot(mode, minute)


def autorefresh(seconds: int = 60):
    """讓畫面每分鐘自動重跑一次。

    st.fragment(run_every=...) 是 Streamlit 1.37 才有的功能。舊版沒有的話
    就靜靜略過，使用者仍可按「立即重新整理」——與其讓整個 app 因為版本
    問題掛掉，不如降級成手動更新。
    """
    try:
        @st.fragment(run_every=f"{seconds}s")
        def _tick():
            # 這個 fragment 本身不畫東西，只負責觸發重跑；
            # 真正的資料更新靠 cached_snapshot 的 ttl 過期。
            st.empty()
        _tick()
        return True
    except (AttributeError, TypeError):
        return False


@st.cache_data(show_spinner=False)
def cached_diagnose(mode: str, fid: str):
    return dd.diagnose(mode, fid)


@st.cache_data(show_spinner=False)
def load_audit():
    """train_lolo.py 產生的信心度稽核表。沒有就回 None。"""
    from pathlib import Path
    p = Path(__file__).parent / "results" / "lolo_confidence.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------- 廠區總覽
if page == "廠區總覽":
    require_models()
    auto_ok = autorefresh(60)
    minute = sim.current_minute()

    # 重新整理按鈕放在頁面最上方靠右，對齊 Streamlit 右上角 Deploy 的位置。
    # 原本夾在 KPI 卡片與廠房圖之間，視覺上像是「重新整理下面那張圖」，
    # 但它其實會清掉整頁快取，位置與作用範圍不符。
    _sp, _refresh = st.columns([7, 1])
    with _refresh:
        if st.button("🔄 重新整理", use_container_width=True,
                     help="清除快取並重新讀取所有機台的感測資料"):
            st.cache_data.clear()
            st.rerun()

    clock = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        T.command_bar(
            "FACTORYPULSE 戰情室",
            f"{dd.MODE_LABELS[mode]}　·　{sim.CNC_MODEL} × {len(sim.fleet(mode))} 台"
            f"　·　主軸馬達 {sim.MOTOR_MODEL}",
            clock,
            "每分鐘自動更新" if auto_ok else "手動更新模式",
        ),
        unsafe_allow_html=True,
    )

    # 模型來源橫幅。這是整個畫面最需要主動說清楚的一件事：
    # 舊版用的是同錄音切分訓練的模型，那組數字《數字口徑表》明文不可對外。
    if mode == "load":
        if dd.lolo_available():
            st.success(
                "**畫面上每一台的診斷，都來自沒看過那個負載的模型。**　"
                "每台機台使用 leave-one-load-out 模型（0/2/4 Nm 各訓練一個，"
                "各自排除自己那個負載），因此不存在「同一支錄音同時出現在訓練與測試」的問題。"
            )
        else:
            st.error(
                "**目前使用的是同錄音切分訓練的模型**，其數字依《數字口徑表》不可對外引用"
                "（畫面上會出現一片 100% 的可信度）。請先執行 `python train_lolo.py` "
                "產生 leave-one-load-out 模型，重啟後即會自動切換。"
            )

    with st.spinner("讀取各機台感測資料..."):
        snap = cached_snapshot(mode, minute)

    df = pd.DataFrame([{k: v for k, v in s.items() if k != "diagnosis"} for s in snap])

    n_stop = int(df["should_stop"].sum())
    avg = df["health"].mean()

    # ⚠️ KPI 的三個桶子必須用「健康分數」切，不能用風險等級的名稱切。
    # 原本寫成「正常/注意」= 健康+注意、「警告」、「高風險/危急」三桶，
    # 但廠房圖的機台顏色是 T.tone_color(健康分數) 決定的，兩邊切點不同：
    # 健康分數 70 分的機台，風險是 30 -> 等級「注意」被算進第一桶（綠），
    # 廠房圖卻依 tone_color 畫成黃色 —— 於是「綠色 1 台，KPI 說 3 台」。
    # 現在直接用 tone_color 的同一組邊界分桶，數量與顏色保證對得上。
    band_by_color = {b["color"]: b["name"] for b in T.BANDS}
    counts = {b["name"]: 0 for b in T.BANDS}
    for h in df["health"]:
        counts[band_by_color[T.tone_color(h)]] += 1

    kpis = [
        dict(label="機台總數", value=str(len(df)), sub="全部同型 VMC-850", tone=T.ACCENT),
        dict(label="平均健康度", value=f"{avg:.0f}", sub="綜合健康分數",
             tone=T.tone_color(avg)),
    ]
    kpis += [
        dict(label=f"{b['dot']} {b['name']}", value=str(counts[b["name"]]),
             sub=f"{b['range']} 分・{b['sub']}", tone=b["color"])
        for b in T.BANDS
    ]
    kpis.append(dict(label="建議停機", value=str(n_stop),
                     sub="立即介入" if n_stop else "無",
                     tone=T.CRIT if n_stop else T.OK))

    st.markdown(T.kpi_row(kpis), unsafe_allow_html=True)
    st.caption(T.SCALE_NOTE)

    with st.expander("分級標準說明（顏色三階、等級五級，分別是怎麼切的）"):
        st.markdown(
            "**顏色三階**　監控畫面要能一眼分辨「可以放著／要排程／要處理」，"
            "所以廠房圖、KPI 卡片、健康分項的分數條都只用三種顏色，切點完全一致：\n\n"
            + "\n".join(
                f"- {b['dot']} **{b['name']}**　健康分數 {b['range']} 分　—　{b['sub']}"
                for b in T.BANDS
            )
            + "\n\n**等級五級**　精確的等級由徽章顯示，用來決定處理時限。"
              "每一級都落在上面某一個顏色帶裡，不會出現徽章寫「高風險」但分數條是黃色的矛盾：\n\n"
            + "\n".join(
                f"- {T.level_dot(lv)}　健康分數 {rng} 分"
                for lv, rng in T.LEVEL_RANGE.items()
            )
            + "\n\n健康分數 = 100 − 風險分數。風險分數由規則引擎依各感測來源的異常機率"
              "加權後計算，再依交叉驗證規則調整。"
        )

    if not auto_ok:
        st.caption("你的 Streamlit 版本不支援自動更新（需 1.37+），"
                   "請按右上角「重新整理」手動更新")

    # ---- 廠房佈局 ----
    st.markdown(
        fv.floor_svg(
            snap,
            f"{dd.MODE_LABELS[mode]}　·　{sim.CNC_MODEL} × {len(snap)} 台",
            clock,
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "每分鐘依最新一段感測訊號更新。數值來自真實錄音的逐窗變化，不是亂數模擬——"
        "每台機台對應一段實際量測訊號，系統每分鐘取一段不同時間區間重新推論。"
    )

    st.markdown("---")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("各機台健康分數")
        rank = df[["machine_id", "name", "health"]].copy()
        rank["機台"] = rank["machine_id"] + "　" + rank["name"]
        # 長條顏色用與廠房圖同一支 tone_color，同一台機器在兩張圖上同色
        rank["_c"] = rank["health"].map(T.tone_color)
        st.altair_chart(
            ch.hbar(rank, "機台", "health",
                    x_title="綜合健康分數（0–100 分，越高越好）",
                    y_title="機台", height=320, color_col="_c"),
            use_container_width=True,
        )
    with right:
        st.subheader("需要處理的機台")
        urgent = df[df["should_stop"]]
        if len(urgent):
            for _, u in urgent.iterrows():
                st.markdown(f"🔴 **{u['machine_id']}** {u['name']} — {u['fault']}")
        else:
            st.success("目前沒有需要立即停機的機台")

    st.markdown("---")
    st.subheader("完整清單")
    show = df[["machine_id", "name", "station", "health", "risk_level",
               "fault", "confidence", "action_window"]].copy()
    # 風險等級加上顏色圓點，與廠房圖、KPI 卡片用同一組配色，
    # 這樣同一台機台在三個地方看到的顏色一定一樣。
    show["risk_level"] = show["risk_level"].map(T.level_dot)
    # 「可信度 100%」全欄一模一樣，看起來像寫死的。現在這一欄是證據型
    # 信心度等級（confidence.py 的四條件判定），不是模型機率。
    show["confidence"] = [T.conf_dot(m["confidence_level"]) for m in snap]
    show.columns = ["機台編號", "工站", "產線", "健康分數", "風險等級",
                    "研判故障", "信心度", "處理時限"]
    st.dataframe(show.sort_values("健康分數"), hide_index=True,
                 use_container_width=True, height=400)
    st.caption(T.CONF_NOTE)


# ---------------------------------------------------------------- 設備診斷
elif page == "設備診斷":
    require_models()

    machines = sim.fleet(mode)
    labels = {m.machine_id: f"{m.machine_id}　{m.name}" for m in machines}
    mid = st.selectbox("選擇機台", [m.machine_id for m in machines],
                       format_func=lambda k: labels[k])
    machine = sim.machine_by_id(mode, mid)
    r = cached_diagnose(mode, machine.source)

    health = r["health_scores"].get("綜合健康", 100 - r["risk_score"])
    conf = r["modality_detail"][r["primary_key"]]["confidence"]
    verdict = r["confidence_verdict"]
    cond = (f"{r['condition_unit']} {r['condition_value']:.0f}"
            if r["condition_value"] is not None else "—")

    st.markdown(
        T.command_bar(
            f"{mid}　{machine.name}",
            f"{machine.station}　·　{sim.CNC_MODEL}　·　運轉條件 {cond}",
            r["fault_label"],
            f"信心度 {verdict['level']}（{verdict['summary']}）",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        T.kpi_row([
            dict(label="綜合健康度", value=f"{health:.0f}", sub="滿分 100",
                 tone=T.tone_color(health)),
            dict(label="風險等級", value=T.level_dot(r["risk_level_display"]),
                 sub=f"風險分數 {r['risk_score']}",
                 tone=RISK_COLOR.get(r["risk_level_display"], T.ACCENT)),
            dict(label="處理時限", value=r["action_window"], sub="建議介入時間",
                 tone=T.WARN if r["action_window"] != "—" else T.OK),
            dict(label="信心度", value=verdict["level"], sub=verdict["summary"],
                 tone=T.conf_color(verdict["level"])),
            dict(label="運轉決策",
                 value="建議停機" if r["should_stop"] else "可運轉",
                 sub="立即安排" if r["should_stop"] else "維持監測",
                 tone=T.CRIT if r["should_stop"] else T.OK),
        ]),
        unsafe_allow_html=True,
    )

    # 物理閘門推翻分類器時一定要講出來，否則畫面顯示「正常」但模型其實有意見，
    # 等於把系統內部的不一致藏起來。
    if r.get("gate_overrode"):
        st.warning(
            f"**兩層判斷不一致**　分類器判為「{r['gate_overrode']}」，"
            f"但頻譜上各項特徵都在同負載正常基準範圍內（最大偏離未達 1.5 倍），"
            f"找不到支持它的物理證據。本系統採「先偵測、再診斷」架構，"
            f"偵測層不經模型，因此依基準判定為正常 —— 但信心度不給「高」，"
            f"建議延長取樣後複檢。"
        )

    st.markdown("**信心度依據**")
    st.markdown(T.check_list(verdict["checks"]), unsafe_allow_html=True)
    st.caption(T.CONF_NOTE)

    # 分類與嚴重度是兩件事：模型認出故障特徵，不代表現在就嚴重。
    # 健康度還好時主動解釋，避免使用者看到「故障名稱 + 綠色分數」以為系統出錯。
    if r["fault_note"]:
        st.info(f"**{r['fault_label']}** — {r['fault_note']}")

    # 已知混淆：主動揭露這個類別的辨識限制。
    # 不對心與不平衡在跨負載測試中互相混淆（Misalign precision 0.70、
    # Unbalance recall 0.77）。與其等評審或現場人員發現，不如自己講，
    # 並給出「兩者都檢查」的可行建議——這比藏起來有用也更誠實。
    if r.get("known_confusion"):
        st.warning(f"**辨識限制**　{r['known_confusion']}")

    st.markdown("")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("健康分項")
        html = "".join(score_bar(k, v) for k, v in r["health_scores"].items())
        st.markdown(html, unsafe_allow_html=True)
        # ⚠️ 這行不要用 `~` 當範圍符號。Streamlit 的 Markdown 會把一行裡的
        # 兩個 `~` 當成刪除線語法吃掉，畫面上就變成「綠 10070／黃 6930」
        # ——分數區間直接消失（debug 清單第 7 點）。用連接號 – 就沒事。
        st.caption(
            f"{T.BAND_OK['dot']} 綠 {T.BAND_OK['range']}／"
            f"{T.BAND_WARN['dot']} 黃 {T.BAND_WARN['range']}／"
            f"{T.BAND_BAD['dot']} 紅 {T.BAND_BAD['range']} 分。"
            "拆成多個面向比單一數字更容易判斷問題出在哪一環。"
        )

    with right:
        st.subheader("觸發的判斷規則")
        if r["triggered_rules"]:
            for rule in r["triggered_rules"]:
                st.markdown(f"- {rule}")
        else:
            st.info("沒有觸發任何異常規則，各感測來源一致顯示正常。")

    st.markdown("---")
    a, b, c3 = st.columns(3)
    with a:
        st.subheader("可能原因")
        for x in r["causes"]:
            st.markdown(f"- {x}")
    with b:
        st.subheader("建議檢查")
        for x in r["checks"]:
            st.markdown(f"- {x}")
    with c3:
        st.subheader("建議行動")
        for x in r["actions"]:
            st.markdown(f"- {x}")

    # 工具 / 備品 / 安全 / 工時：現場工程師真正要問的東西。
    # 收在 expander 裡，讓主畫面維持乾淨，但需要時一鍵展開。
    if r.get("tools") or r.get("parts") or r.get("safety") or r.get("effort"):
        with st.expander("🧰 施工準備（工具、備品、安全事項、預估工時）"):
            t1, t2, t3 = st.columns(3)
            with t1:
                st.markdown("**需要的工具**")
                for x in r.get("tools") or ["（知識庫未收錄）"]:
                    st.markdown(f"- {x}")
            with t2:
                st.markdown("**可能需要的備品**")
                for x in r.get("parts") or ["（知識庫未收錄）"]:
                    st.markdown(f"- {x}")
            with t3:
                st.markdown("**安全注意事項**")
                for x in r.get("safety") or ["（無特別事項）"]:
                    st.markdown(f"- ⚠️ {x}")
            if r.get("effort"):
                st.info(f"**預估工時**　{r['effort']}")

    st.markdown("---")
    st.subheader("嚴重度隨時間變化")

    sev = r.get("severity_series")
    if sev:
        # 主線畫「物理嚴重度」而不是分類器機率。
        # 分類器機率是決策不是量測，在明確樣本上會飽和成一條水平線
        # （實測 FP-10 十七個窗全距 0.000002）；物理量才會真的動。
        srs = pd.DataFrame({f"{sev['名稱']}（倍）": sev["ratios"]})
        st.altair_chart(
            ch.line(srs,
                    x_title="時間（秒，每點為 1 秒訊號窗）",
                    y_title="相對同負載正常基準的倍數（1.0 = 與正常相同）",
                    height=300),
            use_container_width=True,
        )
        _r = pd.Series(sev["ratios"])
        st.caption(
            f"量測項目：**{sev['名稱']} @ {sev['位置']}**　·　"
            f"同負載正常基準 {sev['正常基準']:.4g}　·　"
            f"本段逐窗 {_r.min():.2f}× ~ {_r.max():.2f}×（中位 {_r.median():.2f}×）。"
            f"這條線直接來自頻譜，不經模型 —— 就算不相信 AI，也可以自己拿頻譜圖核對。"
        )
        with st.expander("這個量測項目代表什麼？"):
            st.markdown(sev["物理意義"])
    else:
        st.info("此模式沒有正常基準檔，無法計算物理嚴重度。")

    # 分類器機率降為次要參考。留著是因為它仍然是判斷的一部分，
    # 但畫在下面並且明說它為什麼常常是一條直線。
    with st.expander("分類器的異常機率（次要參考）"):
        vib = r["modality_detail"][r["primary_key"]]
        series = pd.DataFrame({"振動異常機率": vib["anomaly_series"]})
        if "current" in r["modality_detail"] and r.get("current_usable"):
            cur_s = r["modality_detail"]["current"]["anomaly_series"]
            n = min(len(series), len(cur_s))
            series = series.iloc[:n].copy()
            series["電流異常機率"] = cur_s[:n]

        st.altair_chart(
            ch.line(series,
                    x_title="時間（秒，每點為 1 秒訊號窗）",
                    y_title="異常分數（0–1，1 = 完全不像正常）",
                    y_domain=(0.0, 1.0), height=260),
            use_container_width=True,
        )
        _flat = series["振動異常機率"]
        if float(_flat.max() - _flat.min()) < 0.02:
            st.caption(
                f"**這條線是平的（約 {float(_flat.median()):.3f}）。這是正常現象，不是圖表故障。**　"
                "樹模型在明確樣本上機率會飽和到 0 或 1，實測整段訊號的全距只有 1e-6 量級。"
                "這正是我們把趨勢圖主線改成物理嚴重度的原因：機率是決策，不是量測。"
            )
        else:
            st.caption(
                "每個點是一秒的訊號窗。系統取中位數而非平均值判定，"
                "避免單一瞬間干擾造成誤報——工業現場的原則是「持續異常才算異常」。"
            )


# ---------------------------------------------------------------- 多感測證據
elif page == "多感測證據":
    require_models()

    machines = sim.fleet(mode)
    labels = {m.machine_id: f"{m.machine_id}　{m.name}" for m in machines}
    mid = st.selectbox("選擇機台", [m.machine_id for m in machines],
                       format_func=lambda k: labels[k], key="ev")
    machine = sim.machine_by_id(mode, mid)
    r = cached_diagnose(mode, machine.source)
    conf = r["modality_detail"][r["primary_key"]]["confidence"]
    verdict = r["confidence_verdict"]

    st.markdown(
        T.command_bar(
            "EVIDENCE · 多感測證據",
            f"{mid} {machine.name}　·　這一頁回答「憑什麼相信這個結果」",
            r["fault_label"],
            f"信心度 {verdict['level']}（{verdict['summary']}）",
        ),
        unsafe_allow_html=True,
    )

    # ---- 這個結論是哪個模型做出來的 ----
    scope = r.get("model_scope", {}).get(r["primary_key"])
    if scope is not None:
        bundle = dd.load_lolo(r["primary_key"], scope)
        acc = bundle.get("holdout_accuracy")
        f1 = bundle.get("holdout_macro_f1")
        st.success(
            f"**這台機台的判斷來自「沒看過 {scope} Nm」的模型。**　"
            f"訓練資料為 {'、'.join(f'{l} Nm' for l in bundle['training_loads'])}"
            f"（{bundle['n_train']:,} 個時間窗），{scope} Nm 完全排除在訓練之外。"
            + (f"　該模型在這個保留負載上的逐窗準確率 {acc:.1%}、macro-F1 {f1:.1%}。"
               if acc is not None else "")
        )
    elif mode == "load":
        st.error(
            "**這個結論來自看過同一支錄音的模型**，依《數字口徑表》不可對外引用。"
            "請執行 `python train_lolo.py` 後重啟。"
        )

    st.subheader("信心度依據")
    st.markdown(
        f"目前等級 **{T.conf_dot(verdict['level'])}**　·　{verdict['summary']}　—　"
        f"{verdict['note']}"
    )
    st.markdown(T.check_list(verdict["checks"]), unsafe_allow_html=True)
    st.caption(T.CONF_NOTE)

    with st.expander("信心度分級 vs 實際對錯（全部 45 支錄音的稽核結果）"):
        _audit = load_audit()
        if _audit is None:
            st.info("尚未產生稽核表。執行 `python train_lolo.py` 會一併輸出 "
                    "`results/lolo_confidence.csv`。")
        else:
            tab = (_audit.groupby("level")
                   .agg(錄音檔數=("correct", "size"), 判對=("correct", "sum"))
                   .reindex(["高", "中", "低"]).dropna(how="all").reset_index())
            tab["正確率"] = (tab["判對"] / tab["錄音檔數"]).map(lambda v: f"{v:.0%}")
            tab.columns = ["信心度", "錄音檔數", "判對", "正確率"]
            st.dataframe(tab, hide_index=True, use_container_width=True)
            st.markdown(
                "每一支錄音都由「沒看過它那個負載」的模型判定，等級是**判定前**算出來的，"
                "對錯是**事後**比對的。判錯的幾支：")
            wrong = _audit[_audit.correct == 0][
                ["file_id", "truth", "predicted", "level", "agreement"]]
            wrong.columns = ["錄音檔", "真實狀態", "系統研判", "信心度", "逐窗一致率"]
            st.dataframe(wrong, hide_index=True, use_container_width=True)
            st.caption(
                "注意「逐窗一致率」那一欄：判錯的檔案一致率都在 89% 以上。"
                "這就是為什麼信心度不能用模型自己的一致率或機率 —— "
                "它在錯的時候一樣很有把握。45 支錄音樣本很小，"
                "上表是佐證不是機率保證。"
            )

    st.markdown("---")
    st.subheader("各感測來源的意見")
    ev = pd.DataFrame(r["evidence"])
    if not r["has_temperature"]:
        ev = ev[ev["來源"] != "溫度"]
    if not r["uses_current"]:
        ev = ev[ev["來源"] != "電流"]
    st.dataframe(ev, hide_index=True, use_container_width=True)

    if not r["has_current"]:
        st.info(
            "此模式使用「振動+電流融合」的單一模型：電流有進到模型輸入，"
            "但模型只輸出一個綜合判斷，無法拆出各感測器的獨立意見，"
            "因此不做跨感測交叉驗證。"
        )
    if not r["has_temperature"]:
        st.info("變轉速資料集沒有溫度感測，因此不做過熱風險與惡化趨勢判斷。")
    else:
        # 主動揭露溫度門檻的校正限制。這是規則引擎裡最不穩固的一環，
        # 與其被問到才解釋，不如寫在畫面上——顯示我們知道自己的邊界在哪。
        st.caption(
            "⚠️ 溫度門檻（+1／+3／+5°C）是用本資料集 60~300 秒的錄音校出來的，"
            "且不同檔案錄製於不同日期、環境溫度不同。這組門檻適用於本驗證情境；"
            "實際部署時必須改用該設備自身的長期資料重新校正。"
            "振動與電流的判準不受此限制。"
        )

    # ── 物理證據 ────────────────────────────────────────────────────────
    # 這是這一頁真正該回答「憑什麼相信」的部分。上面那張表是各感測來源的
    # 異常機率（模型的意見），這裡是實際量到的物理量與正常基準的比較（證據本身）。
    if r.get("physical_evidence"):
        st.markdown("---")
        st.subheader("物理證據：實際量到什麼")
        st.caption(
            "與「同負載下正常機台」的基準值比較。這些數字直接來自頻譜，"
            "不經模型——即使不相信 AI，也可以自己拿頻譜圖核對。"
        )
        pe = pd.DataFrame([{
            "量測項目": e["名稱"],
            "感測位置": e["位置"],
            "實測值": e["實測值"],
            "正常基準": e["正常基準"],
            "倍數": f"{e['倍數']}×",
        } for e in r["physical_evidence"]])
        st.dataframe(pe, hide_index=True, use_container_width=True)

        with st.expander("這些量測項目代表什麼？"):
            for e in r["physical_evidence"]:
                st.markdown(f"**{e['名稱']}** — {e['物理意義']}")

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("模型在各時間窗的判斷分布")
        vd = r["modality_detail"][r["primary_key"]]["vote_distribution"]
        vote = pd.DataFrame([(dd.fault_display(k), v) for k, v in vd.items()],
                            columns=["判斷結果", "窗數"]).sort_values("窗數", ascending=False)
        # 橫向長條圖。原本用 st.bar_chart，它把「判斷結果」這個類別欄位
        # 當成數值軸處理，Y 軸標成 0/20/40/…（沒有意義），中文標籤還被擠成直排。
        st.altair_chart(
            ch.hbar(vote, "判斷結果", "窗數",
                    x_title="時間窗數量（個，每窗 1 秒）",
                    y_title="模型的判斷結果", height=260),
            use_container_width=True,
        )
        st.markdown(
            f'**逐窗多數決一致率**　<span class="mono" style="color:{T.TEXT};'
            f'font-weight:700">{conf:.0%}</span>', unsafe_allow_html=True)
        st.caption(
            "一致率高代表模型在不同時間窗都給出相同答案；"
            "低於 60% 代表判斷搖擺，這種情況應轉人工複檢。"
            "注意這是「模型前後說法一不一致」，不是「診斷正確率」——"
            "訊號穩定時本來就容易接近 100%。"
        )

    with right:
        st.subheader("這個模型最看重哪些特徵")
        imp = dd.feature_importance(mode, dd.primary_modality(mode), top=10)
        if not imp.empty:
            # 同樣改橫向：`ch3_env_BPFO_sideband` 這種長特徵名，
            # 直向長條圖只能塞進 `ch3_env_BPFO_s...`，看不出是哪個特徵。
            st.altair_chart(
                ch.hbar(imp.rename(columns={"feature": "特徵", "importance": "重要度"}),
                        "特徵", "重要度",
                        x_title="特徵重要度（Gini importance，全部特徵合計為 1）",
                        y_title="特徵名稱", height=300, val_format=".3f"),
                use_container_width=True,
            )
            st.caption(
                "直接取自當前模式的模型。ch0~ch3 分別是軸承座 A/B 的 x/y 方向振動。"
                "`env_BPFO` / `env_BPFI` 是包絡譜在軸承內外環特徵頻率的譜線強度，"
                "`order_1x` / `ratio_3x_1x` 是轉頻諧波與諧波比，"
                "`kurtosis` 是波形峰度（對週期性衝擊敏感）。"
                "注意這是模型的「全域」重要度——它整體上倚重哪些特徵；"
                "這一台為什麼被判成現在這樣，請看上方的「物理證據」。"
            )
        else:
            st.info("這個模型沒有提供特徵重要度")

    st.markdown("---")
    st.subheader("判斷邏輯")
    st.markdown(
        """
        1. **特徵萃取對準物理** — 振動訊號先做包絡解調，量測軸承內外環特徵頻率
           （BPFO 183.5 Hz / BPFI 268.8 Hz）的譜線強度，以及轉頻諧波（1x 50.2 Hz、
           3x 150.4 Hz）。不是丟一堆統計量給模型硬學，每個特徵都對應一個已知的故障機制
        2. **各感測器獨立判斷** — 振動模型與電流模型分別對每一秒的訊號給出異常機率與故障分類
        3. **規則引擎交叉驗證** — 兩者一致時提高可信度；只有單一來源異常時降級為「疑似」並要求複檢
        4. **溫度判斷急迫性** — 溫度不參與故障分類（它變化太慢），只用來判斷是否持續惡化、需不需要立即停機
        5. **對照知識庫** — 依故障類型取出對應的可能原因、檢查項目與處置方式
        """
    )
    st.caption(
        "刻意不讓語言模型直接看原始訊號下判斷——所有結論都來自可追溯的模型輸出與規則，"
        "語言模型只負責把結構化結果轉成人看得懂的文字。"
    )


# ---------------------------------------------------------------- 即時診斷
elif page == "即時診斷":
    require_models()
    import upload_ingest as ui

    st.markdown(
        T.command_bar(
            "LIVE ANALYSIS · 即時診斷",
            "上傳原始感測檔，系統自動轉換並判斷",
            dd.MODE_LABELS[mode],
            "使用當前運轉模式的模型",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "**支援格式**\n\n"
        "- `.mat` — 振動原始波形（4 通道：軸承座 A/B 的 x/y 方向）\n"
        "- `.tdms` — 電流與溫度原始波形\n"
        "- `.csv` — 已算好的特徵表\n\n"
        "上傳後系統會用與訓練時完全相同的流程處理："
        "去除直流偏移 → 切成 1 秒時間窗 → 計算特徵 → 模型推論。"
    )

    tab_up, tab_demo = st.tabs(["上傳檔案", "用廠內機台示範"])

    X = None
    src_note = ""

    with tab_up:
        # 模型訓練時把運轉條件（負載）當成特徵——負載對訊號的影響比故障還大，
        # 不告訴模型現在幾 Nm，它分不出「4Nm 正常」與「0Nm 不對心」。
        # 上傳的檔案沒有這個資訊，所以必須由使用者指定。
        up_load = None
        if mode == "load":
            up_load = st.selectbox(
                "這段訊號的扭矩負載", [0, 2, 4],
                format_func=lambda v: f"{v} Nm",
                help="模型把負載當成輸入條件之一，選錯會影響判斷結果。",
            )

        up = st.file_uploader("選擇檔案", type=["mat", "tdms", "csv"])
        if up is not None:
            # 特徵版本必須跟著當前模式的模型走，不能寫死
            _fv = "v2" if dd.vibration_source(mode) == "vibration_v2" else "v1"
            with st.spinner(f"正在轉換 {up.name}...（只取前 60 秒）"):
                try:
                    res = ui.ingest(up, feature_version=_fv, load_nm=up_load)
                except ui.IngestError as e:
                    st.error(str(e))
                    st.stop()
            st.success(f"{up.name}　{res['note']}")
            if res["kind"] == "current":
                st.warning(
                    "這是電流/溫度檔。目前的主力模型以振動為輸入，"
                    "電流檔可用來看特徵，但無法單獨完成故障分類——請改上傳 .mat 振動檔。"
                )
                st.dataframe(res["features"].head(), use_container_width=True)
                if res["temperature"]:
                    st.subheader("溫度趨勢摘要")
                    st.json(res["temperature"])
                st.stop()
            X = res["features"]
            src_note = f"{up.name}（{res['n_windows']} 個時間窗）"

    with tab_demo:
        machines = sim.fleet(mode)
        labels = {m.machine_id: f"{m.machine_id}　{m.name}" for m in machines}
        mid = st.selectbox("選一台機台模擬上傳", [m.machine_id for m in machines],
                           format_func=lambda k: labels[k], key="rt")
        if st.checkbox("載入這台機台的資料", value=False):
            machine = sim.machine_by_id(mode, mid)
            Xall, _, meta = dd.split_features(mode, dd.primary_modality(mode), "test")
            X = Xall[(meta.file_id == machine.source).values]
            src_note = f"{mid} {machine.name}（{len(X)} 個時間窗）"
            st.info(f"已載入 {src_note}")

    if X is not None and len(X):
        st.markdown("---")
        if st.button("開始診斷", type="primary"):
            bundle = dd.load_model(mode, dd.primary_modality(mode))
            missing = [c for c in bundle["feature_names"] if c not in X.columns]
            if missing:
                st.error(
                    f"缺少模型需要的欄位（共 {len(missing)} 個，前 5 個）：{missing[:5]}\n\n"
                    "常見原因：上傳的檔案通道數不足，或格式與訓練資料不同。"
                )
                st.stop()

            with st.spinner("模型推論中..."):
                res = dd.infer_modality(bundle, X)

            # 把結果存進 session_state，這樣頁面重跑時不會消失。
            # llm_diag 是給 LLM 助手用的完整診斷（含知識庫與量測證據），
            # 不能只丟 infer_modality 的原始輸出，否則 LLM 沒有證據可引用。
            st.session_state["live_diag_result"] = res
            st.session_state["live_diag_llm"] = dd.build_llm_diag(mode, res, X)
            st.session_state["live_diag_src_note"] = src_note

    # ── 從 session_state 讀取診斷結果來顯示 ──────────────────────────
    # 這段放在 st.button 外面，這樣 chat_input 觸發 rerun 時結果仍然顯示。
    if "live_diag_result" in st.session_state:
        res = st.session_state["live_diag_result"]
        src_note_display = st.session_state["live_diag_src_note"]
        live_verdict = (st.session_state.get("live_diag_llm") or {}).get(
            "confidence_verdict") or {"level": "—", "summary": "", "checks": []}

        st.markdown("---")
        st.subheader(f"診斷結果　—　{src_note_display}")
        ap = res["anomaly_prob"]
        st.markdown(
            T.kpi_row([
                dict(label="異常機率", value=f"{ap:.0%}",
                     sub="1 - P(正常) 的中位數",
                     tone=T.tone_color((1 - ap) * 100)),
                dict(label="研判故障", value=dd.fault_display(res["fault_type"]),
                     sub="逐窗多數決", tone=T.ACCENT),
                dict(label="信心度", value=live_verdict["level"],
                     sub=live_verdict["summary"],
                     tone=T.conf_color(live_verdict["level"])),
                dict(label="分析時間窗", value=str(res["n_windows"]),
                     sub="每窗 1 秒訊號", tone=T.ACCENT),
            ]),
            unsafe_allow_html=True,
        )
        st.markdown("**信心度依據**")
        st.markdown(T.check_list(live_verdict["checks"]), unsafe_allow_html=True)
        st.caption(T.CONF_NOTE)

        _sev = (st.session_state.get("live_diag_llm") or {}).get("severity_series")
        if _sev:
            st.subheader("嚴重度隨時間變化")
            st.altair_chart(
                ch.line(pd.DataFrame({f"{_sev['名稱']}（倍）": _sev["ratios"]}),
                        x_title="時間窗編號（每窗 1 秒訊號）",
                        y_title="相對同負載正常基準的倍數（1.0 = 與正常相同）",
                        height=280),
                use_container_width=True,
            )
            st.caption(f"量測項目：{_sev['名稱']} @ {_sev['位置']}　·　"
                       f"正常基準 {_sev['正常基準']:.4g}　·　這條線直接來自頻譜，不經模型。")

        st.subheader("各時間窗的異常程度（分類器機率，次要參考）")
        st.altair_chart(
            ch.line(pd.DataFrame({"異常分數": res["anomaly_series"]}),
                    x_title="時間窗編號（每窗 1 秒訊號）",
                    y_title="模型異常分數（0–1，1 = 完全不像正常）",
                    y_domain=(0.0, 1.0), height=280),
            use_container_width=True,
        )

        st.subheader("判斷分布")
        vd = pd.DataFrame(
            [(dd.fault_display(k), v) for k, v in res["vote_distribution"].items()],
            columns=["判斷結果", "窗數"]).sort_values("窗數", ascending=False)
        st.dataframe(vd, hide_index=True, use_container_width=True)

        if res["confidence"] < 0.6:
            st.warning(
                "模型在不同時間窗之間判斷不一致（一致率低於 60%），"
                "建議延長取樣時間或轉人工複檢。"
            )

        # ---- LLM 維修助手 ------------------------------------------------
        st.markdown("---")
        st.subheader("🤖 LLM 維修助手")

        llm_diag = st.session_state.get("live_diag_llm", res)

        llm_ok, llm_msg = llm_mod.check_availability()
        if not llm_ok:
            # LLM 不可用時不能只顯示一則警告就結束 —— demo 當天沒網路或
            # 額度用完就會變成整頁空白。改成直接顯示離線摘要，
            # 內容全部來自本地的診斷結果與量測證據，只是沒有語言模型潤飾。
            st.warning(llm_msg)
            st.info("目前以離線模式顯示：以下內容由診斷結果與量測證據直接產生。")
            import evidence as ev_mod
            st.markdown(
                "```\n"
                + ev_mod.fallback_summary(llm_diag, llm_diag.get("evidence") or [])
                + "\n```"
            )
        else:
            # session_state key 包含 src_note，這樣換了診斷資料就自動重置對話
            state_key = f"llm_assistant_{src_note_display}"
            history_key = f"llm_history_{src_note_display}"
            summary_key = f"llm_summary_{src_note_display}"

            # 初始化 session state
            if state_key not in st.session_state:
                # 傳完整的 llm_diag（含知識庫與量測證據），不是 infer_modality 的原始輸出
                st.session_state[state_key] = llm_mod.LLMAssistant(llm_diag)
            if history_key not in st.session_state:
                st.session_state[history_key] = []  # [{"role", "content"}]

            assistant: llm_mod.LLMAssistant = st.session_state[state_key]

            # ── 自動摘要區 ──
            col_btn, col_clear = st.columns([3, 1])
            with col_btn:
                gen_summary = st.button(
                    "📋 自動產生維修建議摘要",
                    key="llm_gen_summary",
                    help="根據診斷結果，讓 LLM 幫你整理一份給維修工程師的摘要",
                )
            with col_clear:
                if st.button("🗑️ 清除對話", key="llm_clear"):
                    assistant.clear_history()
                    st.session_state[history_key] = []
                    st.session_state.pop(summary_key, None)
                    st.rerun()

            if gen_summary:
                with st.spinner("LLM 正在生成維修建議摘要..."):
                    try:
                        summary = assistant.generate_summary()
                    except Exception as e:
                        st.error(f"LLM 呼叫失敗：{e}")
                        summary = None
                if summary:
                    # ⚠️ key 必須綁定 src_note。原本用固定的 "llm_summary"，
                    # 換一台機台重新診斷後，上一台的摘要仍然留在畫面上，
                    # 變成「A 機台的診斷結果配 B 機台的維修建議」——
                    # demo 時這種錯誤特別致命，因為看起來完全正常。
                    st.session_state[summary_key] = summary

            # 顯示已生成的摘要（從 session_state 讀取）
            if summary_key in st.session_state:
                # ⚠️ 這裡以前只是把換行換成 <br> 就塞進 div，等於把 LLM 回來的
                # Markdown 當純文字印出來 —— 畫面上直接看到 `**目前狀況**`
                # 這種語法（debug 清單第 14 點）。改用 T.note_panel，
                # 它會先把 Markdown 轉成 HTML 再套面板樣式。
                st.markdown(
                    T.note_panel(st.session_state[summary_key]),
                    unsafe_allow_html=True,
                )
                if assistant.last_error:
                    st.caption("⚠️ 此摘要為離線產生（LLM 服務目前無法連線），"
                               "內容來自診斷結果與量測證據，未經語言模型潤飾。")

            st.markdown("")

            # ── 對話歷史顯示 ──
            history: list[dict] = st.session_state[history_key]
            if history:
                st.markdown("**💬 對話記錄**")
                for turn in history:
                    if turn["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(turn["content"])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(turn["content"])

            # ── 輸入框 ──
            user_input = st.chat_input(
                "詢問維修建議，例如：要準備什麼工具？多久內要處理？",
                key="llm_chat_input",
            )
            if user_input:
                # 存使用者訊息
                st.session_state[history_key].append(
                    {"role": "user", "content": user_input}
                )
                # 呼叫 LLM
                with st.spinner("思考中..."):
                    try:
                        reply = assistant.chat(user_input)
                    except Exception as e:
                        reply = f"⚠️ 呼叫 LLM 時發生錯誤：{e}"
                # 存 LLM 回覆
                st.session_state[history_key].append(
                    {"role": "assistant", "content": reply}
                )
                # 重跑頁面以正確顯示完整的對話歷史
                st.rerun()

            st.caption(
                "LLM 的回答根據 ML 模型診斷結果與故障知識庫生成，"
                "不會自行編造診斷內容。仍需由現場工程師確認後執行。"
            )
