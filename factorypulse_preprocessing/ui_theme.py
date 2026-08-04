"""
戰情室（command center）視覺主題。

設計原則
--------
1. 深色底 + 青色霓虹：投影機打出來對比夠，暗房環境不刺眼
2. 數字用等寬字：監控畫面數字會一直跳動，等寬字不會讓版面左右晃
3. 面板加角標與細邊框：模仿工業 HMI / 軍事戰情系統的框線語彙
4. 顏色只用來表達狀態，不用來裝飾——綠黃紅各自有明確意義

刻意避免的：過多動畫、漸層、陰影。戰情室的重點是「快速讀懂」，
花俏效果會拖慢判讀速度，也會讓評審覺得在用視覺掩蓋內容。
"""

# 色票
BG = "#070d18"
PANEL = "#0d1626"
PANEL_HI = "#111d33"
LINE = "#1e3a5f"
ACCENT = "#22d3ee"
TEXT = "#dbeafe"
MUTED = "#64809f"

OK = "#22c55e"
WARN = "#eab308"
BAD = "#ef4444"
CRIT = "#dc2626"


# ---------------------------------------------------------------- 分級刻度
# 這裡是全系統唯一的分級來源。規則引擎用的風險切點是 30 / 50 / 70 / 85，
# 健康分數 = 100 - 風險分數，所以下面的顏色切點刻意對齊那些邊界：
#
#   風險 <30（健康）        -> 健康 >70  -> 綠
#   風險 30~50（注意）      -> 健康 50~70 ┐
#   風險 50~70（警告）      -> 健康 30~50 ┴ 黃
#   風險 70~85（高風險）    -> 健康 15~30 ┐
#   風險 >=85（危急）       -> 健康 <15   ┴ 紅
#
# 這樣「風險等級徽章」與「健康分數顏色」永遠一致：
# 不會出現徽章寫「高風險」但分數條卻是黃色的矛盾。
# 顏色只有三階是刻意的——監控畫面要能一眼分辨「可以放著 / 要排程 / 要處理」，
# 五種顏色反而會拖慢判讀速度；精確等級由徽章負責顯示。
HEALTH_CUT_OK = 70      # 對應風險 30，規則引擎的 healthy/attention 邊界
HEALTH_CUT_BAD = 30     # 對應風險 70，規則引擎的 warning/high 邊界


def tone_color(score: float) -> str:
    """健康分數 -> 顏色。切點與風險等級邊界對齊，見上方說明。

    注意這裡用嚴格大於（>）而不是大於等於（>=）。
    規則引擎的判斷是「risk >= 30 就算注意」，健康分數 = 100 - risk，
    所以健康剛好 70 分時風險正好是 30，等級已經是「注意」了。
    若這裡寫 >= 70 就會把它畫成綠色，跟徽章上的「注意」互相矛盾。
    邊界值只差一分，但監控系統的顏色不該跟文字說反話。
    """
    if score > HEALTH_CUT_OK:
        return OK
    if score > HEALTH_CUT_BAD:
        return WARN
    return BAD


def risk_tone_color(risk: float) -> str:
    """風險分數 -> 顏色。與 tone_color 互為反向，確保兩者不會給出矛盾的顏色。"""
    return tone_color(100 - risk)


CSS = f"""
<style>
/* ---------- 全域 ---------- */
.stApp {{
    background:
        radial-gradient(1200px 600px at 15% -10%, #10233d 0%, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, #0d2030 0%, transparent 55%),
        {BG};
}}
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {LINE};
}}
h1, h2, h3 {{ color: {TEXT} !important; letter-spacing: .5px; }}
h1 {{ font-weight: 800 !important; }}

/* 數字一律等寬，避免跳動時版面左右晃 */
.mono, .kpi-value, .metric-num {{
    font-family: "SF Mono", "Roboto Mono", "Cascadia Mono", Consolas, monospace;
    font-variant-numeric: tabular-nums;
}}

/* ---------- 頂部狀態列 ---------- */
.cmd-bar {{
    display:flex; align-items:center; gap:18px;
    background: linear-gradient(90deg, {PANEL_HI} 0%, {PANEL} 100%);
    border: 1px solid {LINE};
    border-left: 3px solid {ACCENT};
    border-radius: 6px; padding: 10px 16px; margin-bottom: 14px;
}}
.cmd-title {{ font-size: 19px; font-weight: 800; color: {TEXT}; letter-spacing: 1px; }}
.cmd-sub {{ font-size: 12px; color: {MUTED}; }}
.cmd-right {{ margin-left:auto; text-align:right; }}
.live-dot {{
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:{OK}; margin-right:6px;
    animation: blink 1.8s ease-in-out infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}

/* ---------- KPI 磚 ---------- */
.kpi-row {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.kpi {{
    position:relative; background:{PANEL}; border:1px solid {LINE};
    border-radius:6px; padding:12px 14px 10px 14px; overflow:hidden;
}}
.kpi::before, .kpi::after {{
    content:""; position:absolute; width:10px; height:10px;
    border-color:var(--tone,{ACCENT}); border-style:solid;
}}
.kpi::before {{ top:5px; left:5px; border-width:2px 0 0 2px; }}
.kpi::after  {{ bottom:5px; right:5px; border-width:0 2px 2px 0; }}
.kpi-label {{ font-size:11.5px; color:{MUTED}; letter-spacing:1.2px; text-transform:uppercase; }}
.kpi-value {{ font-size:31px; font-weight:800; line-height:1.15; color:var(--tone,{TEXT}); }}
.kpi-sub {{ font-size:11px; color:{MUTED}; }}

/* ---------- 面板 ---------- */
.panel {{
    background:{PANEL}; border:1px solid {LINE}; border-radius:6px;
    padding:14px 16px; margin-bottom:12px;
}}
.panel-title {{
    font-size:12.5px; color:{ACCENT}; letter-spacing:1.6px;
    text-transform:uppercase; margin-bottom:10px;
    border-bottom:1px solid {LINE}; padding-bottom:6px;
}}

/* ---------- 分數條 ---------- */
.bar-wrap {{ margin-bottom:11px; }}
.bar-head {{ display:flex; justify-content:space-between; font-size:13px; color:{TEXT}; }}
.bar-track {{
    background:#0a1424; border:1px solid {LINE};
    border-radius:3px; height:13px; overflow:hidden;
}}
.bar-fill {{ height:100%; border-radius:2px; }}

/* ---------- 標籤 ---------- */
.tag {{
    display:inline-block; padding:2px 9px; border-radius:3px;
    font-size:12px; font-weight:700; letter-spacing:.5px;
}}
.tag-out {{
    display:inline-block; padding:1px 8px; border-radius:3px;
    font-size:12px; font-weight:700; border:1px solid;
}}

/* ---------- 表格 ---------- */
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:6px; }}

/* Streamlit 預設元件微調 */
div[data-testid="stMetricValue"] {{
    font-family:"SF Mono","Roboto Mono",Consolas,monospace; color:{TEXT};
}}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
.stTabs [data-baseweb="tab"] {{
    background:{PANEL}; border:1px solid {LINE}; border-radius:4px 4px 0 0;
}}
</style>
"""


def inject():
    """在每個頁面最開頭呼叫一次。"""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def command_bar(title: str, subtitle: str, right_main: str, right_sub: str = "") -> str:
    return (
        f'<div class="cmd-bar">'
        f'  <div>'
        f'    <div class="cmd-title">{title}</div>'
        f'    <div class="cmd-sub">{subtitle}</div>'
        f'  </div>'
        f'  <div class="cmd-right">'
        f'    <div class="mono" style="font-size:15px;color:{TEXT}">'
        f'      <span class="live-dot"></span>{right_main}</div>'
        f'    <div class="cmd-sub">{right_sub}</div>'
        f'  </div>'
        f'</div>'
    )


def kpi(label: str, value: str, sub: str = "", tone: str = ACCENT) -> str:
    return (
        f'<div class="kpi" style="--tone:{tone}">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def kpi_row(items: list[dict]) -> str:
    inner = "".join(kpi(**it) for it in items)
    return f'<div class="kpi-row">{inner}</div>'


def panel(title: str, inner_html: str) -> str:
    return (f'<div class="panel"><div class="panel-title">{title}</div>'
            f'{inner_html}</div>')


def score_bar(label: str, score: float) -> str:
    c = tone_color(score)
    pct = max(0, min(100, score))
    return (
        f'<div class="bar-wrap">'
        f'  <div class="bar-head"><span>{label}</span>'
        f'  <span class="mono" style="color:{c};font-weight:700">{score:.0f}</span></div>'
        f'  <div class="bar-track"><div class="bar-fill" '
        f'       style="width:{pct}%;background:{c}"></div></div>'
        f'</div>'
    )


def tag(text: str, color: str, filled: bool = True) -> str:
    if filled:
        return f'<span class="tag" style="background:{color};color:#04121f">{text}</span>'
    return f'<span class="tag-out" style="color:{color};border-color:{color}">{text}</span>'


def confidence(conf: float) -> str:
    c = OK if conf >= 0.8 else (WARN if conf >= 0.6 else BAD)
    return (f'<span class="mono" style="color:{c};font-size:13px;font-weight:700">'
            f'CONF {conf:.0%}</span>')
