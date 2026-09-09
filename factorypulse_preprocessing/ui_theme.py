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

# 三階顏色帶的「顯示用文字」。全系統只有這一份，畫面上每個地方
# （KPI 卡片、廠房圖圖例、分數條說明）都從這裡取，不各自手寫。
# 之前圖例寫「正常 100–70」、分數條說明寫「綠 100~70」、KPI 又自己分了一組
# 桶子，三邊的邊界對不起來，才會出現「綠色機台 1 台但 KPI 說 3 台」。
#
# 邊界說明（tone_color 用嚴格大於，所以是這個切法）：
#   健康分數 71–100  ⟺ 風險 <30    -> 綠
#   健康分數 31–70   ⟺ 風險 30–69  -> 黃
#   健康分數 0–30    ⟺ 風險 ≥70    -> 紅
BAND_OK = {"dot": "🟢", "name": "健康", "range": "71–100", "sub": "可持續運轉", "color": OK}
BAND_WARN = {"dot": "🟡", "name": "注意・警告", "range": "31–70", "sub": "安排檢修", "color": WARN}
BAND_BAD = {"dot": "🔴", "name": "高風險・危急", "range": "0–30", "sub": "立即處理", "color": BAD}
BANDS = (BAND_OK, BAND_WARN, BAND_BAD)

# 風險等級徽章（五級）-> 對應的三階顏色帶。徽章負責「精確是哪一級」，
# 顏色負責「要不要動作」，兩者由這張表綁死，不會互相說反話。
LEVEL_BAND = {
    "健康": BAND_OK,
    "注意": BAND_WARN,
    "警告": BAND_WARN,
    "高風險": BAND_BAD,
    "危急": BAND_BAD,
}

# 五級徽章各自的健康分數區間（給畫面上的分級說明用）
LEVEL_RANGE = {
    "健康": "71–100",
    "注意": "51–70",
    "警告": "31–50",
    "高風險": "16–30",
    "危急": "0–15",
}

SCALE_NOTE = ("健康分數 = 100 − 風險分數。顏色三階："
              f"{BAND_OK['dot']} {BAND_OK['range']} {BAND_OK['sub']}／"
              f"{BAND_WARN['dot']} {BAND_WARN['range']} {BAND_WARN['sub']}／"
              f"{BAND_BAD['dot']} {BAND_BAD['range']} {BAND_BAD['sub']}。")


def level_dot(level: str) -> str:
    """風險等級 -> 「🔴 危急」這種帶顏色圓點的顯示字串。"""
    band = LEVEL_BAND.get(level)
    return f"{band['dot']} {level}" if band else level


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


# ---------------------------------------------------------------- 信心度
# 這裡原本叫「可信度」並顯示百分比，全廠 10 台清一色 100%，看起來像寫死的。
# 它其實是逐窗多數決的一致率，而且實測證明它分不出對錯：跨負載測試裡
# 唯一判錯的檔案一致率 96%，判對的那支反而只有 76%（詳見 confidence.py）。
#
# 現在信心度改由 confidence.assess() 依四條可檢核條件判定，只給等級不給百分比。
# 這個模組只負責「等級 -> 顏色」，判定邏輯不在這裡。
CONF_LEVEL_COLOR = {"高": OK, "中": WARN, "低": BAD}
CONF_LEVEL_DOT = {"高": "🟢", "中": "🟡", "低": "🔴"}

CONF_NOTE = ("信心度不是模型輸出的機率，而是四條可檢核條件（物理證據支持、"
             "跨感測一致、逐窗穩定、非已知混淆對）的通過情形。"
             "45 支錄音、每個故障條件僅一支，樣本數不足以校準出可靠的百分比，"
             "因此只給等級。")


def conf_color(level: str) -> str:
    return CONF_LEVEL_COLOR.get(level, MUTED)


def conf_dot(level: str) -> str:
    """「🟢 高」這種帶顏色圓點的顯示字串。"""
    return f"{CONF_LEVEL_DOT.get(level, '⚪')} {level}"


def confidence(level: str, summary: str = "") -> str:
    """信心度徽章。呼叫端自己加標籤文字，這裡只回傳「高（4/4 項條件成立）」。"""
    c = conf_color(level)
    extra = f"（{summary}）" if summary else ""
    return (f'<span class="mono" style="color:{c};font-size:13px;font-weight:700">'
            f'{level}{extra}</span>')


def check_list(checks: list[dict]) -> str:
    """把四條件攤成一張可讀的清單。

    信心度的重點不是那個等級，是「憑什麼」。把條件攤開來，
    看的人可以自己判斷要不要相信，而不是被要求相信一個數字。
    """
    rows = []
    for c in checks:
        if c["passed"] is None:
            mark, color = "—", MUTED
        elif c["passed"]:
            mark, color = "✓", OK
        else:
            mark, color = "✗", BAD
        rows.append(
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'padding:7px 0;border-bottom:1px solid {LINE}">'
            f'<span style="color:{color};font-weight:800;font-size:15px;'
            f'line-height:1.3;min-width:16px">{mark}</span>'
            f'<div><div style="color:{TEXT};font-size:13px;font-weight:600">'
            f'{c["label"]}</div>'
            f'<div style="color:{MUTED};font-size:12px;line-height:1.6">'
            f'{c["detail"]}</div></div></div>'
        )
    return f'<div style="margin:4px 0 10px">{"".join(rows)}</div>'


# ---------------------------------------------------------------- Markdown
# LLM 回來的是 Markdown。原本的做法是把換行換成 <br> 後塞進 HTML div，
# 結果 **粗體** 這類語法原封不動印在畫面上（見 debug 清單第 14 點）。
# Streamlit 的 st.markdown 會渲染，但那樣就套不上這個面板的樣式，
# 所以這裡自己做一層最小轉換：粗體、斜體、行內程式碼、標題、有序／無序清單。
# 只支援 LLM 實際會用到的語法，不做完整 Markdown 剖析——需求就這麼多。
import html as _html
import re as _re

_MD_BOLD = _re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = _re.compile(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)")
_MD_CODE = _re.compile(r"`([^`]+)`")
_MD_BULLET = _re.compile(r"^\s*[-*・]\s+(.*)$")
_MD_NUMBER = _re.compile(r"^\s*(\d+)[.)、]\s*(.*)$")
_MD_HEADING = _re.compile(r"^\s*#{1,6}\s+(.*)$")


def _md_inline(text: str) -> str:
    # 先跳脫再套規則：跳脫不會動到 * 與 `，所以規則仍然對得上，
    # 但使用者或 LLM 吐出來的 <script> 之類就進不了 DOM。
    out = _html.escape(text)
    out = _MD_CODE.sub(
        lambda m: f'<code style="background:#0a1424;border:1px solid {LINE};'
                  f'border-radius:3px;padding:0 4px;font-size:.92em">{m.group(1)}</code>',
        out,
    )
    out = _MD_BOLD.sub(rf'<strong style="color:{TEXT}">\1</strong>', out)
    out = _MD_ITALIC.sub(r"<em>\1</em>", out)
    return out


def md_to_html(text: str) -> str:
    """把 LLM 的 Markdown 轉成這個主題用得上的 HTML 片段。"""
    parts: list[str] = []
    open_list: str | None = None

    def close_list():
        nonlocal open_list
        if open_list:
            parts.append(f"</{open_list}>")
            open_list = None

    def open_as(kind: str, start: int | None = None):
        """開一個新清單。

        ⚠️ start 是必要的。LLM 的摘要長這樣：

            1. **目前狀況**：
            （空行）
            2. **最可能的原因**：

        中間的空行會結束前一個 <ol>，下一個 <ol> 預設又從 1 開始 ——
        畫面上就變成「1. 2. 3.」全部顯示成「1.」。把原本的數字帶進
        start 屬性，編號才會跟 LLM 寫的一致。
        """
        nonlocal open_list
        if open_list != kind:
            close_list()
            attr = f' start="{start}"' if kind == "ol" and start else ""
            parts.append(f'<{kind}{attr} style="margin:4px 0 8px 1.2rem;padding:0">')
            open_list = kind

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            close_list()
            continue

        m = _MD_HEADING.match(line)
        if m:
            close_list()
            parts.append(f'<div style="font-weight:700;color:{ACCENT};'
                         f'margin:12px 0 4px">{_md_inline(m.group(1))}</div>')
            continue

        m = _MD_BULLET.match(line)
        if m:
            open_as("ul")
            parts.append(f"<li>{_md_inline(m.group(1))}</li>")
            continue

        m = _MD_NUMBER.match(line)
        if m:
            open_as("ol", start=int(m.group(1)))
            parts.append(f"<li>{_md_inline(m.group(2))}</li>")
            continue

        close_list()
        parts.append(f'<div style="margin:5px 0">{_md_inline(line)}</div>')

    close_list()
    return "".join(parts)


def note_panel(md_text: str) -> str:
    """LLM 摘要用的面板：左側青色實線 + 深底，內容以 Markdown 渲染。"""
    return (
        f'<div style="background:#0d2137;border-left:4px solid {ACCENT};'
        f'padding:1rem 1.2rem;border-radius:6px;margin:0.5rem 0;'
        f'color:{TEXT};line-height:1.75">{md_to_html(md_text)}</div>'
    )
