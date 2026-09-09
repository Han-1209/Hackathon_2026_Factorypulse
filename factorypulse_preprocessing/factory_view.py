"""
廠房平面圖（SVG）— 戰情室風格。

為什麼用 SVG 而不是圖片或 3D 套件
--------------------------------
- 機台顏色、數值要隨診斷結果即時改變，靜態圖片做不到
- Streamlit 能直接吃 HTML/SVG，不必額外裝 plotly / three.js
- 向量圖放大不糊，投影機投出來也清楚

視角採「俯視 + 淺立體」而不是完整 3D 等角：
完整 3D 看起來炫，但機台會互相遮擋、文字也會歪斜。
監控畫面最重要的是「掃一眼就知道哪台有問題」，可讀性優先於視覺效果。
"""

from __future__ import annotations

import ui_theme as T

# 佈局參數（SVG 座標）
W, H = 1120, 600
MACH_W, MACH_H = 158, 96
DEPTH = 15
COL_GAP = 40
ROW_A_Y, ROW_B_Y = 148, 378
LEFT = 74
HEAD_H = 62


def _health_color(score: float) -> str:
    return T.tone_color(score)


# ---------------------------------------------------------------- 文字寬度
# SVG 的 <text> 不會自動換行也不會裁切，超出機台方塊就直接畫到隔壁去
# （debug 清單第 1 點：「早期跡象：轉子不平衡」10 個字疊到右邊的邊框上）。
# 這裡先估寬度，塞不下就先縮字級，還是塞不下才截字加省略號——
# 縮字級比截字保留更多資訊，所以優先。
def _text_width(s: str, size: float) -> float:
    """估算字串在 SVG 裡的寬度（px）。

    中日韓字元是全形，寬度約等於字級；ASCII 約 0.55 倍。
    這只是估算，但誤差遠小於我們留的邊距，足夠用來決定要不要縮排。
    """
    w = 0.0
    for ch in s:
        w += size if ord(ch) > 0x2E80 else size * 0.55
    return w


def _fit_text(s: str, size: float, max_w: float, min_size: float = 8.5) -> tuple[str, float]:
    """回傳 (實際要畫的字串, 實際字級)。"""
    while size > min_size and _text_width(s, size) > max_w:
        size -= 0.5
    if _text_width(s, size) <= max_w:
        return s, size
    cut = s
    while cut and _text_width(cut + "…", size) > max_w:
        cut = cut[:-1]
    return (cut + "…") if cut else s, size


def _defs() -> str:
    """漸層、輝光濾鏡、網格圖樣。"""
    return f"""
    <defs>
      <pattern id="grid" width="34" height="34" patternUnits="userSpaceOnUse">
        <path d="M 34 0 L 0 0 0 34" fill="none" stroke="{T.LINE}"
              stroke-width="0.7" opacity="0.55"/>
      </pattern>
      <linearGradient id="floorGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0c1729"/>
        <stop offset="100%" stop-color="#070f1c"/>
      </linearGradient>
      <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3.2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="softglow" x="-40%" y="-40%" width="180%" height="180%">
        <feGaussianBlur stdDeviation="1.6" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    """


def _corner_brackets(x, y, w, h, color, size=13, sw=1.8) -> str:
    """四角的 L 形角標——工業 HMI / 戰情系統的典型框線語彙。"""
    return "".join([
        f'<path d="M{x},{y+size} L{x},{y} L{x+size},{y}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}"/>',
        f'<path d="M{x+w-size},{y} L{x+w},{y} L{x+w},{y+size}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}"/>',
        f'<path d="M{x+w},{y+h-size} L{x+w},{y+h} L{x+w-size},{y+h}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}"/>',
        f'<path d="M{x+size},{y+h} L{x},{y+h} L{x},{y+h-size}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}"/>',
    ])


def _machine(x: float, y: float, m: dict) -> str:
    health = m["health"]
    c = _health_color(health)
    stop = m["should_stop"]

    pulse = ('<animate attributeName="opacity" values="1;0.25;1" '
             'dur="1.3s" repeatCount="indefinite"/>') if stop else ""

    # 立體側面。刻意壓低不透明度：顏色在這個畫面是用來傳達狀態的，
    # 大面積色塊會蓋過真正要看的數字，只保留剛好夠的深度感即可。
    side = (
        f'<polygon points="{x+MACH_W},{y} {x+MACH_W+DEPTH},{y-DEPTH} '
        f'{x+MACH_W+DEPTH},{y+MACH_H-DEPTH} {x+MACH_W},{y+MACH_H}" '
        f'fill="{c}" fill-opacity="0.10"/>'
        f'<polygon points="{x},{y} {x+DEPTH},{y-DEPTH} '
        f'{x+MACH_W+DEPTH},{y-DEPTH} {x+MACH_W},{y}" fill="{c}" fill-opacity="0.14"/>'
        f'<line x1="{x+MACH_W}" y1="{y}" x2="{x+MACH_W+DEPTH}" y2="{y-DEPTH}" '
        f'stroke="{c}" stroke-width="0.9" opacity="0.5"/>'
        f'<line x1="{x}" y1="{y}" x2="{x+DEPTH}" y2="{y-DEPTH}" '
        f'stroke="{c}" stroke-width="0.9" opacity="0.5"/>'
    )

    # 機體
    body = (
        f'<rect x="{x}" y="{y}" width="{MACH_W}" height="{MACH_H}" rx="3" '
        f'fill="#0a1526" stroke="{c}" stroke-width="1.4" opacity="0.95"/>'
        f'<rect x="{x}" y="{y}" width="{MACH_W}" height="3" fill="{c}" filter="url(#softglow)"/>'
    )
    brackets = _corner_brackets(x - 4, y - 4, MACH_W + 8, MACH_H + 8, c, size=11, sw=1.6)

    # 主軸：健康轉得快、故障轉得慢，多一層直覺線索
    cx, cy = x + 30, y + 58
    spin = "1.5s" if health >= 70 else ("2.8s" if health >= 30 else "5s")
    spindle = (
        f'<circle cx="{cx}" cy="{cy}" r="17" fill="none" stroke="{c}" '
        f'stroke-width="1.1" opacity="0.45"/>'
        f'<circle cx="{cx}" cy="{cy}" r="17" fill="none" stroke="{c}" stroke-width="1.8" '
        f'stroke-dasharray="26 80" opacity="0.9">'
        f'<animateTransform attributeName="transform" type="rotate" '
        f'from="0 {cx} {cy}" to="360 {cx} {cy}" dur="{spin}" repeatCount="indefinite"/>'
        f'</circle>'
        f'<circle cx="{cx}" cy="{cy}" r="6" fill="{c}" opacity="0.28"/>'
        f'<circle cx="{cx}" cy="{cy}" r="2.6" fill="{c}" filter="url(#softglow)"/>'
    )

    # 狀態燈
    led = (f'<circle cx="{x+MACH_W-14}" cy="{y+13}" r="4.2" fill="{c}" '
           f'filter="url(#softglow)">{pulse}</circle>')

    tx = x + 56
    # 右欄可用寬度：從 tx 到機體右緣，再留 12px 邊距——
    # 留太少的話文字會貼到右邊的角標上，看起來仍然像溢出。
    inner_w = MACH_W - (tx - x) - 12
    fault_txt, fault_size = _fit_text(m["fault"], 10.5, inner_w)

    # 「可信度 100%」在每一台上都一樣，看起來像寫死的（debug 清單第 6 點）。
    # 現在顯示的是證據型信心度等級，由 confidence.py 依四條件判定。
    conf_text = m.get("confidence_level", "—")
    conf_color = T.conf_color(conf_text)

    txt = (
        f'<text x="{x+9}" y="{y+17}" font-size="12" font-weight="700" fill="{c}" '
        f'font-family="monospace" letter-spacing="1">{m["machine_id"]}</text>'
        f'<text x="{x+9}" y="{y+MACH_H-8}" font-size="9" fill="{T.MUTED}">{m["name"]}</text>'
        f'<text x="{tx}" y="{y+52}" font-size="30" font-weight="800" fill="{c}" '
        f'font-family="monospace" filter="url(#softglow)">{health:.0f}</text>'
        f'<text x="{tx+45}" y="{y+52}" font-size="10" fill="{T.MUTED}" '
        f'font-family="monospace">/100</text>'
        f'<text x="{tx}" y="{y+68}" font-size="{fault_size}" fill="{T.TEXT}">'
        f'<title>{m["fault"]}</title>{fault_txt}</text>'
        f'<text x="{tx}" y="{y+82}" font-size="9" fill="{T.MUTED}">信心度 '
        f'<tspan fill="{conf_color}" font-weight="700">{conf_text}</tspan></text>'
    )

    warn = ""
    if stop:
        warn = (
            f'<rect x="{x+MACH_W-52}" y="{y+MACH_H-22}" width="46" height="16" rx="2" '
            f'fill="none" stroke="{T.BAD}" stroke-width="1.2">{pulse}</rect>'
            f'<text x="{x+MACH_W-29}" y="{y+MACH_H-10}" font-size="9.5" font-weight="700" '
            f'fill="{T.BAD}" text-anchor="middle" font-family="monospace">STOP</text>'
        )

    return f"<g>{side}{body}{brackets}{spindle}{led}{txt}{warn}</g>"


def floor_svg(snapshot: list[dict], subtitle: str = "", clock: str = "") -> str:
    rows: dict[str, list] = {}
    for m in snapshot:
        rows.setdefault(m["station"], []).append(m)

    n_stop = sum(1 for m in snapshot if m["should_stop"])
    avg = sum(m["health"] for m in snapshot) / max(1, len(snapshot))

    p = [_defs()]

    # 底板
    p.append(f'<rect width="{W}" height="{H}" fill="url(#floorGrad)" rx="8"/>')
    p.append(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" fill="none" '
             f'stroke="{T.LINE}" rx="8"/>')

    # 標題列
    p.append(f'<rect x="0" y="0" width="{W}" height="{HEAD_H}" fill="{T.PANEL_HI}" '
             f'opacity="0.75"/>')
    p.append(f'<line x1="0" y1="{HEAD_H}" x2="{W}" y2="{HEAD_H}" stroke="{T.ACCENT}" '
             f'stroke-width="1.4" opacity="0.7"/>')
    p.append(f'<rect x="0" y="0" width="4" height="{HEAD_H}" fill="{T.ACCENT}"/>')
    p.append(
        f'<text x="22" y="27" font-size="16" font-weight="800" fill="{T.TEXT}" '
        f'letter-spacing="2.5">FACTORY FLOOR · 廠房即時監控</text>'
        f'<text x="22" y="46" font-size="11" fill="{T.MUTED}">{subtitle}</text>'
    )
    # 右側統計。用兩組獨立的 text 而不是 tspan——
    # text-anchor="end" 搭配多個 tspan 時，各家瀏覽器的對齊行為不一致，
    # 實測會出現數字互相重疊。分開定位比較保險。
    stop_c = T.BAD if n_stop else T.OK
    p.append(
        f'<text x="{W-104}" y="24" font-size="9.5" fill="{T.MUTED}" text-anchor="middle" '
        f'letter-spacing="1.2">平均健康度</text>'
        f'<text x="{W-104}" y="48" font-size="20" font-weight="800" text-anchor="middle" '
        f'font-family="monospace" fill="{_health_color(avg)}">{avg:.0f}</text>'
        f'<line x1="{W-62}" y1="16" x2="{W-62}" y2="52" stroke="{T.LINE}" stroke-width="1"/>'
        f'<text x="{W-34}" y="24" font-size="9.5" fill="{T.MUTED}" text-anchor="middle" '
        f'letter-spacing="1.2">待停機</text>'
        f'<text x="{W-34}" y="48" font-size="20" font-weight="800" text-anchor="middle" '
        f'font-family="monospace" fill="{stop_c}">{n_stop}</text>'
    )
    if clock:
        p.append(f'<text x="{W/2}" y="38" font-size="13" fill="{T.ACCENT}" '
                 f'text-anchor="middle" font-family="monospace" letter-spacing="2">'
                 f'{clock}</text>')

    # 廠區網格
    gx, gy, gw, gh = 20, HEAD_H + 14, W - 40, H - HEAD_H - 56
    p.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="url(#grid)" '
             f'stroke="{T.LINE}" stroke-width="1" rx="4"/>')
    p.append(_corner_brackets(gx, gy, gw, gh, T.ACCENT, size=16, sw=1.6))

    # 輸送帶
    belt_y = (ROW_A_Y + MACH_H + ROW_B_Y) / 2
    p.append(f'<rect x="{LEFT-16}" y="{belt_y-8}" width="{W-2*LEFT+40}" height="16" '
             f'rx="2" fill="#0a1626" stroke="{T.LINE}"/>')
    for i in range(26):
        bx = LEFT - 6 + i * 38
        if bx < W - LEFT + 20:
            p.append(f'<line x1="{bx}" y1="{belt_y-6}" x2="{bx+7}" y2="{belt_y+6}" '
                     f'stroke="{T.LINE}" stroke-width="1.4"/>')
    p.append(f'<text x="{LEFT-16}" y="{belt_y-14}" font-size="9.5" fill="{T.MUTED}" '
             f'letter-spacing="1.5">CONVEYOR · 輸送帶</text>')

    # 兩條產線
    for line, y in (("A 線", ROW_A_Y), ("B 線", ROW_B_Y)):
        machines = rows.get(line, [])
        if not machines:
            continue
        p.append(f'<rect x="{LEFT-30}" y="{y-6}" width="3" height="{MACH_H+12}" '
                 f'fill="{T.ACCENT}" opacity="0.55"/>')
        p.append(
            f'<text x="{LEFT-38}" y="{y+MACH_H/2}" font-size="12" font-weight="700" '
            f'fill="{T.ACCENT}" text-anchor="middle" letter-spacing="2" '
            f'transform="rotate(-90 {LEFT-38} {y+MACH_H/2})">{line}</text>'
        )
        for i, m in enumerate(machines):
            p.append(_machine(LEFT + i * (MACH_W + COL_GAP), y, m))

    # 底部圖例
    ly = H - 20
    p.append(f'<line x1="20" y1="{ly-22}" x2="{W-20}" y2="{ly-22}" '
             f'stroke="{T.LINE}" stroke-width="1"/>')
    # 圖例的分數區間直接取自 ui_theme.BANDS，與 KPI 卡片、分數條說明同一份來源。
    # 之前這裡寫死「正常 100–70」，KPI 卡片自己另外分桶，兩邊數量對不起來。
    for i, band in enumerate(T.BANDS):
        lx = 26 + i * 172
        p.append(f'<rect x="{lx}" y="{ly-9}" width="10" height="10" fill="{band["color"]}"/>')
        p.append(f'<text x="{lx+16}" y="{ly}" font-size="10.5" fill="{T.MUTED}">'
                 f'{band["name"]} {band["range"]}</text>')
    p.append(
        f'<text x="{W-26}" y="{ly}" font-size="10" fill="{T.MUTED}" text-anchor="end">'
        f'閃爍＝建議立即停機　·　主軸轉速反映健康度　·　數值為綜合健康分數</text>'
    )

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;display:block">{"".join(p)}</svg>')
