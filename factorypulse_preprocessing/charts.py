"""共用圖表元件。

為什麼不直接用 st.line_chart / st.bar_chart
------------------------------------------
Streamlit 內建圖表畫得快，但畫不出「這條軸是什麼、單位是什麼」——
監控畫面上一條沒有標註的曲線，看的人只能用猜的，簡報時一定被問。
這裡改用 Altair（Streamlit 本來就相依，不必額外安裝）重畫，
把座標軸標題與單位全部標清楚。

另外兩件內建圖表做不好的事，這裡一併處理：
  1. 類別型的長條圖：內建版把類別當成數值軸處理，軸標會變成 0/20/40…，
     而且中文或長特徵名會被截掉。改成「橫向長條圖」後名稱有整行可以放。
  2. X 軸起點：內建版直接吃 DataFrame 的索引，若索引沒重設會出現 -40 這種
     不存在的時間點。這裡一律用 0 起算的序號當 X。
"""

from __future__ import annotations

import altair as alt
import pandas as pd

import ui_theme as T

# 座標軸樣式：與戰情室主題一致（深底、青色格線、淺字）
_AXIS = dict(
    labelColor=T.MUTED,
    titleColor=T.TEXT,
    gridColor=T.LINE,
    domainColor=T.LINE,
    tickColor=T.LINE,
    titleFontSize=12,
    labelFontSize=11,
    titleFontWeight="bold",
)


def _dress(chart: alt.Chart) -> alt.Chart:
    return (
        chart
        .configure_view(strokeWidth=0)
        .configure_axis(**_AXIS)
        .configure_legend(labelColor=T.TEXT, titleColor=T.MUTED, orient="bottom")
    )


def line(df: pd.DataFrame, x_title: str, y_title: str,
         y_domain: tuple[float, float] | None = None,
         height: int = 280) -> alt.Chart:
    """多條時間序列的折線圖。

    df 的每一欄是一條線，X 軸一律用 0 起算的列序號（時間窗編號）。
    y_domain 給定時會固定 Y 軸範圍——異常機率這種本來就是 0~1 的量，
    固定範圍才看得出「貼在上限」與「只是波動」的差別。
    """
    long = (
        df.reset_index(drop=True)
        .rename_axis("_x")
        .reset_index()
        .melt("_x", var_name="數列", value_name="_y")
    )

    y_scale = alt.Scale(domain=list(y_domain)) if y_domain else alt.Scale()

    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.8, interpolate="linear")
        .encode(
            x=alt.X("_x:Q", title=x_title, axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("_y:Q", title=y_title, scale=y_scale),
            color=alt.Color(
                "數列:N", title=None,
                scale=alt.Scale(range=[T.ACCENT, T.WARN, T.OK, T.BAD]),
            ),
            tooltip=[
                alt.Tooltip("_x:Q", title=x_title),
                alt.Tooltip("數列:N", title="數列"),
                alt.Tooltip("_y:Q", title=y_title, format=".3f"),
            ],
        )
        .properties(height=height)
    )
    return _dress(chart)


def hbar(df: pd.DataFrame, cat_col: str, val_col: str,
         x_title: str, y_title: str,
         color: str = T.ACCENT, height: int = 280,
         val_format: str = ",.0f", show_value: bool = True,
         color_col: str | None = None) -> alt.Chart:
    """橫向長條圖。

    類別放 Y 軸的好處：中文工站名、`ch3_env_BPFO_sideband` 這種長特徵名
    有一整行可以放，不會被截成 `ch3_env_BPFO_s...`。

    color_col 給定時，該欄直接放色碼（通常是 T.tone_color(分數) 的結果），
    長條就會跟廠房圖上的機台同色——同一台機器在兩個地方不該是兩種顏色。
    """
    # 右側留 15% 空間給數值標籤，否則最長那根的標籤會被畫布邊緣切掉。
    vmax = float(pd.to_numeric(df[val_col], errors="coerce").max() or 0)
    x_scale = alt.Scale(domain=[0, vmax * 1.15]) if show_value and vmax > 0 else alt.Scale()

    enc = dict(
        x=alt.X(f"{val_col}:Q", title=x_title, scale=x_scale),
        # scale 的 paddingInner 讓長條之間留白：類別少的時候長條不會胖成一塊色塊
        # 排序寫成 EncodingSortField 而不是簡寫的 "-x"：加上顏色欄位之後，
        # 簡寫版會失效退回字母序（FP-01、FP-02…），長條圖就不再由大排到小了。
        y=alt.Y(f"{cat_col}:N", title=y_title,
                sort=alt.EncodingSortField(field=val_col, op="max", order="descending"),
                scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25)),
        tooltip=[
            alt.Tooltip(f"{cat_col}:N", title=y_title),
            alt.Tooltip(f"{val_col}:Q", title=x_title, format=val_format),
        ],
    )
    if color_col:
        # 色碼直接放在資料裡，用 identity scale 原樣輸出，不讓 Vega 自己配色
        palette = sorted(set(df[color_col].astype(str)))
        enc["color"] = alt.Color(f"{color_col}:N", legend=None,
                                 scale=alt.Scale(domain=palette, range=palette))

    bars = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=2).encode(**enc)

    chart = bars
    if show_value:
        labels = (
            alt.Chart(df)
            .mark_text(align="left", baseline="middle", dx=5,
                       color=T.TEXT, fontSize=11)
            .encode(x=enc["x"], y=enc["y"],
                    text=alt.Text(f"{val_col}:Q", format=val_format))
        )
        chart = bars + labels

    return _dress(chart.properties(height=height))
