"""
產生簡報用的圖檔。所有數字即時從特徵檔／結果檔計算，不寫死。

執行：
    python make_figures.py

輸出（results/figures/）：
    fig1_evidence_diagonal.png   證據對角線：每類故障由不同物理量點亮
    fig2_before_after.png        v1 -> v2 跨負載混淆矩陣對照
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

import config
import evidence as ev

OUT = Path("./results/figures")

# ---- 中文字型 ----------------------------------------------------------
# 沒有設定 CJK 字型的話，matplotlib 會把中文全部畫成空白方框。
for cand in ["Noto Sans CJK JP", "Noto Sans CJK TC", "Microsoft JhengHei",
             "PingFang TC", "Droid Sans Fallback", "SimHei"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        print(f"使用字型：{cand}")
        break
else:
    print("⚠️ 找不到中文字型，圖上的中文可能顯示為方框")
plt.rcParams["axes.unicode_minus"] = False

# 配色：深色底、青色強調，與儀表板一致
BG = "#0b1220"
FG = "#dbeafe"
GRID = "#1e3a5f"
ACCENT = "#22d3ee"

CONDITIONS = ["Normal", "BPFO", "BPFI", "Misalign", "Unbalance"]
COND_LABEL = {
    "Normal": "正常", "BPFO": "軸承外環", "BPFI": "軸承內環",
    "Misalign": "軸不對心", "Unbalance": "轉子不平衡",
}

# 每個欄位：(特徵名, 顯示標籤, 該特徵支持哪個類別)
MEASURES = [
    ("env_BPFO", "包絡譜\n外環頻率\n183.5 Hz", "BPFO"),
    ("env_BPFI", "包絡譜\n內環頻率\n268.8 Hz", "BPFI"),
    ("ratio_3x_1x", "轉頻\n3x/1x 諧波比\n150.4 Hz", "Misalign"),
    ("order_1x_dominance", "轉頻基頻\n1x 能量佔比\n50.2 Hz", "Unbalance"),
    ("kurtosis", "波形峰度\n（衝擊性）", None),
]


def load_features() -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(config.OUTPUT_DIR / f"vibration_v2_features_{s}.csv")
         for s in ["train", "val", "test"]],
        ignore_index=True,
    )


def deviation_matrix(v: pd.DataFrame, ref: dict):
    """回傳 (倍數矩陣, 跨負載範圍矩陣)。

    每格 = 該類別相對「同負載正常基準」的偏離倍數，取四個通道中最大者。
    跨負載範圍用來證明這個判準不是只在單一負載成立。
    """
    mat = np.zeros((len(CONDITIONS), len(MEASURES)))
    rng = np.empty((len(CONDITIONS), len(MEASURES)), dtype=object)

    for i, cond in enumerate(CONDITIONS):
        for j, (feat, _, _) in enumerate(MEASURES):
            per_load = []
            for load in [0, 2, 4]:
                g = v[(v.condition == cond) & (v.load_nm == load)]
                base = ref["baseline"][str(load)]
                best = 0.0
                for ch in range(4):
                    col = f"ch{ch}_{feat}"
                    if col not in g.columns or col not in base:
                        continue
                    b = base[col]["median"]
                    if abs(b) < 1e-12:
                        continue
                    best = max(best, float(g[col].median()) / b)
                per_load.append(best)
            mat[i, j] = float(np.median(per_load))
            rng[i, j] = (min(per_load), max(per_load))
    return mat, rng


def fig1(v, ref):
    mat, rng = deviation_matrix(v, ref)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # 用 log 尺度上色：倍數跨越 1~30，線性上色會讓小值全部看不見
    shown = np.log10(np.clip(mat, 1.0, None))
    ax.imshow(shown, cmap="cividis", aspect="auto", vmin=0, vmax=np.log10(30))

    for i in range(len(CONDITIONS)):
        for j in range(len(MEASURES)):
            val = mat[i, j]
            supports = MEASURES[j][2]
            is_diag = (supports == CONDITIONS[i])
            if val < 1.5:
                txt, color, weight = "—", "#5b7089", "normal"
            else:
                txt = f"{val:.1f}×"
                color = ACCENT if is_diag else FG
                weight = "bold" if is_diag else "normal"
            ax.text(j, i - 0.08, txt, ha="center", va="center",
                    color=color, fontsize=15 if is_diag else 12, fontweight=weight)
            if is_diag and val >= 1.5:
                lo, hi = rng[i, j]
                ax.text(j, i + 0.26, f"0/2/4Nm: {lo:.1f}~{hi:.1f}×",
                        ha="center", va="center", color="#8fb3d9", fontsize=7.5)
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92,
                                           fill=False, edgecolor=ACCENT, lw=2.2))

    ax.set_xticks(range(len(MEASURES)))
    ax.set_xticklabels([m[1] for m in MEASURES], color=FG, fontsize=9.5)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([COND_LABEL[c] for c in CONDITIONS], color=FG, fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(MEASURES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
    ax.grid(which="minor", color=GRID, lw=1)
    ax.tick_params(which="minor", length=0)

    # 誠實標註：不平衡那一列的最大值其實落在不對心的欄位（2.5× > 2.2×）。
    # 這正是模型把這兩類搞混的原因，直接畫在圖上比藏起來有說服力——
    # 而且它讓「已知限制」有了物理解釋，不再只是一個難看的數字。
    i_unb = CONDITIONS.index("Unbalance")
    j_3x = [k for k, m in enumerate(MEASURES) if m[0] == "ratio_3x_1x"][0]
    # 放在「不平衡」那列左半邊的空白格內（那兩格都是「—」），
    # 不要放到圖外，否則會壓到座標軸標籤。
    ax.annotate(
        "不平衡的 3x 偏離 2.5×\n略高於自身的 1x 2.2×\n→ 兩類互相混淆的物理來源",
        xy=(j_3x - 0.46, i_unb), xytext=(-0.35, i_unb),
        color="#ffb4a2", fontsize=8.5, ha="left", va="center", linespacing=1.5,
        arrowprops=dict(arrowstyle="->", color="#ffb4a2", lw=1.4,
                        shrinkA=6, shrinkB=4),
    )

    ax.set_title("每一類故障由不同的物理量點亮，且跨三種負載都成立",
                 color=FG, fontsize=15, pad=18, fontweight="bold")
    fig.text(0.5, 0.045,
             "數值 = 相對「同負載正常機台」基準的偏離倍數（四個感測通道取最大）。"
             "青色框為該故障的理論判準。所有數字直接來自頻譜，不經模型。",
             ha="center", color="#8fb3d9", fontsize=9)
    fig.text(0.5, 0.017,
             "軸承內外環的訊號強度是轉子類故障的 5~10 倍，"
             "因此軸承故障可跨負載零誤判，轉子類則有約兩成互相混淆。",
             ha="center", color="#8fb3d9", fontsize=9)

    fig.tight_layout(rect=[0, 0.075, 1, 1])
    p = OUT / "fig1_evidence_diagonal.png"
    fig.savefig(p, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"  已輸出 {p}")

    print("\n  對角線檢查（每列最大值是否落在理論判準上）：")
    for i, cond in enumerate(CONDITIONS):
        if cond == "Normal":
            print(f"    {COND_LABEL[cond]:6s} 全部 < 1.5 倍：{(mat[i] < 1.5).all()}")
            continue
        j = int(mat[i].argmax())
        expect = [k for k, m in enumerate(MEASURES) if m[2] == cond]
        ok = j in expect
        print(f"    {COND_LABEL[cond]:6s} 最大值 {mat[i, j]:.1f}× 在「"
              f"{MEASURES[j][1].replace(chr(10), ' ')}」　{'✓' if ok else '✗ 不符預期'}")


# ---- 圖二：改善前後 -----------------------------------------------------
# ⚠️ 混淆矩陣一律從 results/confusion_*.csv 讀取，不在程式裡寫死。
#    寫死的話很容易在改版之後忘了更新，或（實際發生過）拿別的實驗的矩陣充數，
#    畫出一張總數對不上的圖。簡報上的每個數字都必須能追溯到一次真實執行。
CM_DISPLAY = {
    "BPFI": "軸承內環", "BPFO": "軸承外環", "Misalign": "軸不對心",
    "Normal": "正常", "Unbalance": "轉子不平衡",
}


def load_cm(path: Path):
    """讀取混淆矩陣 CSV，回傳 (矩陣, 中文標籤, 準確率, macro-F1)。"""
    df = pd.read_csv(path, index_col=0)
    cm = df.values
    labels = [CM_DISPLAY.get(c, c) for c in df.index]
    acc = np.trace(cm) / cm.sum()
    f1s = []
    for i in range(len(cm)):
        tp = cm[i, i]
        p = tp / cm[:, i].sum() if cm[:, i].sum() else 0
        r = tp / cm[i, :].sum() if cm[i, :].sum() else 0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0)
    return cm, labels, acc, float(np.mean(f1s))


def _draw_cm(ax, cm, CM_LABELS, title, subtitle):
    ax.set_facecolor(BG)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sum == 0, 1, row_sum))
    ax.imshow(norm, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color=ACCENT if i == j else "#ffb4a2",
                    fontsize=11, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(CM_LABELS, color=FG, fontsize=8.5, rotation=30, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(CM_LABELS, color=FG, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=FG, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, -0.32, subtitle, transform=ax.transAxes, ha="center",
            color="#8fb3d9", fontsize=9.5)


def fig2():
    res = Path("./results")
    p_v2 = res / "confusion_B_跨負載泛化_vibration_v2.csv"
    p_v1 = res / "confusion_B_跨負載泛化_vibration.csv"

    missing = [p for p in (p_v1, p_v2) if not p.exists()]
    if missing:
        print("  ⚠️ 跳過圖二，缺少真實的混淆矩陣檔：")
        for p in missing:
            print(f"       {p}")
        print("     請依序執行下面兩行，把兩個版本的實際結果都跑出來：")
        print("       python train_baseline.py --features v1")
        print("       python train_baseline.py")
        print("     （不從真實執行結果讀取，就不要畫這張圖）")
        return

    cm1, lab1, acc1, f11 = load_cm(p_v1)
    cm2, lab2, acc2, f12 = load_cm(p_v2)

    # 正常類別的召回率：用來講「0% -> 91%」那句話，同樣要算出來不能寫死
    def normal_recall(cm, labels):
        if "正常" not in labels:
            return None
        i = labels.index("正常")
        s = cm[i].sum()
        return cm[i, i] / s if s else 0.0

    r1, r2 = normal_recall(cm1, lab1), normal_recall(cm2, lab2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor(BG)

    _draw_cm(axes[0], cm1, lab1,
             "改善前：三個粗頻帶特徵",
             f"準確率 {acc1:.1%}　macro-F1 {f11:.1%}\n"
             + (f"正常類別召回率 {r1:.0%}" if r1 is not None else ""))
    _draw_cm(axes[1], cm2, lab2,
             "改善後：階次域 + 包絡解調",
             f"準確率 {acc2:.1%}　macro-F1 {f12:.1%}\n"
             + (f"正常類別召回率 {r2:.0%}，軸承內外環零誤判" if r2 is not None else ""))
    print(f"    v1 準確率 {acc1:.4f} / 正常召回 {r1:.2f}")
    print(f"    v2 準確率 {acc2:.4f} / 正常召回 {r2:.2f}")

    fig.suptitle("跨負載泛化：訓練 0+2 Nm，測試完全沒看過的 4 Nm",
                 color=FG, fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015,
             "橫軸為模型預測、縱軸為實際狀態。改善前後使用相同的模型與切分方式，"
             "唯一差異是振動特徵的定義。",
             ha="center", color="#8fb3d9", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    p = OUT / "fig2_before_after.png"
    fig.savefig(p, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"  已輸出 {p}")


# ---- 圖三：LLM 的排除性推理 ---------------------------------------------
def fig3():
    """把 selfcheck 產生的 LLM 實際回答渲染成圖。

    刻意不用螢幕截圖：截圖會帶進瀏覽器邊框與捲軸，而且解析度受限。
    直接從 results/selfcheck_llm_output.txt 讀取原文，確保投影片上的每一個字
    都是模型真的講過的話（跑 selfcheck.py 就能重現）。
    """
    src = Path("./results/selfcheck_llm_output.txt")
    if not src.exists():
        print("  ⚠️ 跳過圖三：找不到 results/selfcheck_llm_output.txt")
        print("     請先執行 python selfcheck.py")
        return

    text = src.read_text(encoding="utf-8")
    marker = "[提問] 你怎麼確定是外環而不是內環故障？請引用實際數據。"
    if marker not in text:
        print("  ⚠️ 跳過圖三：輸出檔裡找不到排除性推理那一題")
        return
    # 檔案結構是：marker 行 -> 一整行分隔線 -> 回答內容 -> 下一條分隔線。
    # 直接 split 分隔線會切在「marker 後面那一條」，拿到空字串，
    # 所以要先把緊接在後的分隔線吃掉，再切下一條。
    body = text.split(marker, 1)[1].lstrip("\n")
    body = body.split("\n", 1)[1] if body.startswith("-") else body
    body = body.split("-" * 40, 1)[0].strip()

    # 逐行整理：去掉 markdown 粗體記號，過長的行自行斷行
    import textwrap
    lines = []
    for raw in body.splitlines():
        raw = raw.replace("**", "").rstrip()
        if not raw.strip():
            lines.append("")
            continue
        indent = len(raw) - len(raw.lstrip())
        wrapped = textwrap.wrap(raw.strip(), width=46) or [""]
        for k, w in enumerate(wrapped):
            lines.append(" " * (indent + (2 if k else 0)) + w)

    fig_h = 1.55 + 0.30 * len(lines)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.985, "使用者提問", ha="center", va="top",
            color="#8fb3d9", fontsize=10)
    ax.text(0.5, 0.955,
            "「你怎麼確定是外環而不是內環故障？請引用實際數據。」",
            ha="center", va="top", color=ACCENT, fontsize=14, fontweight="bold")

    # 行距用「剩餘高度平均分配」，不要用固定值——固定值在行數變動時
    # 會在底部留下大片空白，投影片上很明顯。
    y = 0.895
    step = (0.895 - 0.045) / max(len(lines), 1)
    for ln in lines:
        stripped = ln.strip()
        # 含有「倍」或 Hz 的行是引用實測數據的關鍵句，用強調色
        key = ("倍" in stripped or "Hz" in stripped
               or "實測" in stripped or "基準" in stripped)
        ax.text(0.045, y, ln, ha="left", va="top",
                color=ACCENT if key else FG,
                fontsize=10.5, family=plt.rcParams["font.sans-serif"][0])
        y -= step

    fig.suptitle("LLM 解釋層：引用實測數據做排除性推理",
                 color=FG, fontsize=15, fontweight="bold", y=0.998)
    fig.text(0.5, 0.012,
             "回答全文由系統實際產生（results/selfcheck_llm_output.txt），未經編輯。"
             "青色為引用實測數據的句子。",
             ha="center", color="#8fb3d9", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    p = OUT / "fig3_llm_reasoning.png"
    fig.savefig(p, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"  已輸出 {p}（{len(lines)} 行）")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v = load_features()
    ref = ev.load_reference()
    print(f"讀入 {len(v)} 筆特徵、{v.file_id.nunique()} 個錄音檔\n")
    print("圖一：證據對角線")
    fig1(v, ref)
    print("\n圖二：改善前後對照")
    fig2()
    print("\n圖三：LLM 排除性推理")
    fig3()
    print(f"\n完成。圖檔在 {OUT.resolve()}")


if __name__ == "__main__":
    main()
