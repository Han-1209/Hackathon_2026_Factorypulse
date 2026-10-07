"""
FactoryPulse 知識庫 RAG（檢索增強生成）—— 檢索端。

流程：
    kb_docs/*.md ──切塊──> 58 個 chunk ──embedding──> RAG/rag_index.json
    使用者追問 ──embedding──> 與 chunk 算餘弦相似度 ──> 取前 k 段 ──> 交給 llm_assistant

設計決策（評審問到可以照講）：
1. 切塊單位 = Markdown 的「## 小節」。
   一段只講一件事（例如「軸承外圈故障｜需要工具」），且標題與第一行都帶故障名稱，
   單獨被檢索出來時 LLM 仍知道它屬於哪一種故障。
2. embedding 用 Gemini API，不在本機跑模型。
   本專案部署在 Streamlit Cloud，本機模型（sentence-transformers + torch）動輒 1~2 GB，
   會拖垮雲端記憶體；而 Gemini API Key 本來就有。
3. 不用向量資料庫（Chroma / FAISS）。
   只有 58 個 chunk，numpy 直接算餘弦相似度不到 1 毫秒；多一個資料庫只是多一個 demo 會壞的地方。
   之後若加入金豐機械維修紀錄、原廠手冊等上千段文件，再換成 Chroma 即可，介面不變。
4. 三層退路，確保 demo 不開天窗：
   (a) 有索引 + 有網路     → 語意檢索（semantic）
   (b) 沒網路 / 索引過期   → 關鍵字檢索（keyword，中文二字詞重疊）
   (c) 什麼都沒有         → 回傳空結果，LLM 照舊只用 system prompt 裡的知識庫作答
5. 已知故障時用 fault_code 過濾（metadata filter），不讓 A 故障的段落混進 B 故障的回答。

用法：
    python rag_store.py build                  # 建索引（需要 GEMINI_API_KEY）
    python rag_store.py search "要準備什麼工具"  # 測試檢索
    python rag_store.py search "對心要用什麼儀器" --fault Misalign
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent              # .../factorypulse_preprocessing/RAG
# 上一層是主程式資料夾（llm_assistant.py、knowledge_base.py 在那裡）。
# 從 RAG/ 底下直接執行本檔時，Python 預設找不到它們，所以把上一層加進搜尋路徑。
sys.path.insert(0, str(HERE.parent))
KB_DIR = HERE / "kb_docs"
INDEX_PATH = HERE / "rag_index.json"

# embedding 模型候選（依序嘗試）。跟 llm_assistant 的 LLM 一樣，Google 會改名或下架，
# 所以留備援。⚠️ 建索引與查詢必須用同一個模型，否則向量不在同一個空間、相似度沒有意義。
EMBED_MODELS = ["gemini-embedding-001", "text-embedding-004"]
EMBED_DIM = 768            # 縮小維度：索引檔小、計算快，58 段文字綽綽有餘

TOP_K = 3
# 兩道門檻，用途不同（依 rag_eval.py 2026-10-07 實測結果訂定）：
#
#   MIN_SCORE_*      交給 LLM 的門檻 —— 寬鬆。
#                    實測「正確段落最低 0.644」與「無解問題最高 0.638」只差 0.006，
#                    單靠門檻分不開，訂嚴了會把正確答案擋掉。
#                    混進來的無關段落由 system prompt 規則 11（無關就忽略）處理。
#
#   DISPLAY_MIN_*    顯示給使用者看「📚 參考段落」的門檻 —— 嚴格。
#                    使用者看到的來源必須真的相關，否則反而傷害可信度。
#                    實測無解問題最高 0.638，0.70 以上的 7 個全部是正確段落。
MIN_SCORE_SEMANTIC = 0.60
DISPLAY_MIN_SEMANTIC = 0.70
DISPLAY_MIN_KEYWORD = 0.50
# 只顯示「跟第一名差不多相關」的段落。
# 實測問「高風險要多快處理」：高風險 0.738、警告 0.717、健康 0.712，
# 後兩段也超過 0.70（風險等級文件彼此很像），但跟問題無關（差距 0.021、0.026）。
# 只列出與第一名分數差距在此範圍內的段落。⚠️ 目前只用這一題校準過。
DISPLAY_MARGIN = 0.015
MIN_SCORE_KEYWORD = 0.25
PREFER_BONUS = 0.05        # prefer_fault 的加分，只用來打破平手，不足以蓋過真正指名的故障


# ───────────────────────────── 切塊 ─────────────────────────────
@dataclass
class Chunk:
    id: str                 # 例：fault_BPFO.md#6
    source: str             # 檔名
    title: str              # 「軸承外圈故障｜需要工具」
    text: str               # 標題 + 內文（拿去做 embedding 與給 LLM 的就是這段）
    meta: dict = field(default_factory=dict)   # fault_code / doc_type / display ...


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """解析檔頭 YAML（只支援本專案用到的 key: value 與 [a, b] 清單，免裝 pyyaml）。"""
    if not raw.startswith("---"):
        return {}, raw
    _, head, body = raw.split("---", 2)
    meta = {}
    for line in head.strip().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        meta[k.strip()] = v
    return meta, body


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    """讀 kb_docs/ 下的 MD，依「## 」切塊。README.md 是給人看的說明，不入庫。"""
    chunks: list[Chunk] = []
    # ⚠️ 一定要用檔名字串排序。Windows 的 Path 排序不分大小寫、Linux 分大小寫，
    #    同一批檔案在兩邊排出來的順序不同（fault_Ball 與 fault_BPFI 會對調）。
    for path in sorted(kb_dir.glob("*.md"), key=lambda p: p.name):
        if path.name.lower() == "readme.md":
            continue
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        # 以行首「## 」切開；第一段是 # 大標與摘要，不當 chunk
        parts = re.split(r"(?m)^## ", body)
        for n, part in enumerate(parts[1:], start=1):
            title, _, content = part.partition("\n")
            title = title.strip()
            text = f"{title}\n{content.strip()}"
            chunks.append(Chunk(
                id=f"{path.name}#{n}", source=path.name, title=title, text=text,
                meta={**meta, "section": title.split("｜")[-1]},
            ))
    return chunks


def kb_hash(chunks: list[Chunk]) -> str:
    """知識庫內容指紋。MD 一改，指紋就變，代表索引過期需要重建。"""
    h = hashlib.sha256()
    # 依 id 排序後再算，指紋只跟「內容」有關，跟作業系統讀檔順序無關。
    # （索引在 Windows 建、網站在 Streamlit Cloud 的 Linux 上跑，兩邊必須算出同一個指紋）
    for c in sorted(chunks, key=lambda c: c.id):
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


# ─────────────────────────── embedding ───────────────────────────
def _get_client():
    from google import genai  # 延後 import：沒裝套件時，關鍵字檢索仍可用
    import llm_assistant      # 共用同一套 API Key 取得方式（secrets.toml → 環境變數）
    key = llm_assistant._get_api_key()
    if not key:
        # 從 RAG/ 底下執行時，Streamlit 只會在「目前資料夾」找 .streamlit/secrets.toml，
        # 找不到上一層那份。這裡直接去上一層讀。
        secrets = HERE.parent / ".streamlit" / "secrets.toml"
        if secrets.exists():
            try:
                import tomllib                     # Python 3.11+
                key = tomllib.loads(secrets.read_text(encoding="utf-8"))["gemini"]["api_key"]
            except Exception:
                key = None
    if not key or key.startswith("在這裡"):
        raise RuntimeError("未設定 Gemini API Key")
    return genai.Client(api_key=key)


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def embed(texts: list[str], task: str, model: str | None = None) -> tuple[np.ndarray, str]:
    """回傳 (已正規化的向量矩陣, 實際使用的模型名)。
    task："RETRIEVAL_DOCUMENT"（建索引）或 "RETRIEVAL_QUERY"（查詢）。"""
    from google.genai import types
    client = _get_client()
    models = [model] if model else EMBED_MODELS
    last_err = None
    for m in models:
        try:
            vecs = []
            for i in range(0, len(texts), 50):          # 分批，避免單次請求過大
                res = client.models.embed_content(
                    model=m, contents=texts[i:i + 50],
                    config=types.EmbedContentConfig(task_type=task,
                                                    output_dimensionality=EMBED_DIM),
                )
                vecs += [e.values for e in res.embeddings]
            return _normalize(np.asarray(vecs, dtype=np.float32)), m
        except Exception as e:      # 模型下架 / 改名 → 換下一個
            last_err = e
    raise RuntimeError(f"所有 embedding 模型都失敗：{last_err}")


# ─────────────────────────── 建索引 ───────────────────────────
def build_index(path: Path = INDEX_PATH) -> dict:
    chunks = load_chunks()
    vecs, model = embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    data = {
        "embed_model": model, "dim": int(vecs.shape[1]), "kb_hash": kb_hash(chunks),
        "chunks": [{"id": c.id, "vec": [round(float(x), 6) for x in v]}
                   for c, v in zip(chunks, vecs)],
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


# ─────────────────────────── 關鍵字退路 ───────────────────────────
def _terms(s: str) -> set[str]:
    """中文取相鄰二字（bigram），英數取整個單字。不需要斷詞套件。"""
    s = s.lower()
    words = set(re.findall(r"[a-z0-9]+", s))
    cjk = re.findall(r"[一-鿿]", s)
    return words | {a + b for a, b in zip(cjk, cjk[1:])}


def _keyword_scores(query: str, chunks: list[Chunk]) -> np.ndarray:
    """關鍵字分數 = 問題裡「知識庫有出現過的詞」被這段命中的比例（以 IDF 加權）。
    - 知識庫完全沒出現的詞（例如「什麼」「今天」）直接忽略，不拉低分數；
    - 越少段落用到的詞越有鑑別力（IDF 越高）；
    - 命中小節標題的詞算兩倍（問「工具」時優先回「｜需要工具」那段）。"""
    q = _terms(query)
    docs = [(_terms(c.title), _terms(c.text)) for c in chunks]
    n = len(chunks)
    df = {t: sum(t in body for _, body in docs) for t in q}
    q = {t for t in q if df[t] > 0}
    if not q:
        return np.zeros(n)
    idf = {t: math.log(1 + n / df[t]) for t in q}
    full = sum(2 * idf[t] for t in q)
    out = []
    for title, body in docs:
        hit = sum(idf[t] * (2 if t in title else 1) for t in q if t in body)
        out.append(hit / full)
    return np.asarray(out)


# ─────────────────────────── 檢索 ───────────────────────────
class KnowledgeRetriever:
    """
    retriever = KnowledgeRetriever()
    hits = retriever.search("要準備什麼工具？", fault_code="BPFO")
    for h in hits: print(h["title"], h["score"])
    """

    def __init__(self, index_path: Path = INDEX_PATH):
        self.chunks = load_chunks()
        self._by_id = {c.id: i for i, c in enumerate(self.chunks)}
        self.matrix: np.ndarray | None = None
        self.embed_model: str | None = None
        self.status = "keyword"           # semantic / keyword
        self.status_note = ""
        self._load_index(index_path)

    def _load_index(self, path: Path) -> None:
        if not path.exists():
            self.status_note = "尚未建立語意索引（執行 python rag_store.py build），改用關鍵字檢索"
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kb_hash") != kb_hash(self.chunks):
            # 知識庫改了但索引沒重建 —— 舊向量對不上新內容，寧可退回關鍵字也不要用錯的
            self.status_note = "知識庫已更新但索引未重建，改用關鍵字檢索"
            return
        vecs = {d["id"]: d["vec"] for d in data["chunks"]}
        self.matrix = np.asarray([vecs[c.id] for c in self.chunks], dtype=np.float32)
        self.embed_model = data["embed_model"]
        self.status = "semantic"

    def search(self, query: str, fault_code: str | None = None,
               exclude_fault: str | None = None, prefer_fault: str | None = None,
               k: int = TOP_K) -> list[dict]:
        """
        fault_code     只在這個故障（與風險等級文件）裡找 —— 已知診斷時用
        exclude_fault  排除這個故障
        prefer_fault   這個故障的段落小幅加分。問題沒指明故障時（例如「要準備什麼工具？」），
                       同分的情況下優先回本次研判的故障，而不是隨機挑一個別的故障。
        """
        mask = np.ones(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            code = c.meta.get("fault_code")
            if fault_code and code not in (fault_code, None):
                mask[i] = False
            if exclude_fault and code == exclude_fault:
                mask[i] = False
        if not mask.any():
            return []

        mode = self.status
        scores = None
        if self.matrix is not None:
            try:
                q, _ = embed([query], "RETRIEVAL_QUERY", model=self.embed_model)
                scores = self.matrix @ q[0]
            except Exception as e:      # 沒網路、額度用完 → 關鍵字
                mode = "keyword"
                self.status_note = f"語意檢索暫時無法使用（{type(e).__name__}），改用關鍵字檢索"
        if scores is None:
            scores = _keyword_scores(query, self.chunks)
            mode = "keyword"
        threshold = MIN_SCORE_SEMANTIC if mode == "semantic" else MIN_SCORE_KEYWORD

        # 兩種分數分開用：
        #   raw   = 真正的相似度 → 拿來跟門檻比（判斷「知識庫有沒有答案」）
        #   rank  = raw + 本次故障加分 → 只拿來排名次（同分時優先本次故障）
        # 加分若也算進門檻，會讓本次故障的無關段落「混過」門檻。
        raw = np.where(mask, scores, -1.0)
        rank = raw.copy()
        if prefer_fault:
            rank += np.array([PREFER_BONUS if c.meta.get("fault_code") == prefer_fault else 0.0
                              for c in self.chunks])
        hits = []
        for i in np.argsort(-rank)[:k]:
            if raw[i] < threshold:
                continue
            c = self.chunks[i]
            hits.append({**c.meta, "id": c.id, "source": c.source, "title": c.title,
                         "text": c.text, "score": round(float(raw[i]), 3),
                         "mode": mode})
        return hits


def format_for_llm(hits: list[dict]) -> str:
    """把檢索結果排成附在使用者訊息後面的參考資料區塊。"""
    if not hits:
        return ""
    blocks = [f"［參考 {n}｜來源：{h['source']}］\n{h['text']}" for n, h in enumerate(hits, 1)]
    return ("\n\n【檢索參考資料】（依使用者問題從知識庫檢索，可能屬於其他故障；"
            "引用時須說明該段屬於哪一種故障）\n" + "\n\n".join(blocks))


def format_sources(hits: list[dict], trusted: bool = False) -> str:
    """附在回答最後、給使用者看的來源清單。由程式產生，不經 LLM，所以不會被編造。

    trusted=True：這些段落是 LLM 回報「實際引用」的，直接全部列出、不再用分數過濾。
    trusted=False：沒有引用回報時的退路，只列分數夠高、且與第一名差不多的段落。
    """
    if not hits:
        return ""
    if trusted:
        return "\n\n---\n📚 參考的知識庫段落：" + "；".join(
            f"{h['title']}（{h['source']}）" for h in hits)
    best = max(h["score"] for h in hits)
    shown = [h for h in hits
             if h["score"] >= (DISPLAY_MIN_SEMANTIC if h.get("mode") == "semantic"
                               else DISPLAY_MIN_KEYWORD)
             and h["score"] >= best - DISPLAY_MARGIN]
    if not shown:
        return ""
    return "\n\n---\n📚 參考的知識庫段落：" + "；".join(
        f"{h['title']}（{h['source']}）" for h in shown)


# ─────────────────────────── 命令列 ───────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FactoryPulse 知識庫 RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="建立語意索引")
    s = sub.add_parser("search", help="測試檢索")
    s.add_argument("query")
    s.add_argument("--fault", default=None, help="只在此故障代碼內檢索，例如 BPFO")
    s.add_argument("--exclude", default=None, help="排除此故障代碼")
    s.add_argument("--prefer", default=None, help="模擬本次研判故障（同分時優先）")
    args = ap.parse_args()

    if args.cmd == "build":
        d = build_index()
        print(f"已建立索引：{len(d['chunks'])} 個 chunk，模型 {d['embed_model']}，"
              f"維度 {d['dim']} → {INDEX_PATH}")
    else:
        r = KnowledgeRetriever()
        print(f"檢索模式：{r.status}  {r.status_note}")
        hits = r.search(args.query, fault_code=args.fault, exclude_fault=args.exclude,
                         prefer_fault=args.prefer)
        if not hits:
            print("（沒有超過門檻的段落 → 系統會回答『知識庫中沒有這項資訊』）")
        for h in hits:
            print(f"  {h['score']:.3f}  [{h['mode']}]  {h['title']}  ({h['source']})")
