"""
FactoryPulse LLM 維修助手

使用 Google Gemini API，把結構化診斷結果轉成自然語言維修建議，
並支援多輪對話讓使用者追問。

設計原則（與 knowledge_base.py / rule_engine.py 一致）：
  - LLM 只能根據診斷結果和知識庫內容回答，不得自行編造故障原因或維修步驟。
  - 所有診斷判斷來自 ML 模型 + 規則引擎，LLM 只負責把結構化資料轉成人看得懂的文字。
  - 嚴格限制話題：只回答 CNC 設備維修相關問題，拒絕無關話題。

需求：
  pip install google-genai

API Key 設定方式（兩者擇一）：
  1. Streamlit secrets:  .streamlit/secrets.toml → [gemini] api_key = "..."
  2. 環境變數:           set GEMINI_API_KEY=...
"""

from __future__ import annotations

import json
import os
import time
from typing import Generator

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

import knowledge_base as kb
import evidence as ev

# 模型優先順序：依序嘗試，遇到額度不足或模型下架自動切換下一個。
#
# ⚠️ 2026/08 更新過一次。原本用的三個模型都已經不能用了：
#     gemini-2.0-flash        已關閉（shut down）
#     gemini-2.0-flash-lite   已關閉
#     gemini-2.5-flash        不再開放新使用者（回 404 NOT_FOUND）
#
# 這裡的排序考量：本專案的 LLM 只做一件事——把規則引擎已經算好的
# 結構化診斷「翻譯」成自然語言，不需要推理能力，所以優先選最快最便宜的
# flash-lite；真的不可用再往上退到能力較強的版本。
# 最後放 latest 別名當保險：即使上面的具體版本哪天又下架，
# 別名仍會指向當時可用的最新 flash 模型，demo 當天不會整個掛掉。
_MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",   # 最快、最便宜，這個任務綽綽有餘
    "gemini-3.6-flash",        # 能力較強的備援
    "gemini-2.5-flash-lite",   # 舊世代備援（部分帳號仍可用）
    "gemini-flash-latest",     # 別名保險，永遠指向可用的最新 flash
]

# 重試設定
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 12  # 秒（Google 建議 10 秒後重試）

# ── 話題邊界關鍵字（中文 + 英文），用來初篩明顯的偏題問題 ──────────────────
_OFF_TOPIC_HINTS = [
    "天氣", "股市", "股票", "政治", "美食", "旅遊", "遊戲", "電影", "音樂",
    "寫詩", "作文", "程式設計", "python", "javascript", "food", "weather",
    "sports", "movie", "music", "joke", "笑話",
]


def _is_genai_ready() -> bool:
    return _GENAI_AVAILABLE


def _get_api_key() -> str | None:
    """依優先順序取 API Key：streamlit secrets → 環境變數。"""
    # 先試 streamlit secrets
    try:
        import streamlit as st
        return st.secrets["gemini"]["api_key"]
    except Exception:
        pass
    # 再試環境變數
    return os.environ.get("GEMINI_API_KEY")


def _build_system_prompt(diag: dict) -> str:
    """
    根據診斷結果組裝 system prompt。
    把知識庫內容、規則引擎結果注入到 system prompt，
    讓 LLM 有所本地回答而不是憑空編造。
    """
    fault_code = diag.get("fault_type") or diag.get("probable_fault", "Normal")
    fault_info = kb.get_fault_info(fault_code)

    # 把診斷結果整理成易讀格式注入 prompt
    risk_level = diag.get("risk_level_display", diag.get("risk_level", "未知"))
    fault_display_name = (
        diag.get("fault_label")
        or diag.get("probable_fault_display")
        or fault_info.get("display", fault_code)
    )
    anomaly_prob = diag.get("anomaly_prob", diag.get("risk_score", 0))
    if isinstance(anomaly_prob, float) and anomaly_prob <= 1.0:
        anomaly_str = f"{anomaly_prob:.0%}"
    else:
        anomaly_str = str(anomaly_prob)

    confidence = diag.get("confidence", None)
    confidence_str = f"{confidence:.0%}" if confidence is not None else "未知"

    # 信心度區塊。沒有現成的判定就當場算一次，確保 LLM 拿到的等級
    # 與畫面上顯示的完全一致（同一份 confidence.assess 的輸出）。
    import confidence as _cf
    _verdict = diag.get("confidence_verdict")
    if not _verdict:
        try:
            _verdict = _cf.from_diagnosis(diag, diag.get("evidence"))
        except Exception:
            _verdict = None
    confidence_block = (_cf.format_for_llm(_verdict) if _verdict
                        else "  （本次未提供信心度判定）")

    should_stop = diag.get("should_stop", False)
    action_window = diag.get("action_window", "—")
    triggered_rules = diag.get("triggered_rules", [])
    causes = fault_info.get("causes") or diag.get("causes", [])
    checks = fault_info.get("checks") or diag.get("checks", [])
    actions_list = fault_info.get("actions") or diag.get("actions", [])
    description = fault_info.get("description", "")

    # ── 量測證據 ────────────────────────────────────────────────────────
    # 這是「可解釋 AI」與「查表產生說明」的分界線。
    # 沒有這一段，LLM 只能根據 fault_type 去念知識庫裡的通論；
    # 有了這一段，它講的每個數字都來自這台設備自己的頻譜，可以被驗證。
    # diag["evidence_features"] 由 diagnose_pipeline 帶入（該設備的特徵中位數）。
    evidence_list = diag.get("evidence") or []
    if not evidence_list and diag.get("evidence_features"):
        try:
            evidence_list = ev.extract(diag["evidence_features"],
                                       load_nm=diag.get("load_nm", 0))
        except FileNotFoundError:
            evidence_list = []
    evidence_text = ev.format_for_llm(evidence_list)

    rules_text = "\n".join(f"  - {r}" for r in triggered_rules) if triggered_rules else "  （無觸發規則）"
    causes_text = "\n".join(f"  - {c}" for c in causes) if causes else "  （無）"
    checks_text = "\n".join(f"  - {c}" for c in checks) if checks else "  （無）"
    actions_text = "\n".join(f"  - {a}" for a in actions_list) if actions_list else "  （無）"

    # 工具 / 備品 / 安全 / 工時 —— 這四項是現場工程師最常追問的。
    # 原本知識庫沒有這些欄位，LLM 只能回「請參考原廠手冊」，
    # 對 demo 來說是很弱的回答（而且它還是會忍不住自己舉例，反而有幻覺風險）。
    def _bullets(items):
        return "\n".join(f"  - {x}" for x in items) if items else "  （知識庫未收錄）"

    tools_text = _bullets(fault_info.get("tools") or diag.get("tools", []))
    parts_text = _bullets(fault_info.get("parts") or diag.get("parts", []))
    safety_text = _bullets(fault_info.get("safety") or diag.get("safety", []))
    effort_text = fault_info.get("effort") or "（知識庫未收錄）"
    signature_text = fault_info.get("signature") or "（知識庫未收錄）"

    return f"""你是 FactoryPulse 的 CNC 設備維修助手，專門協助維修工程師解讀感測診斷結果，\
並提供具體的維修建議。

【嚴格限制 - 必須遵守】
1. 你只能根據以下提供的【診斷結果】和【知識庫】內容回答問題。
2. 不可以自行編造故障原因、維修步驟、零件型號或任何診斷數據。
3. 如果使用者問的問題與 CNC 設備維修、機械故障診斷完全無關，\
請禮貌地拒絕，說明你只負責設備維修相關問題。
4. 如果問題超出目前診斷資料的範圍（例如詢問你沒有資料的感測器），\
請誠實說明資料不足，並建議進一步量測。
5. 所有的診斷判斷都來自 ML 模型和規則引擎，你不可以推翻或修改這些判斷。
6. 用繁體中文回答，語氣專業但清晰易懂，適合現場維修工程師閱讀。
7. 解釋「為什麼判斷成這個故障」時，必須引用下面【量測證據】裡的實際數字與頻率，\
不可以只講一般性的故障原理。證據已依偏離正常的倍數排序，請以最前面幾項為主要依據。
8. 【量測證據】沒有列出的現象，一律不得聲稱有觀察到。例如證據裡沒有提到邊帶，\
就不可以說「發現邊帶」；沒有提到溫度，就不可以說「溫度上升」。
9. 講到信心度時只能引用【信心度】區塊的等級與四個條件，不可以自己換算成百分比。\
「逐窗一致率」是模型在不同時間窗給出相同答案的比例，**不是**準確率也不是信心度，\
不可以把它講成「可信度」「準確率」或「信心水準」。\
若信心度等級是「中」或「低」，回答中必須明確建議複檢，不可以只講結論。
10. 不要在回答裡使用 Markdown 的表格與程式碼區塊；粗體與清單可以用。

【當前診斷結果】
- 研判故障：{fault_display_name}
- 風險等級：{risk_level}
- 異常機率：{anomaly_str}
- 逐窗一致率：{confidence_str}（僅供參考，不是準確率也不是信心度）
- 建議停機：{"是，請盡快安排" if should_stop else "否，可繼續運轉但需監測"}
- 建議處理時限：{action_window}

【信心度】（由四條可檢核條件判定，不是模型輸出的機率）
{confidence_block}

【量測證據】（本台設備實際量到的數值，與「同負載下正常機台」基準的比較）
{evidence_text}

【觸發的判斷規則】
{rules_text}

【知識庫 - 故障描述】
{description}

【知識庫 - 本系統的訊號判準】
{signature_text}

【知識庫 - 可能原因】
{causes_text}

【知識庫 - 建議檢查項目】
{checks_text}

【知識庫 - 建議行動】
{actions_text}

【知識庫 - 需要的工具】
{tools_text}

【知識庫 - 可能需要的備品】
{parts_text}

【知識庫 - 安全注意事項】
{safety_text}

【知識庫 - 預估工時】
{effort_text}

【你的回答格式建議】
- 摘要類問題：先說結論，再分點說明原因和行動
- 追問類問題：直接回答，不需重複已經說過的診斷結果
- 工具 / 備品 / 工時類問題：直接引用上面對應的知識庫欄位作答。
  只有在該欄位標示「知識庫未收錄」時，才說明資料不足並建議查原廠手冊——
  這種情況下也不要自己舉例補充，寧可少講也不要編。
- 涉及拆裝作業時，主動提醒【安全注意事項】裡的相關項目。
"""


class LLMAssistant:
    """
    LLM 維修助手。

    使用範例：
        assistant = LLMAssistant(diagnosis_result)
        summary = assistant.generate_summary()
        response = assistant.chat("要準備什麼工具？")
    """

    def __init__(self, diagnosis_result: dict):
        self._diag = diagnosis_result
        self._system_prompt = _build_system_prompt(diagnosis_result)
        self._history: list[dict] = []   # [{"role": "user"/"model", "parts": [str]}]
        self._client = None
        self._last_error: str | None = None

    def _evidence(self) -> list[dict]:
        e = self._diag.get("evidence") or []
        if not e and self._diag.get("evidence_features"):
            try:
                e = ev.extract(self._diag["evidence_features"],
                               load_nm=self._diag.get("load_nm", 0))
            except FileNotFoundError:
                e = []
        return e

    @property
    def last_error(self) -> str | None:
        """最近一次 LLM 呼叫失敗的原因。UI 可以顯示這個讓使用者知道
        現在看到的是離線摘要而不是 LLM 生成的內容。"""
        return self._last_error

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "找不到 google-genai 套件。\n"
                "請執行：pip install google-genai"
            )
        api_key = _get_api_key()
        if not api_key or api_key.startswith("在這裡"):
            raise ValueError(
                "尚未設定 Gemini API Key。\n"
                "請開啟 .streamlit/secrets.toml，\n"
                "把 api_key 的值換成你的實際 Key。\n\n"
                "取得 Key：https://aistudio.google.com/app/apikey"
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def generate_summary(self, allow_fallback: bool = True) -> str:
        """
        根據當前診斷結果，自動生成一份維修建議摘要（非對話式）。
        不會加入對話歷史，避免干擾後續追問的脈絡。

        allow_fallback=True（預設）時，若 LLM 因為任何原因不可用
        （沒網路、API Key 未設、額度用完、模型下架…），不會拋出例外，
        而是改回傳 evidence.fallback_summary() 產生的離線摘要。

        ⚠️ 這對 demo 是必要的。原本的寫法在 API 失敗時會直接 raise，
        畫面會開天窗——而 demo 當天最可能出問題的就是網路和 API 額度。
        離線摘要雖然生硬，但每個數字都是真的，該講的都有講到。
        """
        prompt = (
            "請根據以上診斷結果，生成一份給維修工程師看的摘要報告。\n"
            "格式：\n"
            "1. 一句話說明目前的狀況和緊急程度\n"
            "2. 最可能的原因（列點）\n"
            "3. 建議立即採取的行動（列點）\n"
            "4. 後續追蹤建議\n"
            "語氣要像資深工程師在交班時的口頭說明，精確但不艱澀。"
        )
        try:
            return self._call_llm(prompt, add_to_history=False)
        except Exception as e:
            self._last_error = str(e)
            if not allow_fallback:
                raise
            return ev.fallback_summary(self._diag, self._evidence())

    def chat(self, user_message: str) -> str:
        """
        多輪對話：使用者輸入問題，LLM 根據診斷上下文和對話歷史回答。
        """
        # 粗略偵測明顯偏題（減少 API 呼叫費用）
        msg_lower = user_message.lower()
        if any(hint in msg_lower for hint in _OFF_TOPIC_HINTS):
            reply = (
                "這個問題超出我的職責範圍。我只負責協助解讀設備診斷結果並提供維修建議。\n"
                "如果你有關於目前診斷結果的問題，歡迎繼續詢問！"
            )
            self._history.append({"role": "user", "parts": [user_message]})
            self._history.append({"role": "model", "parts": [reply]})
            return reply

        try:
            return self._call_llm(user_message, add_to_history=True)
        except Exception as e:
            # 追問失敗時不能只丟一句「錯誤」，至少把手上的證據交出去，
            # 讓使用者還是能看到這台設備的量測數據。
            self._last_error = str(e)
            return (
                "（LLM 服務目前無法連線，以下為離線的診斷資料）\n\n"
                + ev.fallback_summary(self._diag, self._evidence())
            )

    def _call_llm(self, message: str, add_to_history: bool) -> str:
        """呼叫 Gemini API，回傳文字回應。

        遇到 429 額度耗盡時會自動：
          1. 等待後重試（最多 _MAX_RETRIES 次）
          2. 如果同一個模型一直 429，切換到備用模型
        """
        client = self._get_client()

        # 組裝對話歷史（若有）
        contents = []
        for turn in self._history:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part(text=p) for p in turn["parts"]],
                )
            )
        # 加上當前訊息
        contents.append(
            types.Content(role="user", parts=[types.Part(text=message)])
        )

        config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=0.3,
            max_output_tokens=1024,
        )

        last_error = None
        for model_name in _MODEL_CANDIDATES:
            for attempt in range(_MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    reply_text = response.text or "（LLM 回傳空回應，請重試）"

                    if add_to_history:
                        self._history.append({"role": "user", "parts": [message]})
                        self._history.append({"role": "model", "parts": [reply_text]})

                    return reply_text

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        # 額度不足：如果 limit: 0，直接換模型不用等
                        if "limit: 0" in err_str:
                            break  # 這個模型沒額度，跳到下一個模型
                        # 否則等一下再試
                        delay = _RETRY_BASE_DELAY * (attempt + 1)
                        time.sleep(delay)
                    else:
                        # 非 429 錯誤，直接拋出
                        raise

        # 所有模型都試過了還是失敗
        raise last_error or RuntimeError("所有模型都無法回應")


    def clear_history(self):
        """清除對話歷史（例如換了一份診斷結果時）。"""
        self._history = []

    @property
    def is_ready(self) -> bool:
        """是否可以使用（套件存在 + API Key 已設定）。"""
        if not _GENAI_AVAILABLE:
            return False
        api_key = _get_api_key()
        return bool(api_key and not api_key.startswith("在這裡"))


def check_availability() -> tuple[bool, str]:
    """
    檢查 LLM 助手是否可用，回傳 (可用, 說明訊息)。
    給 Streamlit UI 在啟動時顯示狀態用。
    """
    if not _GENAI_AVAILABLE:
        return False, (
            "⚠️ 尚未安裝 google-genai 套件。\n"
            "請在終端機執行：`pip install google-genai`，然後重啟 Streamlit。"
        )
    api_key = _get_api_key()
    if not api_key:
        return False, (
            "⚠️ 找不到 Gemini API Key。\n"
            "請開啟 `.streamlit/secrets.toml`，填入你的 API Key。\n"
            "取得 Key：https://aistudio.google.com/app/apikey"
        )
    if api_key.startswith("在這裡"):
        return False, (
            "⚠️ 請先設定 Gemini API Key。\n"
            "開啟 `.streamlit/secrets.toml`，把預設文字換成你的實際 Key。"
        )
    return True, "✅ Gemini API 已就緒"
