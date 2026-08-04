"""快速測試 Gemini API Key 是否正常運作。"""
from google import genai

# 直接從 secrets 讀取 Key
import tomllib
with open(".streamlit/secrets.toml", "rb") as f:
    key = tomllib.load(f)["gemini"]["api_key"]

print(f"API Key: {key[:10]}...{key[-4:]}")
print()

client = genai.Client(api_key=key)

# 嘗試每個模型。清單與 llm_assistant.py 保持一致，
# 這樣測試通過的模型就是實際會用到的模型。
from llm_assistant import _MODEL_CANDIDATES

for model in _MODEL_CANDIDATES:
    print(f"測試 {model}...", end=" ")
    try:
        r = client.models.generate_content(
            model=model,
            contents="說 OK",
        )
        print(f"✅ 成功！回覆：{r.text.strip()}")
        break
    except Exception as e:
        err = str(e)
        if "limit: 0" in err:
            print("❌ 額度為 0")
        elif "429" in err:
            print("⚠️ 速率限制，需等待")
        elif "404" in err or "NOT_FOUND" in err:
            # 模型下架或不開放此帳號——這是 2026/08 遇過的狀況
            print("❌ 模型不存在或此帳號無權使用（可能已下架）")
        else:
            print(f"❌ {err[:100]}")
else:
    print()
    print("=" * 50)
    print("所有模型都無法使用。請嘗試以下步驟：")
    print()
    print("1. 前往 https://aistudio.google.com/")
    print("   確認你可以在網頁上正常跟 Gemini 對話")
    print()
    print("2. 前往 https://aistudio.google.com/app/apikey")
    print("   刪除舊的 Key，重新建一個")
    print()
    print("3. 如果還是不行，前往 Google Cloud Console：")
    print("   https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
    print("   點「啟用」來開啟 Generative Language API")
