from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 環境変数からOpenAIのAPIキーを取得
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLMモデルの初期化
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# 専門家（システムメッセージ）の定義
EXPERTS = {
    "健康アドバイザー": "あなたは健康に関するアドバイザーです。安全なアドバイスを提供してください。",
    "IT専門家": "あなたはWeb開発の専門家です。専門的な知識に基づいて、技術的な問題に関する詳細な回答を提供してください。",
    "料理専門家": "あなたは料理の専門家です。レシピや料理のコツ、食材に関するアドバイスを提供してください。"
}

# --- アプリケーションの概要と操作方法 ---
st.title("LangChainとStreamlitによるLLM専門家チャットアプリ")
st.markdown("""
このアプリは、LangChainとStreamlitを使用して、AI専門家との対話をシミュレートします。
あなたの質問内容と、回答してほしい専門家の種類を選択して、対話をお楽しみください。

### 操作方法
1.  **専門家の選択:** ラジオボタンから、質問したい専門家を選択してください。
2.  **テキストの入力:** テキストボックスに質問内容を入力してください。
3.  **送信:** 「送信」ボタンをクリックすると、選択した専門家の視点からAIが回答を生成します。
""")

# --- メイン画面にラジオボタンと入力フォームを配置 ---
selected_expert = st.radio(
    "専門家を選んでください:",
    list(EXPERTS.keys())
)

# 選択された専門家に応じたシステムメッセージを取得
system_message = EXPERTS[selected_expert]

user_input = st.text_input(label="質問を入力してください:")

# LLMとの対話を行う関数
def get_llm_response(text_input, expert_type):
    """
    ユーザーの入力と専門家の種類を引数として受け取り、LLMの回答を返す関数
    """
    try:
        # LangChainのSystemMessageとHumanMessageを使用してメッセージのリストを作成
        messages = [
            SystemMessage(content=expert_type),
            HumanMessage(content=text_input)
        ]
        # メッセージのリストを直接LLMに渡して回答を取得
        response = llm.invoke(messages)
        return response.content, None
    
    except Exception as e:
        return None, str(e)

# 送信ボタンが押された時の処理
if st.button("送信"):
    if user_input:
        response, error = get_llm_response(user_input, system_message)
        st.markdown("---")
        st.subheader(f"回答 ({selected_expert}より):")
        if error:
            st.error(f"エラー内容: {error}")
        else:
            st.write(response)
    else:
        st.error("質問内容を入力してください。")

