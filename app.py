import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

api_key = st.secrets["OPENROUTER_API_KEY"]
def get_mistral_response(prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": ".أنت مساعد طبي محترف متخصص في الإسعافات الأولية. أجب دائمًا بنصائح إسعاف أولية واضحة ودقيقة وآمنة. اقرأ السؤال التالي بعناية وقدم الإجابة المناسبة"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 256    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ Error from API: {response.text}"
    

# Streamlit UI
st.title("🩺 Arabic First-Aid Chatbot")

api_key = st.text_input("Enter your OpenRouter API key:", type="password")
user_input = st.text_area("🗣️ Write your question in Arabic:")

if st.button("Ask"):
    if not api_key:
        st.warning("Please enter your OpenRouter API key.")
    elif not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing your question..."):
            answer = get_mistral_response(user_input, api_key)
            st.write("### 💬 Response:")
            st.write(answer)
