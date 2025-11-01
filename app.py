import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

# 1️⃣ Load the classifier model
MODEL_NAME = "imaneumabderahmane/FA-Arabertv2-classifier-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ["LABEL_0", "LABEL_1"]  # LABEL_1 = first aid, LABEL_0 = not first aid

def classify_question(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[pred]

# 2️⃣ Define Mistral API call function
def get_mistral_response(prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "أنت مساعد طبي متخصص في الإسعافات الأولية. "
                    "قدم دائمًا نصائح واضحة، دقيقة وآمنة باللغة العربية الفصحى. "
                    "اقرأ السؤال التالي بعناية وقدم الإجابة المناسبة."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ Error from API: {response.text}"

# 3️⃣ Streamlit UI
st.title("🩺 Arabic First-Aid Chatbot")

api_key = st.text_input("🔑 Enter your OpenRouter API key:", type="password")
user_input = st.text_area("🗣️ Write your question in Arabic:")

if st.button("Ask"):
    if not api_key:
        st.warning("Please enter your OpenRouter API key.")
    elif not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing your question..."):
            category = classify_question(user_input)

            if category == "LABEL_1":
                st.success("✅ First-aid question detected. Getting advice...")
                with st.spinner("Contacting the AI assistant..."):
                    answer = get_mistral_response(user_input, api_key)
                    st.write("### 💬 Response:")
                    st.write(answer)
            else:
                st.error("❌ Sorry, I can only answer first-aid–related questions.")
