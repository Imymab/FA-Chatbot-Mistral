import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import torch



#classification
MODEL_NAME = "imaneumabderahmane/FA-Arabertv2-classifier-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
LABELS = ["LABEL_0", "LABEL_1"]



def classify_question(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[pred]

# =======================
# LLM first-aid check
# =======================
def llm_first_aid_check(query, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""
هل السؤال التالي يندرج ضمن الإسعافات الأولية؟ أجب فقط بـ "نعم" أو "لا":
{query}
"""
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "أنت مساعد ذكي عربي يساعد في تصنيف الأسئلة الطبية."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 10
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return "نعم" in answer
    else:
        return False

#chatbot response
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
        
  {
    "role": "system",
    "content": ".أنت مساعد طبي محترف متخصص في الإسعافات الأولية. أجب دائمًا بنصائح إسعاف أولية واضحة ودقيقة وآمنة. اقرأ السؤال التالي بعناية وقدم الإجابة المناسبة"
  },

  {
    "role": "user",
    "content": "أصابتني ضربة على عظمة حاجبي الأيسر وحدوث انتفاخ بسيط. وضعت كمادات باردة. ماذا ينبغي أن أفعل أيضًا؟"
  },
  {
    "role": "assistant",
    "content": "استمر بوضع كمادات باردة لفترات قصيرة (10–15 دقيقة كل ساعة) لتخفيف الورم. ارفع رأسك عند الاستلقاء لتقليل التورم، وتجنّب الضغط على المنطقة. راقب أي تغيّر في الرؤية، فقدان للوعي، أو تقيؤ — إذا ظهرت أي من هذه الأعراض فراجع الطوارئ فورًا."
  },

  {
    "role": "user",
    "content": "حدث نزيف داخل المخ ثم تلا ذلك ضعف في الساق اليسرى وخرف وهلوسة. هل هذا زهايمر أم شيء آخر؟"
  },
  {
    "role": "assistant",
    "content": "هذه أعراض خطيرة تتطلب تقييمًا عصبيًا عاجلًا. حالات النزيف والجلطة بالمخ قد تسبب ضعفًا موضعيًا واضطرابًا إدراكيًا؛ لذلك يلزم تصوير مقطعي أو رنين مغناطيسي وتقييم من أخصائي أعصاب لتحديد السبب وخطة العلاج. تواصل مع مقدم الرعاية الطارئة أو قسم الأعصاب فورًا."
  },

  {
    "role": "user",
    "content": "كيف أتعامل مع شخص يتعرض لنوبة صرع؟"
  },
  {
    "role": "assistant",
    "content": "أبعد حوله الأشياء الحادة وحافظ على الهدوء. ضع شيئًا ناعمًا تحت رأسه، لا تُقيد حركاته، ولا تضع شيئًا في فمه. بعد انتهاء النوبة، ضع الشخص على جنبه للحفاظ على مجرى الهواء وراقبه حتى يستعيد وعيه. إذا استمرت النوبة أكثر من 5 دقائق أو تكررت، اطلب الإسعاف فورًا."
  },

  {
    "role": "user",
    "content": "السلام عليكم، كيف أعرف ما إذا كان جرح الرأس يحتاج خياطة، علماً أنه لا يوجد نزيف الآن؟"
  },
  {
    "role": "assistant",
    "content": "إن كانت الحافة واسعة أو مفتوحة بعمق أو كان الحجم أكبر من 1 سم أو كان الجرح في وجهٍ يؤثر على الشكل، فقد يحتاج للخياطة ويُفضل عرض المصاب على الطبيب. نظف الجرح بمحلول فيزيولوجي، غطّه بشاش معقم، وتجنب إعادة الخياطة بعد مرور أكثر من 12 ساعة دون استشارة طبية. إذا ظهرت علامات التهاب أو عدم التئام، راجع الطبيب."
  },

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
st.title("المساعد الخاص بالإسعافات الأولية")

api_key = st.text_input(" :من فضلك أدخل المفتاح الخاص بك ", type="password")

user_input = st.text_area(":اكتب سؤالك باللغة العربية", height = 100 )

if st.button("اسأل"):
    if not api_key:
        st.warning("من فضلك أدخل المفتاح الخاص بك")
    elif not user_input.strip():
        st.warning("من فضلك اكتب سؤالك ")
    else:
        with st.spinner("...جار تحليل سؤالك"):
            category = classify_question(user_input)

        if category == "LABEL_1":
                with st.spinner("جارٍ التحقق من صحة السؤال..."):
                     is_first_aid = llm_first_aid_check(user_input, api_key)

                if is_first_aid:
                     st.success("..تم التعرف على سؤال إسعافات أولية. نحصل الآن على الإرشادات ✅")
                     with st.spinner("جارٍ التواصل مع المساعد الذكي..."):
                          answer = get_mistral_response(user_input, api_key)
                          st.write("### 💬 الجواب:")
                          st.write(answer)
                else:
                     st.warning("يبدو أن السؤال لا يتعلق بالإسعافات الأولية ❌")
        else: 
                st.error("عذرًا، يمكنني الإجابة فقط على الأسئلة المتعلقة بالإسعافات الأولية❌")
                 


