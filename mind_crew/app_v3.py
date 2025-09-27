# app.py
import streamlit as st
from predict import predict_disease, symptom_list
import re
import os

# ----------------- OpenAI Setup ----------------- #
USE_GPT = False
GPT_AVAILABLE = False
openai_api_key = None

try:
    from dotenv import load_dotenv
    import openai
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
        GPT_AVAILABLE = True
except ImportError:
    pass

# If GPT is selected but key missing, ask for input
if not openai_api_key:
    st.info("OpenAI API key not found in .env. Enter your key to use GPT mode.")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if openai_api_key:
        try:
            import openai
            openai.api_key = openai_api_key
            GPT_AVAILABLE = True
        except ImportError:
            st.error("OpenAI library not installed. GPT mode unavailable.")

# ----------------- Symptom extraction ----------------- #
def extract_symptoms_keyword(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', ' ', user_input)
    extracted = [symptom for symptom in symptom_list if re.search(r'\b' + re.escape(symptom.lower()) + r'\b', user_input)]
    return extracted

def extract_symptoms_gpt(user_input):
    import openai
    prompt = f"""
You are a medical assistant. Extract all symptoms mentioned in the user's text.
Only return symptoms that match exactly (or are very similar) to this list: {symptom_list}.
Respond with a Python-style list of symptoms.
User input: "{user_input}"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        extracted_text = response.choices[0].message.content.strip()
        extracted_symptoms = re.findall(r"'(.*?)'|\"(.*?)\"", extracted_text)
        return [s[0] or s[1] for s in extracted_symptoms if s[0] or s[1]]
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return []

# ----------------- Streamlit UI ----------------- #
st.set_page_config(page_title="Disease Symptom Prediction", layout="wide")
st.title("🩺 Disease Symptom Prediction System")

# Mode selection
mode = st.radio(
    "Select Symptom Extraction Mode:",
    ("Non-GPT (Keyword Matching)", "GPT (AI Extraction)" if GPT_AVAILABLE else "GPT Not Available")
)

st.write("You can either describe your symptoms in plain language OR select them from the list below:")

# User input
user_input = st.text_area("Describe your symptoms:", height=100)

# Sidebar checkbox selection
st.sidebar.header("Select Symptoms (Optional)")
selected_symptoms = []
cols = st.sidebar.columns(2)  # 2 columns for checkboxes
for i, symptom in enumerate(symptom_list):
    with cols[i % 2]:
        if st.checkbox(symptom):
            selected_symptoms.append(symptom)

# Predict button
if st.button("Predict Disease"):
    combined_symptoms = set(selected_symptoms)

    # Extract symptoms from input based on selected mode
    if user_input.strip():
        if mode.startswith("GPT") and GPT_AVAILABLE:
            ai_symptoms = extract_symptoms_gpt(user_input)
        else:
            ai_symptoms = extract_symptoms_keyword(user_input)
        combined_symptoms |= set(ai_symptoms)

    if not combined_symptoms:
        st.warning("No recognizable symptoms found. Please type or select at least one symptom!")
    else:
        # Prediction
        with st.spinner("Analyzing symptoms and predicting disease..."):
            result = predict_disease(list(combined_symptoms))

        # ----------------- Display Results ----------------- #
        tabs = st.tabs(["Predicted Disease", "Description", "Precautions"])

        with tabs[0]:
            st.success(result["disease"])

        with tabs[1]:
            st.info(result["description"])

        with tabs[2]:
            for p in result["precautions"]:
                with st.expander(p):
                    st.write("Follow this precaution carefully.")
