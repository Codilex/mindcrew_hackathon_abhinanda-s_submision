# app.py
import streamlit as st
from predict import predict_disease, symptom_list
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load OpenAI API key from .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_symptoms_openai(user_input, symptom_list):
    """
    Use OpenAI API (v1.0+ syntax) to extract symptoms from user input
    """
    prompt = f"""
You are a medical assistant. Extract all symptoms mentioned in the user's text.
Only return symptoms that match exactly (or are very similar) to this list: {symptom_list}.
Respond with a Python-style list of symptoms.
User input: "{user_input}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        extracted_text = response.choices[0].message.content.strip()

        # Extract list of symptoms from the response
        extracted_symptoms = re.findall(r"'(.*?)'|\"(.*?)\"", extracted_text)
        extracted_symptoms = [s[0] or s[1] for s in extracted_symptoms if s[0] or s[1]]
        return extracted_symptoms
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return []

# Streamlit UI
st.set_page_config(page_title="Disease Symptom Prediction", layout="centered")
st.title("🩺 Disease Symptom Prediction System with AI")
st.write("You can either describe your symptoms in plain language OR select them from the list below.")

# User input
user_input = st.text_area("Describe your symptoms:", height=100)

st.write("### Select Symptoms from the list (optional):")
selected_symptoms = []
cols = st.columns(4)  # 4 columns for checkboxes
for i, symptom in enumerate(symptom_list):
    with cols[i % 4]:
        if st.checkbox(symptom):
            selected_symptoms.append(symptom)

if st.button("Predict Disease"):
    combined_symptoms = set(selected_symptoms)
    
    if user_input.strip():
        ai_symptoms = extract_symptoms_openai(user_input, symptom_list)
        combined_symptoms |= set(ai_symptoms)
    
    if not combined_symptoms:
        st.warning("No recognizable symptoms found. Please type or select at least one symptom!")
    else:
        result = predict_disease(list(combined_symptoms))
        
        st.subheader("Predicted Disease:")
        st.success(result["disease"])
        
        st.subheader("Description:")
        st.info(result["description"])
        
        st.subheader("Precautions:")
        for p in result["precautions"]:
            st.write("- " + p)
