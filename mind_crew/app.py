# app.py
import streamlit as st
from predict import predict_disease, symptom_list
import re

# Function to extract symptoms from user input
def extract_symptoms(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', ' ', user_input)

    extracted = []
    for symptom in symptom_list:
        pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
        if re.search(pattern, user_input):
            extracted.append(symptom)

    return extracted

# Streamlit UI
st.set_page_config(page_title="Disease Symptom Prediction", layout="centered")
st.title("🩺 Disease Symptom Prediction System")
st.write("You can either describe your symptoms in plain language OR select them from the list below.")

# User input text
user_input = st.text_area("Describe your symptoms:", height=100)

st.write("### Select Symptoms from the list (optional):")
selected_symptoms = []
cols = st.columns(4)  # 4 columns for checkboxes
for i, symptom in enumerate(symptom_list):
    with cols[i % 4]:
        if st.checkbox(symptom):
            selected_symptoms.append(symptom)

# Combine symptoms from text input and checkboxes
combined_symptoms = set(extract_symptoms(user_input)) | set(selected_symptoms)

if st.button("Predict Disease"):
    if not combined_symptoms:
        st.warning("Please enter or select at least one symptom!")
    else:
        result = predict_disease(list(combined_symptoms))
        
        st.subheader("Predicted Disease:")
        st.success(result["disease"])
        
        st.subheader("Description:")
        st.info(result["description"])
        
        st.subheader("Precautions:")
        for p in result["precautions"]:
            st.write("- " + p)
