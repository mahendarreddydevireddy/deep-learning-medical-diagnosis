import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Load model and files
model = load_model("model.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
le_disease = pickle.load(open("le_disease.pkl", "rb"))
le_prescription = pickle.load(open("le_prescription.pkl", "rb"))
max_length = pickle.load(open("max_length.pkl", "rb"))

# UI
st.set_page_config(page_title="Medical Diagnoser", page_icon="🩺")

st.title("🩺 AI Medical Diagnoser")
st.write("Enter your symptoms and get predictions")

user_input = st.text_area("Describe your symptoms:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms")
    else:
        # Preprocess
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict
        pred_disease, pred_prescription = model.predict(padded)

        disease_idx = np.argmax(pred_disease)
        prescription_idx = np.argmax(pred_prescription)

        disease = le_disease.inverse_transform([disease_idx])[0]
        prescription = le_prescription.inverse_transform([prescription_idx])[0]

        # Output
        st.success(f"🦠 Predicted Disease: {disease}")
        st.info(f"💊 Suggested Prescription: {prescription}")
       
