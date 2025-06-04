import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("tiger_lstm_model.h5")
scaler = joblib.load("scaler.save")

st.title("🐅 Tiger Population Predictor (Bandipur)")
st.markdown("Enter tiger populations from the **last 3 years** to predict the next year.")

year1 = st.number_input("Year 1 Population", min_value=1000, max_value=10000, step=100)
year2 = st.number_input("Year 2 Population", min_value=1000, max_value=10000, step=100)
year3 = st.number_input("Year 3 Population", min_value=1000, max_value=10000, step=100)

if st.button("Predict"):
    try:
        input_seq = np.array([[year1], [year2], [year3]])
        scaled_input = scaler.transform(input_seq)
        reshaped_input = scaled_input.reshape(1, 3, 1)
        prediction = model.predict(reshaped_input)
        predicted_value = scaler.inverse_transform([[prediction[0][0]]])[0][0]
        st.success(f"📈 Predicted Population: **{int(predicted_value)}**")
    except Exception as e:
        st.error(f"Error: {e}")
