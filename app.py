import streamlit as st
import pickle
import numpy as np
import os
import subprocess

# Streamlit Page Config
st.set_page_config(page_title="Tourist Spending Prediction", page_icon="🌍")
st.title("🌍 Tourist Spending Prediction")
st.write("Predict tourist spending using the latest model from DVC.")

# Step 1: Pull latest model from DVC
st.info("🔄 Pulling latest model from DVC...")
try:
    subprocess.run(["dvc", "pull", "-f"], check=True)
    st.success("✅ Latest model pulled from DVC.")
except subprocess.CalledProcessError as e:
    st.error(f"❌ Failed to pull model from DVC: {e}")
    st.stop()

# Step 2: Load the model
model_path = "random_forest_best.pkl"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found even after DVC pull.")
    st.stop()

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Step 3: Input form for prediction
st.subheader("Enter Tourist Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=1000)
with col2:
    nights = st.number_input("Nights Stayed", min_value=0, value=3)
    num_activities = st.number_input("Number of Activities", min_value=0, value=2)

# Step 4: Predict
if st.button("🔮 Predict Spending"):
    try:
        features = np.array([[age, income, nights, num_activities]])
        prediction = model.predict(features)
        st.success(f"💰 Estimated Spending: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
