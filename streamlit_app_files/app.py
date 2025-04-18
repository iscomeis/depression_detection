import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the necessary files
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("dummies_columns.pkl", "rb") as f:
    dummies_columns = pickle.load(f)

model = load_model("depression_model.h5")

# App styling: Fancy design
page_bg = """
<style>
body {
    background-color: #EAF6FF; /* Soft light blue */
    font-family: 'Verdana', sans-serif;
}

h1, h2 {
    color: #0A3D62;
}

h1 {
    text-shadow: 2px 2px 5px #93CCEA;
}

button {
    background-color: #1E90FF;
    border-radius: 10px;
    color: white;
    padding: 10px 20px;
    font-size: 18px;
    text-shadow: 1px 1px 2px #000;
    cursor: pointer;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App Title
st.title("🌟 **Depression Prediction App** 🧠")
st.markdown(
    """
Welcome to the **Depression Prediction App**.  
Your mental well-being matters! Please fill in the details below and let our intelligent model guide you.  
""",
    unsafe_allow_html=True,
)

# Question-based sections
st.markdown("### **Personal Details**")
gender = st.selectbox("🧍 Gender:", ["Male", "Female"])
age = st.text_input("🎂 Age (write a number):", placeholder="Enter your age between 18-60")
working_status = st.radio("👨‍💻 Your Status:", ["Working Professional", "Student"])

st.markdown("### **Pressures**")
academic_pressure = st.slider("📚 Academic Pressure:", min_value=1, max_value=5, step=1)
work_pressure = st.slider("🏢 Work Pressure:", min_value=1, max_value=5, step=1)
job_satisfaction = st.slider("😊 Job Satisfaction:", min_value=1, max_value=5, step=1)

st.markdown("### **Mental Health History**")
suicidal_thoughts = st.radio("⚠️ Have you ever had suicidal thoughts?", ["Yes", "No"])
family_history = st.radio("👨‍👩‍👧 Family Mental History:", ["Yes", "No"])

st.markdown("### **Lifestyle Details**")
dietary_habits = st.selectbox("🥗 Dietary Habits:", ["Moderate", "Unhealthy", "Healthy"])
sleep_duration = st.selectbox(
    "💤 Sleep Duration:",
    ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
)
work_study_hours = st.text_input("🕒 Work/Study Hours per day (write a number):", placeholder="Enter hours between 0-12")
financial_stress = st.slider("💰 Financial Stress:", min_value=1, max_value=5, step=1)

# Prediction Button
if st.button("🧮 **Predict Depression Risk**"):
    st.markdown("### **Processing your data... Please wait!**")
    
    try:
        # Convert age and work/study hours to float
        age = float(age)
        work_study_hours = float(work_study_hours)
        
        if not (18 <= age <= 60):
            st.error("⚠️ Age must be between 18 and 60!")
        elif not (0 <= work_study_hours <= 12):
            st.error("⚠️ Work/Study Hours must be between 0 and 12!")
        else:
            # Create an input dataframe
            input_data = pd.DataFrame([{
                "Age": age,
                "Work Pressure": work_pressure,
                "Job Satisfaction": job_satisfaction,
                "Work/Study Hours": work_study_hours,
                "Financial Stress": financial_stress,
                "Gender_Male": 1 if gender == "Male" else 0,
                "Working Professional or Student_Working Professional": 1 if working_status == "Working Professional" else 0,
                "Sleep Duration_7-8 hours": 1 if sleep_duration == "7-8 hours" else 0,
                "Sleep Duration_Less than 5 hours": 1 if sleep_duration == "Less than 5 hours" else 0,
                "Sleep Duration_More than 8 hours": 1 if sleep_duration == "More than 8 hours" else 0,
                "Dietary Habits_Moderate": 1 if dietary_habits == "Moderate" else 0,
                "Dietary Habits_Unhealthy": 1 if dietary_habits == "Unhealthy" else 0,
                "Have you ever had suicidal thoughts ?_Yes": 1 if suicidal_thoughts == "Yes" else 0,
                "Family History of Mental Illness_Yes": 1 if family_history == "Yes" else 0
            }])

            # Align columns with the training data
            input_data = input_data.reindex(columns=dummies_columns, fill_value=0)

            # Scale the input data
            input_scaled = scaler.transform(input_data)

            # Predict using the model
            prediction = model.predict(input_scaled)
            risk_score = prediction[0][0]

            # Display the result
            if risk_score > 0.1689758449792862:
                st.error(
                    "⚠️ **High Risk of Depression Detected!** It's strongly advised to seek professional help."
                )
            else:
                st.success("✅ **Low Risk of Depression Detected!** Keep taking care of your mental health!")

    except ValueError:
        st.error("⚠️ Please enter valid numeric values for Age and Work/Study Hours!")

# Footer Disclaimer
st.markdown("---")
st.markdown(
    """
<sub>
⚠️ **Disclaimer:** This application is intended for demonstration purposes only.  
It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult your physician or other qualified health provider with any questions regarding a medical condition.  
</sub>
""",
    unsafe_allow_html=True,
)
