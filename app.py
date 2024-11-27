import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("admission_model.pkl")

st.title("University Admission Prediction")

# Load test data to display accuracy
X_test, y_test = joblib.load("test_data.pkl")
accuracy = model.score(X_test, y_test)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

st.write("Enter the exam scores to predict the probability of admission.")

# Input fields for exam scores
exam1 = st.number_input(
    "Exam 1 Score:", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
exam2 = st.number_input(
    "Exam 2 Score:", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)

# Predict button
if st.button("Predict"):
    input_data = np.array([[exam1, exam2]])

    # Predict probability and class
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write(f"### Probability of Admission: {probability * 100:.2f}%")
    st.write(f"### Prediction: {'Admitted' if prediction == 1 else 'Not Admitted'}")
