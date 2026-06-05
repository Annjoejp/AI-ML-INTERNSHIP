import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

file = r"C:\Users\anjoe\Downloads\student_marks.xlsx"
st.title("Student Marks predictor")

df = pd.read_excel(file)

X = df[["Hours"]]
y = df["Marks"]


model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("marks_model.pkl", "wb"))
model = pickle.load(open("marks_model.pkl", "rb"))

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=1.0
)

if st.button("Predict Marks"):
    prediction = model.predict([[hours]])

    st.success(
        f"Predicted Marks: {prediction[0]:.2f}"
    )
    
