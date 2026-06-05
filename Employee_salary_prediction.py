import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.title("💼 Employee Salary Prediction WebApp")
st.write("""
This website predicts the expected salary of an employee based on their professional job level 
using a trained **Linear Regression** model.
""")

st.divider()


@st.cache_resource
def load_model():
    try:
        model = joblib.load('predict_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ 'linear_model.pkl' not found! Please make sure your saved model file is in the same folder.")
        return None

model = load_model()

# 3. Create Frontend User Input Section
if model is not None:
    st.subheader("💡 Enter Job Position Details")
    
    # Text-based helper map to help users pick the correct numerical 'Level' 
    # based on the dataset structure seen in your notebook
    position_mapping = {
        1: "Business Analyst",
        2: "Junior Consultant",
        3: "Senior Consultant",
        4: "Manager",
        5: "Country Manager",
        6: "Region Manager",
        7: "Partner",
        8: "Senior Partner",
        9: "C-level",
        10: "CEO"
    }
    
    # Create a simple slider for the user to select the professional position level (1 to 10)
    selected_level = st.slider("Select Employee Level:", min_value=1, max_value=10, value=4)
    
    # Dynamically display corresponding typical position title
    st.info(f"Equivalent typical position: **{position_mapping[selected_level]}**")
    
    st.write("") # Blank space
    
    # 4. Predict Button and Result Output
    if st.button("🔮 Predict Salary"):
        # Convert input into a 2D array structure expected by scikit-learn's model.predict()
        input_data = np.array([[selected_level]])
        
        # Make the prediction
        predicted_salary = model.predict(input_data)[0]

        st.success(f"🎉 **Estimated Salary Prediction:** ${predicted_salary:,.2f}")