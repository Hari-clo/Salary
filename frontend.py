import streamlit as st

def show_ui(model, predict_func):
    st.title("💼 Salary Predictor")
    st.markdown("Enter your years of experience:")

    years = st.slider("Years of Experience", 0.0, 20.0, 2.0, step=0.1)

    if st.button("Predict Salary"):
        prediction = predict_func(model, years)
        st.success(f"Estimated Salary: ₹ {prediction:,.2f}")
