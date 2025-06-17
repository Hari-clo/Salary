import streamlit as st
import random

def show_ui(model, predict_func):
    st.markdown("""
        <h1 style='text-align: center; color: #FF6F61;'>💸 AI Salary Predictor</h1>
        <p style='text-align: center; font-size: 18px;'>Wanna know how fat your paycheck might be? Let's find out 😎</p>
        <hr style="border:1px solid #ccc;">
    """, unsafe_allow_html=True)

    years = st.slider("📅 How many years of experience do you have?", 0.0, 20.0, 1.0, step=0.1)

    if st.button("💥 Predict My Salary"):
        salary = predict_func(model, years)
        
        reactions = [
            f"You might be making ₹ {salary:,.2f} soon!",
            f"Looks like your experience is finally gonna pay off! ₹ {salary:,.2f} 💰",
            f"Holy smokes! You could be earning ₹ {salary:,.2f} 😮",
            f"Bro you better ask for a raise if you're getting less than ₹ {salary:,.2f} 😂",
            f"Not bad at all! ₹ {salary:,.2f} is pretty solid, my dude 💪"
        ]

        st.success(random.choice(reactions))

        if salary > 100000:
            st.balloons()
            st.markdown("### 🎉 You rich rich huh?! Drinks on you today!")

        elif salary < 30000:
            st.warning("💀 You need to talk to HR... or start freelancing on weekends 😂")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("🔮 Powered by Streamlit + ML + President-level vibes.")
