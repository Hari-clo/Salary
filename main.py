import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title('Salary Prediction App 💰')

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Salary_Data.csv')
    df.dropna(inplace=True)
    return df

# Load the data
df = load_data()

# Prepare the model
def prepare_model(df):
    le = LabelEncoder()
    df['Ge'] = le.fit_transform(df['Gender'])
    df['Edu'] = le.fit_transform(df['Education Level'])
    
    X = df[['Age', 'Ge', 'Edu', 'Years of Experience']]
    y = df['Salary']
    
    model = LinearRegression()
    model.fit(X, y)
    return model, le

# Create the model
model, le = prepare_model(df)

# Create input fields
st.markdown('## Enter Your Information')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
years_exp = st.slider('Years of Experience', 0, 50, 5)
gender = st.selectbox('Gender', df['Gender'].unique())
education = st.selectbox('Education Level', df['Education Level'].unique())

# Make prediction
if st.button('Predict Salary'):
    # Transform categorical inputs
    gender_encoded = le.fit_transform([gender])[0]
    education_encoded = le.fit_transform([education])[0]
    
    # Create input array
    input_data = np.array([[age, gender_encoded, education_encoded, years_exp]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.markdown('## Predicted Salary 💵')
    st.success(f'The predicted salary is ${prediction:,.2f}')

# Add information about the model
st.markdown('---')
st.markdown('### About this Model')
st.markdown('''
This model predicts salaries based on:
- Age
- Gender
- Education Level
- Years of Experience

The predictions are based on historical salary data and use linear regression for the predictions.
''')