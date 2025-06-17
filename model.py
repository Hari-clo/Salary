import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(csv_path):
    data = pd.read_csv(csv_path)
    X = data[['YearsExperience']]
    y = data['Salary']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_salary(model, years):
    return model.predict([[years]])[0]
