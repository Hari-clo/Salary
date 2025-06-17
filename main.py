from model import train_model, predict_salary
from frontend import show_ui

model = train_model("Salary_dataset.csv")
if __name__ == "__main__":
    show_ui(model, predict_salary)
