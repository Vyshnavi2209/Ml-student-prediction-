import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Scores": [20, 25, 35, 45, 50, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]
y = df["Scores"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

study_hours = float(input("Enter number of study hours: "))
prediction = model.predict([[study_hours]])

print(f"Predicted Score: {prediction[0]:.2f}")

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")
