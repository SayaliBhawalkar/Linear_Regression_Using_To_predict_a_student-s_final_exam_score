import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# step no 2
data = {'Hours_study' : [2,3,4,5,6,7,8,9,10], 'Exam_score' : [50,60,70,75,80,85,90,92,95]}

# step no 3
df = pd.DataFrame(data)
print(df)

# step no 4
x = df[['Hours_study']]
y = df[['Exam_score']]

# step no 5
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

# step no 6
model = LinearRegression()

# step no 7
model.fit(x_train, y_train)

# user input testing
user_input = float(input("Enter the number of hours you study"))

predicted_score = model.predict([[user_input]])

# printing the output
print(f"Predicted Exam Score: {predicted_score[0]}")
