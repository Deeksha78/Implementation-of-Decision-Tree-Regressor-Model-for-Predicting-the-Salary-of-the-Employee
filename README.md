# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets
2. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters
3. Train your model -Fit model to training data -Calculate mean salary value for each subset
4. Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
5. Tune hyperparameters -Experiment with different hyperparameters to improve performance
6. Deploy your model Use model to make predictions on new data in real-world application.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Deeksha P
RegisterNumber:  212222040031
*/
```
```
/*
import pandas as pd
df=pd.read_csv('/content/Salary.csv')

df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:
1.data.head()

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/aff20424-34ad-40b1-8485-a20130b067e0)

2.data.info()

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/c22d86f9-3ae8-4826-963e-345a5ed2523c)

3.data.isnull().sum()

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/42ad78e3-5e5b-4347-8f72-a2c6ad915926)

4.data.head() for position:

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/1df1f4e0-5304-4dad-b2af-e2289142fe96)

5.MSE value:

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/3e2bebea-7c04-4191-a2a6-4a1dbfc25738)

6.R2 value:

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/5e284041-4628-49fc-9d5f-ca516b5339a8)

7.Prediction Value:

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/128116204/3703e2c8-b70a-43ad-b84a-8c19146ab668)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
