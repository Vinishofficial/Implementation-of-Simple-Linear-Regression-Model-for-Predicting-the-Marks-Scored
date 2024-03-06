# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph.
5.Predict the regression for the marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VINISHRAJ R
RegisterNumber:212223230243


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/
```

## Output:
## df.head():
![309643706-6ded435a-910a-4f7b-9956-19fca18fa635](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/ef46c0f6-4447-4f56-8afc-fe318cb3608d)
## df.tail():
![309643788-0b62a000-1029-4629-add8-2d5e68f0ee82](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/017fdacf-7a01-49fb-8f18-b7a5775cc833)
## Values of x:
![309643815-f2faf9c5-7319-43dd-abf1-e900739fa167](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/5e05953c-0793-4a91-8a66-a5d8ef347e2e)
## Values of y:
![309643840-baa26732-40d3-46f9-888a-eafe35d8112c](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/6ff1ba6b-3660-4575-b74d-dde54aac85e6)
## Values of y prediction:
![image](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/0207ca65-32ff-431f-8a4e-5a256b3e9b31)
## Values of y test:
![image](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/727196ee-e3ca-4964-acfe-eb7b0b9ebb53)
## Training set graph:
![309643933-11d63dd7-0e99-4804-b8b4-ec16d882ff95](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/986f7110-1109-4855-9167-c9acc503bca8)
## Test set graph:
![309643956-9b138331-5501-48e8-bc43-baa054b8f0c7](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/c1275365-add5-4e06-8fda-52e99f3c2a6b)
## Value of MSE,MAE & RMSE:
![image](https://github.com/Vinishofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146931793/fe8fac50-634a-4a65-b207-1f9c39870dbe)










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
