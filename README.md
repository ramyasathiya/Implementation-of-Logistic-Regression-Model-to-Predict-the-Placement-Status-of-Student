# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAMYA S
RegisterNumber:  212222040130
*/
```
import pandas as pd
data = pd.read_csv('/content/Placement_Data.csv')
data.head()
```
![image](https://github.com/user-attachments/assets/47fb9a8a-ba12-4f31-8ea0-94f0700f76d2)
```
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
```
![image](https://github.com/user-attachments/assets/4caa4cc3-9eb3-4f30-91af-e3374937c5dc)
```
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/e5f0d869-70ec-401c-9ee7-d8d61470dfca)
```
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/02ae8c1d-4488-4def-bf96-cbd8365b4b1d)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1['gender'] = le.fit_transform(data1['gender'])
data1['ssc_b'] = le.fit_transform(data1['ssc_b'])
data1['hsc_b'] = le.fit_transform(data1['hsc_b'])
data1['hsc_s'] = le.fit_transform(data1['hsc_s'])
data1['degree_t'] = le.fit_transform(data1['degree_t'])
data1['workex'] = le.fit_transform(data1['workex'])
data1['specialisation'] = le.fit_transform(data1['specialisation'])
data1['status'] = le.fit_transform(data1['status'])
data1.head()
```
![image](https://github.com/user-attachments/assets/2de67e31-f4b3-4639-a564-53facd0c78a7)
```
X = data1.iloc[:,:-1]
X.head()
```
![image](https://github.com/user-attachments/assets/dc731ce1-1c13-4911-9fef-9840cbf1be0e)
```
y = data1["status"]
y.head()
```
![image](https://github.com/user-attachments/assets/98af5a89-4cb0-416e-9dc4-8592e0d9bf65)
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred
```
![image](https://github.com/user-attachments/assets/f259cc83-3a1d-4d7d-a6ab-0e2413144369)
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/867f7672-4bbe-405b-b767-9bdfa7b01e95)
```
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/cc996ab1-9d48-4993-8ee5-0facd4f06f09)
![image](https://github.com/user-attachments/assets/8e3df140-6c85-44ca-b52f-03643cdc40d6)













## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
