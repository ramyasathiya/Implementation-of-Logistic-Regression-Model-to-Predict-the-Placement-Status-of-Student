# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Clean Data: Read the dataset, drop irrelevant columns, and check for null or duplicate values.

2.Encode Categorical Variables: Use LabelEncoder to convert categorical variables (e.g., gender, education background) into numerical form for modeling.

3.Split Data: Separate the features (x) and target variable (y), then split the data into training and testing sets.

4.Train Model: Train a logistic regression model using the training data and predict on the test data.

5.Evaluate Model: Calculate and print accuracy, confusion matrix, and classification report to assess model performance.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAMYA S

RegisterNumber:  212222040130

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print("\nClassification Report:\n",cr)
*/
```

## Output:
## ACCURACY
![image](https://github.com/user-attachments/assets/4f898811-2021-4b72-ae0b-113f9e875bb2)
# CONFUSION MATRIX

![image](https://github.com/user-attachments/assets/f910dd69-64e1-4b3e-970b-dde6b1057178)

![image](https://github.com/user-attachments/assets/c34da8ef-1bb0-468d-b164-0d57bd7aaf06)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
