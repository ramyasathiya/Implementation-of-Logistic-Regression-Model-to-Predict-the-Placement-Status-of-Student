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
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
```

```
data1.isnull().sum()
```

```
data1.duplicated().sum()
```

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

```
X = data1.iloc[:,:-1]
X.head()
```

```
y = data1["status"]
y.head()
```

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred
```

```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
```

```
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
```














## Output:
![image](https://github.com/user-attachments/assets/8d539355-17f5-4585-b341-c6ec13e35fe7)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
