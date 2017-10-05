import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Load the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values  
y = dataset.iloc[:,1].values  # index of dependent column "Salary"

#### Splitting the dataset into Training and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#### Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test)   # no need to fit as sc_X is already fitted to the X_train set 
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#### Fit Simple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#### Predicting the Test set results
y_pred = regressor.predict(X_test)

## Visualising the Training set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs. Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

## Visualising the Test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs. Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


