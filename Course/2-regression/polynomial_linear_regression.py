import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Load the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values   # making it a matrix instead of vector
y = dataset.iloc[:,2].values   # y should be a vector

#### Fitting linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#### Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)  # replace 4 with 2 or 3
X_poly = poly_reg.fit_transform(X)  # creates a new X matrix with constant and poly-square columns

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


#### Visualising linear Regression to the dataset
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()


#### Visualising Polynomial Regression to the dataset
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()

# Better fitting using smaller increments
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid) , 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()


#### Predicting with Linear Regression model
lin_reg.predict(6.5)   # predict salary for level 6.5

`
#### Predicting with Polynomial Regression model
lin_reg2.predict(poly_reg.fit_transform(6.5))   # predict salary for level 6.5








