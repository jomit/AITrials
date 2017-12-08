import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Load the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values  
y = dataset.iloc[:,4].values

#### Encoding Categorical Data  "State" column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

onehoteencoder = OneHotEncoder(categorical_features=[3])  
X = onehoteencoder.fit_transform(X).toarray()

#### Avoiding the Dummy variable trap
X = X[:,1:]   # removing first column from the X

#### Splitting the dataset into Training and Test
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#### Fit Multiple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#### Predicting the Test set results
y_pred = regressor.predict(X_test)

#### Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # adding extra column for b0 constant as the first column, as per the formula

X_opt = X[:,[0,1,2,3,4,5]]  # select all features as optimal to begin with
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()  

# assuming the significance level = 0.05, 
# remove column with highest P value
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()  

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


















