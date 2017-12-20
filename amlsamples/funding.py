#========================= LOAD DATASET =========================
from azureml.dataprep.package import run

dataset = run('startups.dprep', dataflow_idx=0, spark=False)
#print ('Startups dataset shape: {}'.format(dataset.shape))

X = dataset.iloc[:,:-1].values  
y = dataset.iloc[:,4].values

#print(X)
#print(y)

#========================= DATA PREPROCESSING =========================

# 1) Encoding Categorical data in "State" column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

# Columns => 0 = R&D, 1 = Administration, 2 = Marketing, 3 = State
#print(X)

# 2) Converting Categorial values into individual columns

onehoteencoder = OneHotEncoder(categorical_features = [3])  
X = onehoteencoder.fit_transform(X).toarray()

#Columns => 0 = California, 1 = Florida, 2 = Newyork, 3 = R&D, 4 = Administration, 5 = Marketing
#print(X)

# 2) Avoiding the Dummy variable trap (a.k.a Multicollinearity)

X = X[:,1:]   # removing first column from the X

# 3) Splitting the dataset into Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#========================= MODEL BUILDING & PREDICTING =========================

# 1) Fit the Multiple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 2) Predict the results on Test set
y_pred = regressor.predict(X_test)

#print(y_test)
#print(y_pred)


#========================= OPTIMIZING THE MODEL =========================

# 1) Signifiance Level = 0.05

# 2) Building the optimal model using Backward Elimination (Ordinary Least Squares model) using all independent variables
import statsmodels.formula.api as sm
import numpy as np

# adding extra column for b0 constant as the first column, as per the math
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# select all columns to begin with
# Columns => 0 = California, 1 = Florida, 2 = Newyork, 3 = R&D, 4 = Administration, 5 = Marketing
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

# 3) Keep removing columns with hightest P-values and re-fit the model
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regressor_OLS.summary())


