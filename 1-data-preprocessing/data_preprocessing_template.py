import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Load the dataset
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values  #  [all rows, columns-1]
y = dataset.iloc[:,3].values  # [all rows, index of last column]


#### Replace Missing Values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean",axis = 0)
imputer = imputer.fit(X[:, 1:3])  # [rows, column index start : column index end], end index is excluded so it only includes column index 1, 2
X[:,1:3] = imputer.transform(X[:,1:3])


#### Encoding Categorical Data (strings)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])  # encoding the "Country" column

# Problem with above is that ML alrogithms may think France (0) is greater than Germany(2)
# To avoid this we would create dummy encodings
# So create 3 columns for each Country category, and then provide 1 or 0 for each row 

onehoteencoder = OneHotEncoder(categorical_features=[0])  # 0 = column index for country column
X = onehoteencoder.fit_transform(X).toarray()

# Encode the Dependent Variable "Purchased" column
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y) 


#### Splitting the dataset into Training and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



#### Feature Scaling

# e.g. Age and Salary are on different scales, Salary has greater range than Age
# lot of ML models are based on Euclidean Distance between points, 
# if all features are not on common scale than some features may dominate the model

# 1) Standardisation  (use standard deviation)
# 2) Normalisation  (use max - min)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test)   # no need to fit as sc_X is already fitted to the X_train set """













 













