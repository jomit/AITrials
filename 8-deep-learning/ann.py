# Install Theano : pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Install Tensorflow : https://www.tensorflow.org/install/install_windows
# Install Keras : pip install --upgrade keras

#### 1) Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # include all columns from index 3 to 12
y = dataset.iloc[:, 13].values

# Encoding Categorical Data (strings)
# Encoding Independent Variables  "Country" and "Gender"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_country = LabelEncoder()
X[:,1] = labelencoder_country.fit_transform(X[:,1])
labelencoder_gender = LabelEncoder()
X[:,2] = labelencoder_gender.fit_transform(X[:,2])

# only categorize the country column as we remove 1 column at the end and gender only has 2 columns so it will be only 1 in the end anyway
onehoteencoder = OneHotEncoder(categorical_features=[1]) 
X = onehoteencoder.fit_transform(X).toarray()

# Remove Dummy Variable Trap, so remove 1 category column from the country
X = X[:, 1:]  # removing the first column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#### 2) Create Artificial Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
# units = nodes in the output layer
# units = (number of nodes in input layer + number of nodes in output layer) / 2
# units = ( 11 + 1 ) / 2
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11 ))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# activation = "sigmoid" as we need the output as probabilities
# activation = "softmax" for categorical output or more than 1 outputs
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# optimize = algorithm to optimize the weights, i.e. stocastic gradient algorithm
# loss = loss function within the stocastic grading algorithm, "binary_crossentropy" or "categorical_crossentropy"
# metrics = criterion used to evaluate the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch_size =  number of samples per gradient, i.e number of observations after which we want to update the weights
# nb_epoch = number of iterations over the training data
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#### 3) Making Predictions

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  # convert the probablies into '0' or '1' based on a threshold

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#### Accuracy = (TP + TN) / (TP + TN + FP + FN)
#### Precision = TP / (TP + FP)
#### Recall = TP / (TP + FN)
#### F1 Score = 2 * Precision * Recall / (Precision + Recall)

tp = cm[0][0]
tn = cm[1][1]
fp = cm[0][1]
fn = cm[1][0]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)    # measuring exactness
recall  = tp / (tp + fn)    # measuring completeness
f1 = 2 * precision * recall / (precision + recall)   # compromise between precision and recall

print("Accuracy => ",accuracy)
print("Precision => ",precision)
print("Recall => ",recall)
print("F1 => ",f1)


