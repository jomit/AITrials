#========================= LOAD DATASET USING DATAPREP PACKAGE =========================
from azureml.dataprep.package import run

dataset = run('social-ads.dprep', dataflow_idx=0, spark=False)
#print(dataset)

X = dataset.iloc[:, [2,3]].values  # Just use salary column [1] first to show accuracy improvement
y = dataset.iloc[:, 4].values

print(X)
print(y)

'''
#========================= LOAD DATASET USING BLOB STORAGE =========================
from azure.storage.blob import BlockBlobService
import pandas as pd

ACCOUNT_NAME = "mlgputraining" #"<account name>"
ACCOUNT_KEY = "fDP85xqAP9aO/cPgl53ROhe5u3rpMhMI60E+/FDP6sRNWfCm2vAxjRtyApX6/QtOGmvqJtFaw+QMOpp610fjVA==" #"<acccount key>"
CONTAINER_NAME = "datasets" # "<container name>"

blobService = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
blobService.get_blob_to_path(CONTAINER_NAME, 'social_network_ads.csv', 'social_network_ads.csv')

dataset = pd.read_csv('social_network_ads.csv')
print(dataset)

X = dataset.iloc[:, [2, 3]].values  # Just use salary column [1] first to show accurancy improvement
y = dataset.iloc[:, 4].values

print(X)
print(y)
'''


#========================= DATA PREPROCESSING =========================

# 1) Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 2) Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(X_train)
#print(X_test)

#========================= MODEL BUILDING & PREDICTING =========================

# 1) Fitting 2 different models to the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)

classifier.fit(X_train, y_train)

# 2) Predicting the Test set results
y_pred = classifier.predict(X_test)

print(y_pred)

#========================= MODEL EVALUATION =========================

# 1) Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  #  [0,0] and [1,1] are correct predictions

print(cm)

# 2) Calculate and Log Accuracy

#### Accuracy = (TP + TN) / (TP + TN + FP + FN)
#### Precision = TP / (TP + FP)
#### Recall = TP / (TP + FN)
#### F1 Score = 2 * Precision * Recall / (Precision + Recall)

tp = cm[0][0]
tn = cm[1][1]

fp = cm[0][1]
fn = cm[1][0]

accuracy = (tp + tn) / (tp + tn + fp + fn)
print("Accuracy => {}".format(accuracy))
precision = tp / (tp + fp)    # measuring exactness
recall  = tp / (tp + fn)    # measuring completeness
f1 = 2 * precision * recall / (precision + recall)   # compromise between precision and recall
print("F1 Score => {}".format(f1))

#========================= LOGGING MODEL EVALUATION =========================

# initialize the logger
from azureml.logging import get_azureml_logger
run_logger = get_azureml_logger() 

accuracy = classifier.score(X_test, y_test)
run_logger.log("Accuracy", accuracy)
#print ("Accuracy is {}".format(accuracy))


#========================= SAVE MODEL =========================
import pickle
import sys, os

 # create the outputs folder
os.makedirs('./outputs', exist_ok=True)

# serialize the model
print ("Saved the model => socialads.pkl")
f = open('./outputs/socialads.pkl', 'wb')
pickle.dump(classifier, f)
f.close()

#========================= CREATE WEB SERVICE SCHEMA =========================
def run(inputData):
    import json
        
    prediction = classifier.predict(input_df)

    prediction = "%s %d" % (str(input_df), classifier)
    return json.dumps(str(prediction))

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

inputs = {"inputData": SampleDefinition(DataTypes.NUMPY, X_test)}
print(generate_schema(run_func=run, inputs=inputs, filepath="./outputs/schema.json"))

'''
#========================= VISUALISING THE RESULTS =========================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''