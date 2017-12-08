import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Perform Stemming
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,1000):
    
    # Replace all characters other than alphabets with 'spaces'
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    
    # # Remove stop words and Stemming
    ps = PorterStemmer()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]   # set() is way faster
    
    review = ' '.join(review)
    corpus.append(review)
    

# Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   # 1500 based on the number of columns, max_features is to include most relevant words
X = cv.fit_transform(corpus).toarray()   # creates a sparse matrix with lots of 0's
y = dataset.iloc[:,1].values


### Now we can use the Classification Model steps

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
""" from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train) """

# Fitting Decision Tree to the Training set
""" from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train) """

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

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
