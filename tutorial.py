import os
import pandas as pd
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

class MLTutorial:
    def __init__(self):
        # Load the train and test datasets to create two DataFrames
        train_url = "https://jomitmlstorage.blob.core.windows.net/aitrials/train.csv"
        self.train = pd.read_csv(train_url)

        test_url = "https://jomitmlstorage.blob.core.windows.net/aitrials/test.csv"
        self.test = pd.read_csv(test_url)

        print(self.train.head())
        print(self.test.head())
        # print(train.describe())
        # print(test.describe().shape)

    def inspectData(self):
        # Passengers that survived vs passengers that passed away
        print(self.train["Survived"].value_counts())

        # As proportions
        print(self.train["Survived"].value_counts(normalize = True))

        # Males that survived vs males that passed away
        print(self.train["Survived"][self.train["Sex"] == 'male'].value_counts())

        # Females that survived vs Females that passed away
        print(self.train["Survived"][self.train["Sex"] == 'female'].value_counts())

        # Normalized male survival
        print(self.train["Survived"][self.train["Sex"] == 'male'].value_counts(normalize = True))

        # Normalized female survival
        print(self.train["Survived"][self.train["Sex"] == 'female'].value_counts(normalize = True))

    def createChildColumns(self):
        # Create the column Child and assign to 'NaN'
        self.train["Child"] = float('NaN')

        # Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
        self.train["Child"][self.train["Age"] < 18.0 ] = 1
        self.train["Child"][self.train["Age"] >= 18.0 ] = 0
        print(self.train["Child"])

        # Print normalized Survival Rates for passengers under 18
        print(self.train["Survived"][self.train["Child"] == 1].value_counts(normalize = True))

        # Print normalized Survival Rates for passengers 18 or older
        print(self.train["Survived"][self.train["Child"] == 0].value_counts(normalize = True))

    def firstBasicPrediction(self):
        # Create a copy of test: test_one
        test_one = self.test

        # Initialize a Survived column to 0
        test_one["Survived"] = 0

        # Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
        test_one["Survived"][test_one["Sex"] == "female"] = 1
        print(test_one["Survived"])

    def cleaningAndFormattingData(self):
        self.train["Age"] = self.train["Age"].fillna(self.train["Age"].median())

        # Convert the male and female groups to integer form
        self.train["Sex"][self.train["Sex"] == "male"] = 0
        self.train["Sex"][self.train["Sex"] == "female"] = 1

        # Impute the Embarked variable
        self.train["Embarked"] = self.train["Embarked"].fillna('S')

        # Convert the Embarked classes to integer form
        self.train["Embarked"][self.train["Embarked"] == "S"] = 0
        self.train["Embarked"][self.train["Embarked"] == "C"] = 1
        self.train["Embarked"][self.train["Embarked"] == "Q"] = 2

        #Print the Sex and Embarked columns
        print(self.train["Sex"])
        print(self.train["Embarked"])

    def createFirstDecisionTree(self):
        # Print the train data to see the available features
        print(self.train)

        # Create the target and features numpy arrays: target, features_one
        target = self.train["Survived"].values
        features_one = self.train[["Pclass", "Sex", "Age", "Fare"]].values

        # Fit your first decision tree: my_tree_one
        self.my_tree_one = tree.DecisionTreeClassifier()
        self.my_tree_one = self.my_tree_one.fit(features_one, target)

        # Look at the importance and score of the included features
        print(self.my_tree_one.feature_importances_)
        print(self.my_tree_one.score(features_one, target))

    def predictAndExport(self):

        # Impute the missing value with the median
        self.test.Fare[152] = self.test.Fare.median()
        self.test["Age"] = self.test["Age"].fillna(self.test["Age"].median())

        self.test["Sex"][self.test["Sex"] == "male"] = 0
        self.test["Sex"][self.test["Sex"] == "female"] = 1        

        # Extract the features from the test set: Pclass, Sex, Age, and Fare.
        test_features = self.test[["Pclass", "Sex", "Age", "Fare"]].values

        # Make your prediction using the test set and print them.
        my_prediction = self.my_tree_one.predict(test_features)
        print(my_prediction)

        # Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
        PassengerId =np.array(self.test["PassengerId"]).astype(int)
        my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
        print(my_solution)

        # Check that your data frame has 418 entries
        print(my_solution.shape)

        # Write your solution to a csv file with the name my_solution.csv
        my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

    def controlOverfitting(self):
        # Create a new array with the added features: features_two
        self.features_two = self.train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
        target = self.train["Survived"].values

        #Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
        max_depth = 10
        min_samples_split = 5
        self.my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
        self.my_tree_two = self.my_tree_two.fit(self.features_two,target)

        #Print the score of the new decison tree
        print(self.my_tree_two.score(self.features_two,target))

    def featureEngineering(self):
        # Create train_two with the newly defined feature
        train_two = self.train.copy()
        train_two["family_size"] = (train_two["SibSp"] + train_two["Parch"] + 1)
        target_two = train_two["Survived"].values

        # Create a new feature set and add the new feature
        features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

        # Define the tree classifier, then fit the model
        self.my_tree_three = tree.DecisionTreeClassifier()
        self.my_tree_three = self.my_tree_three.fit(features_three, target_two)

        # Print the score of this decision tree
        print(self.my_tree_three.score(features_three, target_two))

    def randomForestAnalysis(self):
        # Import the `RandomForestClassifier`
        from sklearn.ensemble import RandomForestClassifier

        # We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
        features_forest = self.train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
        target = self.train["Survived"].values

        # Building and fitting my_forest
        forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
        my_forest = forest.fit(features_forest, target)

        # Print the score of the fitted random forest
        print(my_forest.score(features_forest, target))

        # Impute the Embarked variable
        self.test["Embarked"] = self.test["Embarked"].fillna('S')

        # Convert the Embarked classes to integer form
        self.test["Embarked"][self.test["Embarked"] == "S"] = 0
        self.test["Embarked"][self.test["Embarked"] == "C"] = 1
        self.test["Embarked"][self.test["Embarked"] == "Q"] = 2

        # Compute predictions on our test set features then print the length of the prediction vector
        test_features = self.test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
        pred_forest = my_forest.predict(test_features)
        print(len(pred_forest))

        #Request and print the `.feature_importances_` attribute
        print(self.my_tree_two.feature_importances_)
        print(my_forest.feature_importances_)

        #Compute and print the mean accuracy score for both models
        print(self.my_tree_two.score(self.features_two, target))
        print(my_forest.score(features_forest,target))



#end of MLTutorial class

def main():
  ml = MLTutorial()
  ml.inspectData()
  ml.createChildColumns()
  ml.firstBasicPrediction()
  ml.cleaningAndFormattingData()
  ml.createFirstDecisionTree()
  ml.predictAndExport()
  ml.controlOverfitting()
  ml.randomForestAnalysis()

if __name__ == "__main__":
    main()

# end script