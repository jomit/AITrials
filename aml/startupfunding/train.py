from azure.storage.blob import BlockBlobService
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import sys, os

def loaddata():
    ACCOUNT_NAME = "mlgputraining"
    ACCOUNT_KEY = "GE+U8dqKF/KxRJBSpa4WTEDSe0YUlZRtbqJEwmWk2bi5vZrFivjvxWtfILlvSkbzKyQXzm8Mp8N9+ttLphdz9Q=="
    CONTAINER_NAME = "datasets"

    blobService = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
    blobService.get_blob_to_path(CONTAINER_NAME, 'startups.csv', 'startups.csv')

    dataset = pd.read_csv('startups.csv')
    #print ('Startups dataset shape: {}'.format(dataset.shape))

    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,4].values
    return X, y

def datapreprocessing(X,y):

    # encoding Categorical data in "State" column
    labelencoder_X = LabelEncoder()
    X[:,3] = labelencoder_X.fit_transform(X[:,3])

    # Result Columns => 0 = R&D, 1 = Administration, 2 = Marketing, 3 = State
    # print(X)

    # converting Categorial values into individual columns
    onehoteencoder = OneHotEncoder(categorical_features = [3])  
    X = onehoteencoder.fit_transform(X).toarray()

    # Result Columns => 0 = California, 1 = Florida, 2 = Newyork, 3 = R&D, 4 = Administration, 5 = Marketing
    # print(X)

    # avoiding the Dummy variable trap (a.k.a Multicollinearity)
    X = X[:,1:]   # removing first column from the X

    # splitting the dataset into Training and Test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

    return X_train, X_test, y_train, y_test, X

def modelbuilding(X_train, X_test, y_train, y_test):

    # fit the Multiple linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # evaluate the test set and get accuracy
    accuracy = regressor.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))

    return regressor

def addmodelcomparison(X_test, y_test, model):

    # initialize the logger
    from azureml.logging import get_azureml_logger
    run_logger = get_azureml_logger() 

    # log accuracy which is a single numerical value
    accuracy = model.score(X_test, y_test)
    run_logger.log("Accuracy", accuracy)

def savemodel(model):

    # create the outputs folder
    os.makedirs('./outputs', exist_ok=True)

    # serialize the model
    print ("Export the model to startupfunding.pkl")
    f = open('./outputs/startupfunding.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

def run(input_df):
    import json
        
    prediction = regressor.predict(input_df)

    prediction = "%s %d" % (str(input_df), regressor)
    return json.dumps(str(prediction))

def createwebserviceschema():

    #========================= Create schema.json  =========================
    
    from azureml.api.schema.dataTypes import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services import generate_schema

    inputs = {"input_df": SampleDefinition(DataTypes.NUMPY, X_test)}
    print(generate_schema(run_func=run, inputs=inputs, filepath="./outputs/schema.json"))

def modeloptimization():
    X, y = loaddata()
    X_train, X_test, y_train, y_test, X = datapreprocessing(X, y)
    modelbuilding(X_train, X_test, y_train, y_test)

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
    print(regressor_OLS.summary())

    # 4) Keep removing columns with hightest P-values and re-fit the model
    X_opt = X[:,[0,3,4,5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    print(regressor_OLS.summary())

if __name__ == '__main__':
    
    # Load Data from Azure Blob Storage
    X, y = loaddata()

    # Data Preprocessing (convert categorial values)
    X_train, X_test, y_train, y_test, X = datapreprocessing(X, y)
    
    # Build Model
    model = modelbuilding(X_train, X_test, y_train, y_test)

    # Add model comparison numerical value using Azure ML Logger
    addmodelcomparison(X_test, y_test, model)

    # Save the model in Azure Blob Storage
    savemodel(model)
    
    
    # (Optional) Optimizing the model
    #modeloptimization()

