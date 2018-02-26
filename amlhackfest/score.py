def init():
    from sklearn.externals import joblib

    global model
    model = joblib.load('socialads.pkl')

    '''
    import pandas as pd
    dataset = pd.read_csv('social_network_ads.csv')
    X = dataset.iloc[:, [2, 3]].values

    global sc
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit_transform(X)
    '''

def run(inputData):
    import json
    
    #scaledInputData = sc.transform(inputData)

    # predict profit using input data
    #prediction = model.predict(scaledInputData)
    
    prediction = model.predict(inputData)
    return json.dumps(str(prediction))

if __name__ == '__main__':
    init()
    # Indexes => 0 = Age, 1 = Salary
    age = 19  #19
    salary = 50000 #25000
    predictedValue = run([[age, salary]])
    print("Prediction => ",predictedValue)
