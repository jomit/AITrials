def init():
    from sklearn.externals import joblib
    global model
    model = joblib.load('startupfunding.pkl')

def run(inputData):
    import json
    
    # predict profit using input data
    prediction = model.predict(inputData)
    return json.dumps(str(prediction))

if __name__ == '__main__':
    init()
    # Indexes => 0 = Florida, 1 = Newyork, 2 = R&D, 3 = Administration, 4 = Marketing
    isCompanyInFlorida = 0
    isCompanyInNewyork = 1
    rndSpend = 75000
    adminSpend = 10000
    marketingSpend = 150000
    predictedProfit = run([[isCompanyInFlorida,isCompanyInNewyork,rndSpend,adminSpend,marketingSpend]])
    print("Predicted Profit => ",predictedProfit)
