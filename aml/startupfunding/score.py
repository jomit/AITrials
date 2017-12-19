#from azureml.assets import get_local_path

#model = None

def init():
    # Get the path to the model asset
    #local_path = get_local_path('startupfunding.pkl.link')
    
    from sklearn.externals import joblib
    global model
    model = joblib.load('startupfunding.pkl')

def run(input_df):
    import json
    
    # Predict using appropriate functions
    prediction = model.predict(input_df)
    return json.dumps(str(prediction))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    init()
    rndSpend = [[75000]]
    predictedProfit = run(rndSpend)
    print("Predicted Profit => ",predictedProfit)
