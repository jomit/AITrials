## Create DSVM's for all participants

- `cd armtemplates`

- `az group create --name MachineLearningHackfest --location westus`

- `az network vnet create -n MLVNet -g MachineLearningHackfest -l westus --subnet-name default`

- Run the `AgreeTerms-Script.ps1` to agree the terms before creating the VM's

- Update `count` in the `parameters.json` file

- `az group deployment create -n CreateDSVM -g MachineLearningHackfest --template-file template.json --parameters @parameters.json`

- Install Docker for Windows : `https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe`

- Install AML Workbench : `https://aka.ms/azureml-wb-msi` (TODO)


## Create AML Experimentation and Model Management Services using Portal

- Experimentation Service Name : `mlexperiments`

- Model Management Service Name : `mlexprimentsModelMgmt` 

## Create ML Environment for Web Service

- `az provider register -n Microsoft.MachineLearningCompute`
    
- `az provider register -n Microsoft.ContainerRegistry`
    
- `az provider register -n Microsoft.ContainerService`

- `az group create -n MachineLearningWebServices -l eastus2`

- `az ml env setup --cluster -l eastus2 -n mlapicluster -g MachineLearningWebServices`

- `az ml env show -g MachineLearningWebServices -n mlapicluster`

## Walkthrough

- Download data file `https://aka.ms/socialdata`

- Load data using `dprep` package and `azure storage`

- Use multiple algorithms `LogisticRegression` and `RandomForest` to improve accuracy

- Evaulate models using confusion matrix and f1 score

- Generate job run graphs to track accuracy

- Save the model in `socialads.pkl` file

- Create `schema.json` for web service

- Create the scoring web service code and test it locally

- Deploy model

    - `az ml account modelmanagement set -n mlexprimentsModelMgmt -g MachineLearningHackfest`

    - `az ml env set -n mlapicluster -g MachineLearningWebServices`

    - Browse to `http://localhost:<port>/ui` to see the kubernetes cluster dashboard

        - `http://localhost:<port>/api/v1/namespaces/kube-system/services/kubernetes-dashboard/proxy/#!/deployment?namespace=default`

    - `az ml service create realtime -n jvservice --model-file socialads.pkl -f score.py -r python -s schema.json`

- Test the Deployed Model

    - `az ml service run realtime -i jvservice.mlapicluster-b7655778.eastus2 -d "{\"inputData\": [[20, 10000]]}"`

