# Azure Machine Learning end to end Walkthrough

![AML Concepts](https://raw.githubusercontent.com/jomit/AITrials/master/aml/img/hierarchy.png)
[Azure Machine Learning Concepts](https://docs.microsoft.com/en-us/azure/machine-learning/preview/overview-general-concepts)

## Prerequisites

##### Local

- Install [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation#install-azure-machine-learning-workbench-on-windows)

- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- Install [Docker Community Edition](https://www.docker.com/community-edition#/download)

- Install [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)

##### On a Virtual Machine

- Create [Data Science Virtual Machine - Windows 2016](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.windows-data-science-vm)  with size `D4S_v3`

- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- Download and Update [Azure Machine Learning Workbench](https://aka.ms/azureml-wb-msi)

- Install [Docker for Windows](https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe)

- Install [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)

## Experimentation

#### Create Experimentation Account in Azure

- `az group create -n mlgroup -l eastus2`

- `az storage account create -n mlexpstorejomit -g mlgroup --sku Standard_LRS -l eastus2`

- `az storage account show -n mlexpstorejomit -g mlgroup --query "id"`

- Run `setpath.bat` to set the path on any command line or open the command line from AML Workbench

- `az ml account experimentation create --n mlexperiments --resource-group mlgroup -l eastus2 --seats 2 --storage <storage account id>`

- `az ml workspace create --name mlworkspace --account mlexperiments --resource-group mlgroup -l eastus2`

#### Create New Project

- `az ml project create --name startupfunding --workspace mlworkspace --account mlexperiments --resource-group mlgroup --template -l eastus2 --path .`

- Open the `startupfunding` folder in Visual Studio Code

#### Create Model

- Replace content of `train.py` and `score.py` files from the `startupfunding` folder

- Replace all files in `aml_config` folder with files in `startupfunding\aml_config` folder

- Upload the `startupfunding\startups.csv` file in Azure Blob Storage

- Update the blob account details in `train.py` under `loaddata` method

- `cd startupfunding`

- `az ml experiment submit -c local train.py`

- Uncomment `addmodelcomparison` method call on `line 140` in `train.py` and submit the experiment using above command again

- Open the project in Azure Machine Learning Workbench and see the accuracy graph under `Runs\All Runs`

- `az ml history list -o table`

#### Save the Model

- Uncomment `savemodel` method call on `line 143` in `train.py`

- `az ml experiment submit -c local train.py`

- Note the `RunId`

- Open the storage account created above `mlexpstorejomit` in Storage Explorer or in Portal

- Browse to `azureml\ExperimentRun\<RunId>\outputs` folder, you should see `startupfunding.pkl` file 


## Train Models on Remote Environments

#### Create and Test docker image locally

- `docker images`  (Make sure docker is running)

- `az ml experiment prepare -c docker`  (this would take some time..)

- `az ml experiment submit -c docker train.py`

#### Create new Ubuntu Data Science Virtual Machine

- `az group create -n mlvmgroup -l eastus2`

- `az group deployment create -g mlvmgroup --template-uri https://raw.githubusercontent.com/Azure/DataScienceVM/master/Scripts/CreateDSVM/Ubuntu/azuredeploy.json --parameters remotevm-cpu.json`

- `az vm show -g mlvmgroup -n mltrainingvm -d --query "publicIps"`

- `az vm show -g mlvmgroup -n mltrainingvm -d --query "fqdns"`

#### Deploy and Run docker image on Ubuntu DSVM

- `az ml computetarget attach remotedocker --name remotedsvm --address "<IP Address or FQDN>" --username "jomit" --password "<password>"`

- See 2 new files `remotedsvm.computer` and `remotedsvm.runconfig` created under `aml_config`

- Update the `Framework=Python` in `remotedsvm.runconfig`

- `az ml experiment prepare -c remotedsvm`

- SSH into the DSVM and run `sudo docker images` to verify that our docker images have been deployed

- `az ml experiment submit -c remotedsvm train.py`

#### Training Models on Kubernetes Cluster with GPU's

- See instructions [here](https://github.com/jomit/ACSTrials/tree/master/Kubernetes/GPU-Cluster)

#### Training Models on Azure Batch AI

- TODO

## Model Management

![Model Management Workflow](https://raw.githubusercontent.com/jomit/AITrials/master/aml/img/modelmanagementworkflow.png)

#### Select the Model to publish

- `az ml history list -o table`

- Note the `RunId` from the table with best `Accuracy`

- `az ml history info --run "<run id>" --artifact driver_log`

- `az ml history promote --run "<runid>" --artifact-path outputs/startupfunding.pkl --name startupfunding.pkl`

- See the `assets/startupfunding.pkl.link`

- `az ml asset download --link-file assets\startupfunding.pkl.link -d .`

- (Optional) You can also do this from AML Workbench Job Runs UI 

#### Create the Swagger schema for web service

- Uncomment `createwebserviceschema` method call on `line 146` in `train.py`

- `az ml experiment submit -c local train.py`

- Note the `RunId`

- Open the storage account created above `mlexpstorejomit` in Storage Explorer or in Portal

- Browse to `azureml\ExperimentRun\<RunId>\outputs` folder, you should see `schema.json` file, download the file

#### Test web service code locally

- `az ml experiment submit -c local score.py`

#### Create kubernetes cluster to deploy web service

- `az ml env setup --cluster -l eastus2 -n mlcluster -g mlgroup`

- `az ml env show -n mlcluster -g mlgroup`

- Wait until `Provisioning State` is `Succeeded`

- `az ml env set -n mlcluster -g mlgroup`

- Browse to `http://localhost:<port>/ui` to see the kubernetes cluster dashboard

#### Create Model Management Account

- `az ml account modelmanagement create -n mlmodelmgmt -g mlgroup -l eastus2`

- `az ml account modelmanagement set -n mlmodelmgmt -g mlgroup`

#### Create and deploy the web service

- `az ml service create realtime -n fundingservice --model-file startupfunding.pkl -f score.py -r python -s schema.json`

- `az ml service show realtime -n fundingservice -v`

- `az ml service show realtime -n fundingservice -v --query [].Id`

- `az ml service keys realtime -i <Id>`  (Copy Primary Key)

- `az ml service usage realtime -i <Id>`  (Copy Scoring Url)

- (Optional) You can also view service keys and scoring url details on `Model Management` UI in Azure

#### Test the Web Service

- `az ml service run realtime -i <Id> -d "{\"inputData\" : [[0,1,75000,10000,15000]]}"`

- (Optional) You can also use tools like Postman or Fiddler and submit a POST request to verify

#### View Web Service Logs

- `az ml service logs realtime -i <Id>`


## Collecting data from Web Service for Retraining 

- Copy `startupfunding\scoreandcollect.py` file to your project folder

- `pip install azureml.datacollector`

- `az ml service create realtime -n fundingservicev2 --model-file startupfunding.pkl -f scoreandcollect.py -r python -s schema.json --collect-model-data true`

- `az ml service run realtime -i <Id> -d "{\"inputData\" : [[0,1,75000,10000,15000]]}"`

- Run the above command few times with different values

- `az ml env show -v --query storage_account`  (Get the model storage account)

- You should see the data in `inputs` and `prediction` folders under `/modeldata/<subscription_id>/<resource_group_name>/<model_management_account_name>/<webservice_name>/<model_id>-<model_name>-<model_version>`

![Kubernetes Dashboard](https://raw.githubusercontent.com/jomit/AITrials/master/aml/img/k8sdashboard.png)

## Additional Resources

- scikit-learn [Documentation](http://scikit-learn.org/0.18/modules/classes.html)
- Configure password-less sudoers access to run aml workbench on remote hosts ([link](https://docs.microsoft.com/en-us/azure/machine-learning/preview/known-issues-and-troubleshooting-guide#remove-vm-execution-error-no-tty-present))