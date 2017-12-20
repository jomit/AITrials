# Azure Machine Learning end to end Walkthrough

![AML Concepts](https://raw.githubusercontent.com/jomit/AITrials/master/aml/img/hierarchy.png)
[Azure Machine Learning Concepts](https://docs.microsoft.com/en-us/azure/machine-learning/preview/overview-general-concepts)

## Prerequisites

### Local

- Install [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation#install-azure-machine-learning-workbench-on-windows)

- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- Install [Docker Community Edition](https://www.docker.com/community-edition#/download)

- Install [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)

### On a Virtual Machine

- Create [Data Science Virtual Machine - Windows 2016](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.windows-data-science-vm)  with size `D4S_v3`

- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- Download and Update [Azure Machine Learning Workbench](https://aka.ms/azureml-wb-msi)

- Install [Docker for Windows](https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe)

- Install [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)

## Experimentation

### Create Experimentation Account in Azure

- `az group create -n mlgroup -l eastus2`

- `az storage account create -n mlexpstorejomit -g mlgroup --sku Standard_LRS -l eastus2`

- `az storage account show -n mlexpstorejomit -g mlgroup --query "id"`

- Run `setpath.bat` to set the path on any command line or open the command line from AML Workbench

- `az ml account experimentation create --n mlexperiments --resource-group mlgroup -l eastus2 --seats 2 --storage <storage account id>`

- `az ml workspace create --name mlworkspace --account mlexperiments --resource-group mlgroup -l eastus2`

### Create New Project

- `az ml project create --name startupfunding --workspace mlworkspace --account mlexperiments --resource-group mlgroup --template -l eastus2 --path .`

- Open the `startupfunding` folder in Visual Studio Code

### Create Model

- Replace content of `train.py` and `score.py` files from the `startupfunding` folder

- Replace all files in `aml_config` folder with files in `startupfunding\aml_config` folder

- `az ml experiment submit -c local train.py`

- Uncomment `addmodelcomparison` method call on `line 143` in `train.py` and submit the experiment using above command again

- Open the project in Azure Machine Learning Workbench and see the accuracy graph under `Runs\All Runs`

- `az ml history list -o table`

### Save the Model

- Uncomment `savemodel` method call on `line 146` in `train.py`

- `az ml experiment submit -c local train.py`

- Note the `RunId`

- Open the storage account created above `mlexpstorejomit` in Storage Explorer or in Portal

- Browse to `azureml\ExperimentRun\<RunId>\outputs` folder, you should see `startupfunding.pkl` file 


## Train Models on Remote Environments

### Create and Test docker image locally

- `docker images`  (Make sure docker is running)

- `az ml experiment prepare -c docker`  (this would take some time..)

- `az ml experiment submit -c docker train.py`

### Create new Ubuntu Data Science Virtual Machine

- `az group create -n mlvmgroup -l eastus2`

- `az group deployment create -g mlvmgroup --template-uri https://raw.githubusercontent.com/Azure/DataScienceVM/master/Scripts/CreateDSVM/Ubuntu/azuredeploy.json --parameters remotevm-cpu.json`

- `az vm show -g mlvmgroup -n mltrainingvm -d --query "publicIps"`

- `az vm show -g mlvmgroup -n mltrainingvm -d --query "fqdns"`

### Deploy and Run docker image on Ubuntu DSVM

- `az ml computetarget attach remotedocker --name remotedsvm --address "<IP Address or FQDN>" --username "jomit" --password "<password>"`

- See 2 new files `remotedsvm.computer` and `remotedsvm.runconfig` created under `aml_config`

- Update the `Framework=Python` in `remotedsvm.runconfig`

- `az ml experiment prepare -c remotedsvm`

- SSH into the DSVM and run `sudo docker images` to verify that our docker images have been deployed

- `az ml experiment submit -c remotedsvm train.py`

### Training Models on Kubernetes Cluster with GPU's

- See instructions [here](https://github.com/jomit/ACSTrials/tree/master/Kubernetes/GPU-Cluster)

### Training Models on Azure Batch AI

- TODO

# Deploy Model as a Web service

## Model Selection

- Go to job runs and "Promote" the output model in Workbench UI or
 
- az ml history list -o table

- az ml history promote --run "<runid>" --artifact-path outputs/startupfunding.pkl --name startupfunding.pkl 

- Verify the link by downloading the files

- az ml asset download --link-file assets/startupfunding.pkl.link -d outputs

## Create the Swagger schema for the service

- See "Create schema.json" section

- Download the "schema.json" file from Blob Storage (ExperimentationRun/<RunId>/outputs)

## Create and Test scoring service code

- See "sore.py"

- az ml experiment submit -c local score.py

## Setup k8s cluster to deploy web service

- az ml env setup --cluster -l westcentralus -n acsbootcamp -g AIBootCamp

- az ml env show -n acsbootcamp -g AIBootCamp

- (Wait until provisiong is Succeeded)

- az ml env set -n acsbootcamp -g AIBootCamp

- (Browse to http://localhost:<port>/ui) to see the k8s cluster dashboard

## Deploy the web service

Set the Model Management Account

- az ml account modelmanagement set -n aibootcampModelMgmt -g AIBootCamp

Download the latest model file

- az ml asset download --link-file assets/startupfunding.pkl.link -d .

- az ml service create realtime -n funding --model-file outputs/startupfunding.pkl -f score.py -r python -s schema.json

- az ml service show realtime -n funding -v

- az ml service usage realtime -i funding.acsbootcamp-fc346666.westcentralus

Get the Bearer token from the Azure Model Management Portal

Use Postman to submit a request and test the web service

https://editor.swagger.io


To view logs or delete the service

- az ml service logs realtime -i funding.acsbootcamp-fc346666.westcentralus

- az ml service delete realtime -i funding.acsbootcamp-fc346666.westcentralus


# Additional Resources

- scikit-learn Documentation
    - http://scikit-learn.org/0.18/modules/classes.html
- Configure password-less sudoers access to run aml workbench on remote hosts
    - https://docs.microsoft.com/en-us/azure/machine-learning/preview/known-issues-and-troubleshooting-guide#remove-vm-execution-error-no-tty-present