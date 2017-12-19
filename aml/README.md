# Azure Machine Learning end to end Walkthrough

![AML Concepts](https://raw.githubusercontent.com/jomit/AITrials/blob/master/img/hierarchy.png)

## Prerequisites

#### Local

- Install [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation#install-azure-machine-learning-workbench-on-windows)

- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- Install [Docker Community Edition](https://www.docker.com/community-edition#/download)

- Install [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)

#### On a Virtual Machine

- Create [Data Science Virtual Machine - Windows 2016](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.windows-data-science-vm)  with size `D4S_v3`

- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

- Download and Update [Azure Machine Learning Workbench](https://aka.ms/azureml-wb-msi)

- Install [Docker for Windows](https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe)

- Install [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)

## Experimentation

#### Create Experimentation Account in Azure

`az group create -n mlgroup -l eastus2`

`az storage account create -n mlexpstorejomit -g mlgroup --sku Standard_LRS -l eastus2`

`az storage account show -n mlexpstorejomit -g mlgroup --query "id"`

Run `setpath.bat` to set the path on any command line or open the command line from AML Workbench

`az ml account experimentation create --n mlexperiments --resource-group mlgroup -l eastus2 --seats 2 --storage <storage account id>`

`az ml workspace create --name mlworkspace --account mlexperiments --resource-group mlgroup -l eastus2`

#### Create New Project

`az ml project create --name startupfunding --workspace mlworkspace --account mlexperiments --resource-group mlgroup --template -l eastus2 --path .`

Open the `startupfunding` folder in Visual Studio Code

#### Create Model

### Compare Multiple Models

- Show "run_logger.log("Accuracy", accuracy)"

- setpath.bat

- az ml experiment submit -c local funding.py

- See the graph in "All run" for Accuracy

- az ml history list -o table

## Serialize the Model

- See "pickle.dump" section

- Show the models saved in Azure Blob Storage


## Train Models on Remote Environments

## Create docker image for 

- az ml experiment prepare -c docker

- az ml experiment submit -c docker funding.py

- Change "Framework = Python" and "PrepareEnvironment = true" in "docker.runconfig"

- az ml experiment submit -c docker funding.py

## Setup remove environments

- Create new Linux DSVM

- az ml computetarget attach remotedocker --name "azuredsvm" --address "<IP Address>" --username "<username>" --password "<password>"

- az ml experiment prepare -c azuredsvm

- az ml experiment submit -c azuredsvm funding.py


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