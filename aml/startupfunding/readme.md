# Prerequisites

- AML Workbench

- [VS Code Tools for AI](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)


# Create Experiments and Models

## Create Azure resources 

- Azure Portal

- ARM Template run

## Comparing Multiple Models

- Show "run_logger.log("Accuracy", accuracy)"

- setpath.bat

- az ml experiment submit -c local funding.py

- See the graph in "All run" for Accuracy

- az ml history list -o table

## Saving the Model

- See "pickle.dump" section

- Show the models saved in Azure Blob Storage


# Train Models on Remote Environments

## Create docker image

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









