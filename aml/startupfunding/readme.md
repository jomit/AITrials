# End to end demo script

## Prep

- start `mltrainingvm`
- start docker on local machine 
- create k8s cluster

    `az group create -n mlk8sgroup -l eastus2`

	`az ml env setup --cluster -l eastus2 -n mlcluster -g mlk8sgroup`

	`az ml env show -n mlcluster -g mlk8sgroup`

## Demo

- Explain the code in train.py

- Update the blob storage details and run the experiment locally

    `ACCOUNT_NAME = "mlgputraining"`

    `ACCOUNT_KEY = "<key>"`

    `CONTAINER_NAME = "datasets"`

    `az ml experiment submit -c local train.py`

- Show .pkl file in blob storage

- Create new computetarget

    `az ml runconfiguration list`

    `az ml computetarget list`

    login to the mltrainingvm vm to show the docker container creation

    `az ml computetarget attach remotedocker --name trainingvm --address "mltrainingvm.eastus2.cloudapp.azure.com" --username "jomit" --password "<password>"`

    `az ml experiment prepare -c trainingvm`

- Train experiment on the remote vm

    `az ml experiment submit -c local train.py`

- Explain the code in score.py, train.py (createwebserviceschema)

- Create kubernetes cluster to deploy the model as a service

    `az ml env list`

    `** az ml env setup --cluster -l eastus2 -n mlcluster -g mlk8sgroup`

    `az ml env set -n mlcluster -g mlk8sgroup`

    `** open the localhost:<port >url `

    `http://localhost:1642/api/v1/namespaces/kube-system/services/kubernetes-dashboard/proxy/#!/deployment?namespace=default`

- Set Model Management

    show model management UI in portal

    `az ml account modelmanagement set -n mlmodelmgmt -g mlgroup`

- Create the webservice

    `az ml service create realtime -n <myservice> --model-file startupfunding.pkl -f score.py -r python -s schema.json`

    show k8s dashboard (Deployments)

    show model management UI (Models, Manifests, Images, Services) in portal

    `az ml service show realtime -n <myservice> -v`

- Test the webservice

    `az ml service usage realtime -i <Id>`

    `az ml service run realtime -i <Id> -d "{\"inputData\" : [[0,1,75000,10000,15000]]}"`

    get the URL and Primary Key of the service from the model management UI or run the below command

    `az ml service keys realtime -i <Id>`

    Open Postman and show the REST API call using the bearer token