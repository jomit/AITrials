## Create DSVM's for all participants

- `cd armtemplates`

- `az group create --name MachineLearningHackfest --location westus`

- `az network vnet create -n MLVNet -g MachineLearningHackfest -l westus --subnet-name default`

- Update `count` in the `parameters.json` file

- `az group deployment create -n CreateDSVM -g MachineLearningHackfest --template-file template.json --parameters @parameters.json`

- Install Docker for Windows : `https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe`

- Install AML Workbench : `https://aka.ms/azureml-wb-msi` (TODO)


## Create AML Experimentation Account

- `az storage account create -n mlexpstore0 -g MachineLearningHackfest --sku Standard_LRS -l westcentralus`

- `az storage account show -n mlexpstore0 -g MachineLearningHackfest --query "id"`

- `az ml account experimentation create --n mlexperiment0 -g MachineLearningHackfest -l westcentralus --seats 2 --storage <storage account id>`