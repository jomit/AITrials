# Train Tensorflow Models at Scale with Kubernetes on Azure (workshop [repo](https://github.com/wbuchwalter/tensorflow-k8s-azure))

## Create docker image and push it to docker hub

- `cd mnist-src`

- `docker login`

- `docker build -t jomit/tf-mnist .`

- `docker run -it jomit/tf-mnist --max-steps 100`

- `docker push jomit/tf-mnist`

## Create k8s cluster on Azure

- `az group create --name k8s-tensorflow --location westus`

- `az acs create --agent-vm-size Standard_D2_v2 --resource-group k8s-tensorflow --name jack8s --orchestrator-type Kubernetes --agent-count 2 --location westus --generate-ssh-keys`

- `az acs kubernetes get-credentials --name jack8s --resource-group k8s-tensorflow`

- `kubectl get nodes`

## Deploy mnist training model

- `kubectl create -f mnist-training.yaml`

- `kubectl get job`   (wait until the Successful has value 1)

## Helm

- Download `helm` for Windows : https://github.com/kubernetes/helm/releases

- `helm init`

- `helm init --upgrade`

- `helm install stable/wordpress`  (using default settings)

    - It will output the command to get the External IP and Credentials automatically
    - `kubectl get svc --namespace default -w mangy-gerbil-wordpress`
    - User => user
    - Password => `kubectl get secret --namespace default mangy-gerbil-wordpress -o jsonpath="{.data.wordpress-password}"`

- `helm install --name my-wordpress --set wordpressUsername=admin,wordpressPassword=password,mariadb.mariadbRootPassword=secretpassword stable/wordpress`  (using custom settings)

- `helm list`

- `helm delete <release name>`

- `helm install --name firstwiki --set dokuwikiWikiName="Jomit MLADS" stable/dokuwiki`

- `helm delete firstwiki`

## Create custom Helm Chart

- `helm create jomitchart`

- `heml lint jomitchart`  (to check errors)

- `cd jomitchart`

- `helm install . --name jomitchartrelease`

- `helm delete jomitchartrelease`


## Using [tensorflow/k8s](https://github.com/tensorflow/k8s)

- Install using helm

    `helm install https://storage.googleapis.com/tf-on-k8s-dogfood-releases/latest/tf-job-operator-chart-latest.tgz -n tf-job --wait --replace --set cloud=azure`

    `kubectl describe configmaps tf-job-operator-config`

- Create Simple TfJob

    `kubectl create -f mnist-tfjob.yaml`

    `kubectl get tfjob`

    `kubectl get job`

    `kubectl get pod`

    `kubectl logs <pod name>`

    `kubectl delete tfjob mnist-tfjob`


## Persist the model and logs on Azure Files

- Create an Azure Storage account (with Files)

- Create a new FileShare named `tensorflow`

- Update the `azure-secret.yaml` with the base64 encoded string of Storage Account Name and Storage Account Key

- `kubectl create -f azure-secret.yaml`

- `kubectl get secrets`

- `kubectl create -f mnist-training-volumes.yaml`

- `kubectl get pods -w`


## Adding Tensorboard

- `kubectl create -f mnist-tensorboard.yaml`

- `kubectl get services`

- Browse to the EXTERNAL-IP of the tensorboard service.










