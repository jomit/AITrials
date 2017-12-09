# Training Clusters on Demand with Batch AI

- `git clone https://github.com/Azure/BatchAI`

- `az account clear`

- `az login`

- `az account set -s VamanbBatchAI`

- `az configure -d group=mlads`
- `az configure -d location=eastus`

- `cd recipes\CNTK\CNTK-GPU-Python\`

- `az batchai job create --cluster-name nc6 -n jomit -c job.json`

- `az batchai job list-files -n jomit -d stdouterr`

- `az batchai job list -o table`

- `az batchai job stream-file -j jomit -d stdouterr -n stdouterr.txt`

- `az batchai cluster list-nodes -n nc6 -o table`


## Helpers

- https://github.com/Azure/BatchAI

- https://azure.microsoft.com/en-us/blog/linux-fuse-adapter-for-blob-storage/

- * Low priority / Dedicated VM's in Batch

- https://github.com/Azure/AMLBatchAISample

