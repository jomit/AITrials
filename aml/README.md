# Execute ML experient in a Docker container on a remote machine

`az login` 

`az ml computetarget attach --name azurevm --address <IP address> --username <username> --password <password> --type remotedocker`

`az ml experiment prepare -c azurevm`

`az ml experiment submit -c azurevm .\iris_sklearn.py`


# Resources
	- scikit-learn Documentation
        - http://scikit-learn.org/0.18/modules/classes.html
    - Configure password-less sudoers access to run aml workbench on remote hosts
        - https://docs.microsoft.com/en-us/azure/machine-learning/preview/known-issues-and-troubleshooting-guide#remove-vm-execution-error-no-tty-present