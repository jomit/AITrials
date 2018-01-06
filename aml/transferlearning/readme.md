# Transfer Learning using ResNet50

## Open Workbench CLI and install following packages before running the models:

- `pip install tensorflow`
- `pip install keras`
- `pip install h5py`
- `pip install pillow`

## Run the experiment

- `az ml experiment submit -c local resnet50-custom.py`


# Additional Resources

## Use GPU on Windows 10

- Install [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)

- Signup and download [cudnn 6](https://developer.nvidia.com/rdp/cudnn-download)

- Extract the cudnn 6 files in the toolkit directory - "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0"

- Install Anaconda and setup new environment

- `conda create -n tensorflow-gpu python=3.5.2`
- `activate tensorflow-gpu`
- `pip install tensorflow-gpu`
- `>>>python`
- `>>>import tensorflow as tf`
- `>>>sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))`


