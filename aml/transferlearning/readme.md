# Image Recognition + Transfer Learning using ResNet

Deep [Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/) have become the default choice for Image Recognition since [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).  One of the newer techniques for Image Recognition is to reuse an existing model (a.k.a. [Transfer Learning](http://cs231n.github.io/transfer-learning/)) as a starting point which can save a ton of time & resources needed to train a CNN from scratch. 

This repo contains a sample code walkthrough for this technique, which creates a custom CNN model using [Micrsoft ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) to recognize two sets of images. For this example I have used baseball vs. cricket bats but you can replace this with any images (e.g. good product vs defected product, full shelf vs empty shelf, happy face vs sad face…) and create your custom model. You many need to do some serious fine tuning or may be start from a different model but the core technique remains the same.

- [ResNet research paper](https://arxiv.org/pdf/1512.03385v1.pdf) by Microsoft
- [Resnet github repo](https://github.com/KaimingHe/deep-residual-networks)

## Run Experiment Locally

#### Open Workbench CLI and install following packages:

- `pip install tensorflow` or `pip install tensorflow-gpu` (see instructions below to setup GPU on Windows 10)
- `pip install keras`
- `pip install h5py`
- `pip install pillow`

#### Run the experiment

- `az ml experiment submit -c local resnet50-custom.py`


## Run Experiment on a GPU VM

- `TODO`

## Run Experiment at scale on a GPU Cluster

- `TODO`

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


