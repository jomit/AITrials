# Sign Language Gesture Recognition on the Intelligent Edge using Azure Cognitive Services

For this walkthrough we will use an Android Phone/Tablet as the Intelligent Edge device. The goal is to show how we can quickly create image recognition models using [Custom Vision Service](https://www.customvision.ai/) and export it to consume it offline at the Edge.

![Sign Language](https://raw.githubusercontent.com/jomit/AITrials/master/sign-language-recognition/signs.png)

#### Setup

- [Install Android Studio](https://developer.android.com/studio/index.html)

- Download `dataset.zip` file and extract it
    - Original dataset can be found [here](https://www.kaggle.com/datamunge/sign-language-mnist/version/1)

#### Create Sign Language Recognition ML Model

- Sigin to [Custom Vision Service](https://www.customvision.ai/) using your Azure Account

- Create New Project with Domains as **General (compact)**

- Upload all images from **dataset\A** folder with Tag **A**

- ***Repeat** above step for all alphabets in the dataset...*

- Click the **Train** button at top to start training the model

- Once the training is complete use the **Quick Test** button to upload a new image and test it.

#### Export the ML Model

- Under **Performance** tab click **Export**

- Select **Android (Tensorflow)** and download the zip file

- Extract the zip file and verify that it contains **model.pb** and **labels.txt** file


#### Create the Android App

- Clone `https://github.com/Azure-Samples/cognitive-services-android-customvision-sample` repo as a template

- Replace both **model.pb** and **labels.txt** files in `cognitive-services-android-customvision-sample\app\src\main\assets\`

- Open the project in Android Studio

- *Make any updates to UI / Labels as necessary*

#### Deploy the Android App on the device

- First enable Developer Mode + USB Debugging on the Android device
    - See instructions for Samsung Galaxy S7 [here](https://www.androidcentral.com/how-enable-developer-mode-galaxy-s7)

- Connect your device to laptop via USB

- Click **Run** and select the **app**

- Select the **Connected Device**

    - For first time you need to allow the camera and other permissions and run it again.


#### Testing

