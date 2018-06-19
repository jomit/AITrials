import numpy as np
import os
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def loadImages():

	# get current working directory path
	folder = 'data'  # bats dataset

	dataroot = os.getcwd() + '/' + folder
	directoryList = os.listdir(dataroot)

	imageList =[]

	# loop over all the subdirectories and create the image array of all the images and store it in an array
	for categoryDir in directoryList:
		currentImageList = os.listdir(dataroot +'/'+ categoryDir)
		print ('Loading images for '+'{}\n'.format(categoryDir))
		for currentImage in currentImageList:
			imagePath = dataroot + '/'+ categoryDir + '/'+ currentImage 
			currentImage = image.load_img(imagePath, target_size=(224, 224))
			x = image.img_to_array(currentImage)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			print('Image shape:', x.shape)
			imageList.append(x)


	# reshaping the data as needed by the resnet50 model i.e. (<observations>, 224, 224, 3)
	imageData = np.array(imageList)
	print(imageData.shape)  # it should show (40,1, 224, 224, 3)

	imageData=np.rollaxis(imageData,1,0)
	print (imageData.shape) # it should show (1, 40, 224, 224, 3) 

	imageData=imageData[0]
	print (imageData.shape) # now it should show (40, 224, 224, 3) , which is exactly the shape that we need

	return imageData

def labelImages(imageData, categories):
	# Note : this can also be done by using specific folder structure)
	totalImageSamples = imageData.shape[0]
	labels = np.ones((totalImageSamples,),dtype='int64')

	# hardcoding labels for first 20 and last 20 images (can be done dynamically if needed)

	labels[0:20]=0   # baseball bats
	labels[20:40]=1  # cricket bats

	# convert labels to one-hot encoding as they are categorical
	Y = np_utils.to_categorical(labels, categories)

	# shuffle the dataset
	x, y = shuffle(imageData,Y, random_state=0)

	return x,y

def getResnetModel():

	imageInput = Input(shape=(224, 224, 3))

	# get the resnet50 model and weights
	# https://keras.io/applications/#resnet50
	model = ResNet50(input_tensor=imageInput, include_top=False,weights='imagenet')
	#model.summary()

	return model

def fineTuneModel(model, categories):

	# get last layer of the model
	lastLayer = model.output

	# adding global spatial average pooling layer
	x = GlobalAveragePooling2D()(lastLayer)

	# fine tuning the network by adding 2 fully-connected & dropout layers
	x = Dense(512, activation='relu',name='connected-layer-1')(x)
	x = Dropout(0.5)(x)

	x = Dense(256, activation='relu',name='connected-layer-2')(x)
	x = Dropout(0.5)(x)

	# adding last output layer for 2 categories/classes
	output = Dense(categories, activation='softmax',name='output_layer')(x)

	return output

def createCustomModel(model, output):

	# create our custom model using the inputs and outputs from the above resnet50 updated model
	customModel = Model(inputs=model.input, outputs=output)
	customModel.summary()

	# mark all layers as not trainable, except the 6 that we added last for fine tuning 
	for currentLayer in customModel.layers[:-6]:
		currentLayer.trainable = False
		
	#print(customModel.layers[-7].trainable)
	#print(customModel.layers[-1].trainable)

	customModel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return customModel

def savemodel(model):

    # create the outputs folder
    os.makedirs('./outputs', exist_ok=True)

    # serialize the model
    print ("Saved the model => startupfunding.pkl")
    f = open('./outputs/startupfunding.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

if __name__ == '__main__':
	categories = 2  # baseball bat & cricket bat

	# Load Images from a folder 
	imageData = loadImages()

	# Label the images 
	x, y = labelImages(imageData,categories)

	# Split the dataset for training/testing   (TODO: add validation set)
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	# Get ResNet model
	resnetModel = getResnetModel()

	# Fine tune the model
	output = fineTuneModel(resnetModel, categories)

    # Build Custom Model using resnet model
	customModel = createCustomModel(resnetModel, output)

	# Train custom model
	#startTime = time.time()
	iterations = 50  # changes this to 2 or 5 for quickly testing the code or while using CPU
	hist = customModel.fit(X_train, y_train, batch_size=10, epochs=iterations, validation_data=(X_test, y_test))
	#print('Training time: %s' % (startTime - time.time()))

	# Evaluate custom model
	(loss, accuracy) = customModel.evaluate(X_test, y_test, batch_size=10, verbose=1)

	print("EVALUATION RESULT\n")
	print("LOSS => {:.4f}, ACCURACY => {:.4f}%".format(loss,accuracy * 100))