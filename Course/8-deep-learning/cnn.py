# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#### Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential();

# Step 1 - Create the first convolutional Layer

# convolutional layer = multiple feature maps
# filters = 32 = number of features maps to create (start with 32 for CPU, can use 64 or 128 )
# kernel_size = 3 = number of rows for the feature detector table, which is used to create the feature map
# kernel_size = 3 = number of columns for the feature detector table, which is used to create the feature map
# input_shape = 64 = pixel dimension to use for the input (can use 128 or 256 for GPU)
# input_shape = 64 = pixel dimension to use for the input (can use 128 or 256 for GPU)
# input_shape = 3 = number of channels which is 3 for colored images and 2 for black & white images

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# Step 2 - Pooling  (reducing the size of feature map using maxpooling)
# pool_size = 2 X 2 to filter the feature maps

classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer to improve the results
#classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening   (create single vector so that we can use it as input for ANN)
classifier.add(Flatten())
    

# Step 4 - Full connection  (add the layers for artifical neural network)

# Create hidden layer
# units = nodes in the hidden layer. 128 is based on experimentation, typically use number around 100 and power of 2 
classifier.add(Dense(units = 128, activation = 'relu'))

# Create output layer
# units = nodes in the output layer  (we want binary probability output cat or dog and hence just 1 output is needed)
# activation = sigmoid as we want probability as output
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Compiling the CNN
# optimize = algorithm to optimize the weights, i.e. stocastic gradient algorithm
# loss = loss function within the stocastic grading algorithm, "binary_crossentropy" or "categorical_crossentropy"
# metrics = criterion used to evaluate the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#### Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# Doc - https://keras.io/preprocessing/image/

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# target_size = 64, 64 = dimensions of input_shape in our CNN
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# steps_per_epoch = number of images in our training set = 8000
# epochs = training epocs for CNN
# validation_steps = numner of images in our test set = 2000

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=2, validation_data=test_set, validation_steps=2000)
















