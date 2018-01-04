import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

## Can also clone this repo : https://github.com/fchollet/deep-learning-models
## for imagente_utils and resnet50 source code


# Restnet with all layers
model = ResNet50(include_top=True,weights='imagenet')

# See all layers of the resent50 network
model.summary()

# See the configuration of the last layer
model.layers[-1].get_config()

# load the image
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess the input
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict the results
preds = model.predict(x)
print('Predicted:', decode_predictions(preds))