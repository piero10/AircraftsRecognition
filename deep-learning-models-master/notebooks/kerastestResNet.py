import sys
sys.path.append('/home/semen/aircraftRecognition/deep-learning-models-master')
sys.path.append('/home/semen/AircraftsRecognition/deep-learning-models-master')

from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from sklearn.linear_model import SGDClassifier
sys.setrecursionlimit(10000)

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = ResNet50(weights='imagenet')

#img_path = '/home/semen/aircraftRecognition/deep-learning-models-master/ff09.jpg'#'cat256.jpg'
img_path = '/home/semen/AircraftsRecognition/deep-learning-models-master/ff09.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
model.layers.pop()
