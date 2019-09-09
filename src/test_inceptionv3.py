#from resnet50 import ResNet50
from inception_v3 import InceptionV3
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np

model = InceptionV3(weights='imagenet')

img_path = 'f117_sky0000.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
