from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
import numpy as np
import sys
sys.path.append('/home/semen/AircraftsRecognition/deep-learning-models-master')

import glob
from imagenet_utils import preprocess_input
from keras.preprocessing import image

#model = load_model('first_try1.h5') 


def printPrediction(img_path, prefix=""):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(prefix + "   " +  str(preds) + "    " + img_path)
    return preds[0]

   
printPrediction('/home/semen/images_new/validation/f16/f16_1 (148).jpg')
printPrediction('/home/semen/images_new/validation/f15/f15_1 (5).jpg')
print()

#0 = f16
#1 = f15

def EvalAllFilesFrom(path, prefix = "", maxFiles = 10):
    files =  glob.glob(path + "/*.*")
    print(prefix + ":  ")
    classesNum = 2
    s = np.zeros((min(len(files), maxFiles), classesNum), dtype = float)
    k = 0
    for file in files:
        e1 = printPrediction(file, prefix)
        s[k] = e1
        k = k + 1
        if k >= maxFiles:
            break
    
    print(prefix + " average  " + str(sum(s[:,0]) / k) + "   " + str(sum(s[:,1]) / k))


#EvalAllFilesFrom("/home/semen/AircraftsRecognition/images/realImages/f15real", "f15 real", maxFiles = 50)
#EvalAllFilesFrom("/home/semen/AircraftsRecognition/images/realImages/f16real", "f16 real", maxFiles = 50)
EvalAllFilesFrom("/home/semen/images_new/validation/f15", "f15 model", maxFiles = 20)
EvalAllFilesFrom("/home/semen/images_new/validation/f16", "f16 model", maxFiles = 20)
EvalAllFilesFrom("/home/semen/images_new/train/f15", "f15 model", maxFiles = 20)
EvalAllFilesFrom("/home/semen/images_new/train/f16", "f16 model", maxFiles = 20)


