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

model = load_model('first_try.h5') 



def printPrediction(img_path, prefix=""):
    img = image.load_img(img_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict_proba(x)
    #print(prefix + "   " +  str(preds) + "    " + img_path)
    return preds[0,0]

   
printPrediction('/home/semen/images/images_val/f16/a00.jpg')
printPrediction('/home/semen/images/images_val/f15/a00.jpg')
print()

#0 = f16
#1 = f15

def EvalAllFilesFrom(path, prefix = "", maxFiles = 10):
    files =  glob.glob(path + "/*.*")
    print(prefix + ":  ")
    s = [0] * min(len(files), maxFiles)
    k = 0
    for file in files:
        e = printPrediction(file, prefix)
        s[k] = e
        k = k + 1
        if k >= maxFiles:
            break
    
    print(prefix + " average  " + str(sum(s) / k))


EvalAllFilesFrom("/home/semen/AircraftsRecognition/images/realImages/f15real", "f15 real", 100)
EvalAllFilesFrom("/home/semen/AircraftsRecognition/images/realImages/f16real", "f16 real", 100)
EvalAllFilesFrom("/home/semen/images/images_val/f15", "f15 model", 100)
EvalAllFilesFrom("/home/semen/images/images_val/f16", "f16 model", 100)



