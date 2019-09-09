import sys 
sys.path.append('/home/semen/AircraftsRecognition/deep-learning-models-master') 
 
from resnet50 import ResNet50 
from keras.preprocessing import image 
from imagenet_utils import preprocess_input, decode_predictions 
import numpy as np 
from sklearn.linear_model import SGDClassifier 
import glob, os 
 
from keras.models import Sequential, Model 
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense, Input 
from keras.optimizers import SGD 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from keras.preprocessing import image 
sys.setrecursionlimit(10000) 
from sklearn.model_selection import KFold 
from keras.models import load_model 
 
 
 
base_model = ResNet50(weights='imagenet', include_top=False) 
     
x = base_model.output 
x = GlobalAveragePooling2D()(x) 
x = Dense(2048, activation='relu')(x) 
predictions = Dense(1, activation='softmax')(x) 
     
model = Model(input=base_model.inputs, output=predictions) 
     
print(len(base_model.layers)) 
    # freeze first layers 
for layerNum in range(len(base_model.layers)): #175 
    model.layers[layerNum].trainable = False 
         
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy']) 
    #model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy']) 
    #model.compile(loss='categorical_crossentropy', optimizer= 'adam'#SGD(lr=0.01, momentum=0.9) 
    #, metrics=['accuracy']) 
    
 
 
 
#fitting 
#-------------------------------------------------------------- 
     
def FIT(epochNum = 10):   
    train_datagen = ImageDataGenerator( 
            #rotation_range=10., 
            rescale=1./255, 
            shear_range=0.2, 
            zoom_range=0.2, 
            horizontal_flip=True) 
     
    # this is the augmentation configuration we will use for testing: 
    # only rescaling 
    test_datagen = ImageDataGenerator(rescale=1./255,              
            shear_range=0.2, 
            zoom_range=0.2, 
            horizontal_flip=True) 
     
    # this is a generator that will read pictures found in 
    # subfolers of 'data/train', and indefinitely generate 
    # batches of augmented image data 
    train_generator = train_datagen.flow_from_directory( 
            '/home/semen/images_new/train',  # this is the target directory 
            target_size=(224, 224),  # all images will be resized to ... 
            batch_size=32, 
            class_mode='binary', 
            shuffle=True)  # since we use binary_crossentropy loss, we need binary labels 
     
    # this is a similar generator, for validation data 
    validation_generator = test_datagen.flow_from_directory( 
            '/home/semen/images_new/validation', 
            target_size=(224, 224), 
            batch_size=32, 
            class_mode='binary', 
            shuffle=True) 
             
             
             
    model.fit_generator( 
            train_generator, 
            samples_per_epoch=500, 
            nb_epoch=epochNum, 
            validation_data = validation_generator, 
            nb_val_samples=100) 
             
     
 
 
 
 
 
def printPrediction(img_path): 
    img = image.load_img(img_path, target_size=(224, 224)) 
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x) 
    preds = model.predict(x) 
    print(preds) 
     
  
#model = load_model('first_try1.h5')     
for layerNum in range(175): 
    model.layers[layerNum].trainable = False 
FIT(epochNum = 5)     
 
    
printPrediction('/home/semen/images/images_val/f16/a00.jpg') 
printPrediction('/home/semen/images/images_val/f15/a00.jpg') 
#0 = f16 
#1 = f15 
 
 
for layer in model.layers: 
    layer.trainable = True 
     
FIT(epochNum = 5) 
 
model.save('first_try1.h5')  
print('fit done weight saved') 
