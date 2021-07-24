# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:57:08 2021

@author: abc
"""

from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
import os
from keras.applications.vgg16 import VGG16

#define model
vggmodel = VGG16()

#Check the model how it looks like
vggmodel.summary()

#Create new model
newmodel = Sequential()

#Add vgg model into newmodel
for i, layer in enumerate(vggmodel.layers):
    if i<19:  #Only up to 19th layer to include feature extraction only
        newmodel.add(layer)
        
newmodel.summary()

#if you want to update or train your model or not
for layer in newmodel.layers:
    layer.trainable=False
    
   
#Give the path of our images
path = "images/"

#Rescale our images 
train_datagen = ImageDataGenerator(rescale=1. /255)  #(Rescale it between 0 and 1)

#Train our images
train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=32, class_mode=None)

#Convert this RGB images into LAB format 
X = []
Y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:,:,0])
        Y.append(lab[:,:,1:] / 128) #A and B values range from -127 to 128
        #so we divide the values by 128 to restrict values to between -1 and 1.
    except:
        print("error")
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1, )) #dimensions to be same for X and Y
print(X.shape)
print(Y.shape)

     
#Convert our LAB images size into VGG16 format
vggfeatures = []
for i, sample in enumerate(X):
    sample = gray2rgb(sample)
    sample = sample.reshape((1, 224, 224, 3))
    prediction = newmodel.predict(sample)
    prediction = prediction.reshape((7,7,512))
    vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)

#Decoder
model = Sequential()

model.add(Conv2D(256, (3,3), activation="relu", padding="same", input_shape=(7,7,512)))
model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(16, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(2, (3,3), activation="tanh", padding="same"))
model.add(UpSampling2D((2,2)))
model.summary()


#Compile and fit our model
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.fit(vggfeatures, Y, verbose=1, epochs=10, batch_size=128)

#Save the model
model.save("Colorize_autoencoder_VGG16.model")

#Predicting using saved model
model = tf.keras.models.load_model('Colorize_autoencoder_VGG16.model',
                                   custom_objects=None,
                                   compile=True)

testpath = "images/test_images"
files = os.listdir(testpath)
for idx, file in enumerate(files):
    test = img_to_array(load_img(testpath+file))
    test = resize(test, (224, 224), anti_aliasing=True)
    test*= 1.0/255
    lab = rgb2lab(test)
    l = lab[:,:,0]
    L = gray2rgb(l)
    L = L.reshape((1, 224, 224, 3))
    #print(L.shape)
    vggpred = newmodel.predict(L)
    ab = model.predict(vggpred)
    #print(ab.shape)
    ab = ab*128
    cur = np.zeros((224, 224, 3))
    cur[:, :, 0] = l
    cur[:, :, 1:] = ab
    imsave("vgg_result/result/"+str(idx)+".jpg", lab2rgb(cur))



    

