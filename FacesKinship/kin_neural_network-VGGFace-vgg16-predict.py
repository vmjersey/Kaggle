#!/usr/bin/env python
import numpy as np
import pandas as pd
import keras
import timeit
import copy 
import gc
import scipy as sp
import cv2
import os
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.models import Model
from keras.layers import concatenate,subtract,add,average,Concatenate,GlobalMaxPool2D, GlobalAvgPool2D,Subtract,Multiply
from keras.layers import Input,Dense,Dropout,Activation,LSTM,GRU,Conv2D,Flatten,MaxPooling2D,BatchNormalization,Lambda
from keras import backend as K
from keras.regularizers import l1,l2,l1_l2
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.callbacks import Callback
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from random import randint
import random
import skimage






# Definitions
data_dir = "/home/james/data/FacesKinship/"
train_dir = data_dir + "train/"

modeltype = "cnn"
batch_size=20
width=197
height=197


traindf = pd.read_csv(data_dir+"finaltrain.csv",usecols=['p1','p2','related'])

y = traindf['related']

train_steps = int(traindf.shape[0]/batch_size)

class KinImageReader(keras.utils.Sequence):

    def __init__(self,df,target,batch_size,height,width):
        
        self.df = df
        self.target = target
        self.width=width
        self.height=height
        self.batch_size = batch_size
        self.on_epoch_end()

        
    def __len__(self):
        return int(self.df.shape[0]/self.batch_size)
    
    #def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = np.arange(0,self.y.shape[0])

    def __getitem__(self,index):
        
        start = index*self.batch_size
        end = (index+1)*self.batch_size

        batchy = self.target[start:end]
        image_placeholders = np.arange(start,end)

        x1 = np.array([])
        x1 = np.zeros((len(image_placeholders),self.height,self.width,3))

        x2 = np.array([])
        x2 = np.zeros((len(image_placeholders),self.height,self.width,3))       
        
        counter = 0
        
        for i in image_placeholders:
            
            image1 = self.df.loc[i,'p1']
            image2 = self.df.loc[i,'p2']

            #read the image
            cv_image1 = cv2.imread(image1,1)
            cv_image2 = cv2.imread(image2,1)
            
            #resize the image
            cv_image1 = cv2.resize(cv_image1,(self.width,self.height))
            cv_image2 = cv2.resize(cv_image2,(self.width,self.height))
            
            # Add to array
            x1[counter] = cv_image1/255
            x2[counter] = cv_image2/255
            
            counter+=1

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(batchy)
        encoded_y = encoder.transform(batchy)

        return [x1,x2],encoded_y



train_gen = KinImageReader(traindf,y,batch_size,height,width)


vgg_features1 = VGGFace(include_top=False, model='vgg16')
vgg_features2 = VGGFace(include_top=False, model='vgg16')


# Define Pretrained VGG16 Application from Keras
model1 = vgg_features1
model2 = vgg_features2

# Append label for each head so merge layer doesnt get confused
for layer in model1.layers:
    layer.name = layer.name + str("_1")
    layer.trainable = False
    #layer.trainable=True

for layer in model2.layers:
    layer.name = layer.name + str("_2")
    layer.trainable = False
    #layer.trainable = True


x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(model1.output), GlobalAvgPool2D()(model1.output)])
x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(model2.output), GlobalAvgPool2D()(model2.output)])

x3 = Subtract()([x1, x2])
x3 = Multiply()([x3, x3])

x1_ = Multiply()([x1, x1])
x2_ = Multiply()([x2, x2])
x4 = Subtract()([x1_, x2_])
x = Concatenate(axis=-1)([x4, x3])

x = Dense(100, activation="relu")(x)
x = Dropout(0.01)(x)
final = Dense(1, activation="sigmoid")(x)

    
model = Model(inputs=[model1.input,model2.input], outputs=final)
 


model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

# Compile the model 
#optim = Adam(lr = 0.001)
#model.compile(loss='mse', optimizer=optim,metrics=['acc','mse',auroc])




# Print basic summary
model.summary()


best_model = data_dir + modeltype + '_weights_best.hdf5'
current_model = data_dir + modeltype + '_weights_current.hdf5'


model.load_weights(best_model)


test_dir=data_dir + "test/"

testdf = pd.read_csv(data_dir + "sample_submission.csv")

x1 = np.array([])
x1 = np.zeros((testdf.shape[0],height,width,3))

x2 = np.array([])
x2 = np.zeros((testdf.shape[0],height,width,3))
    
for index,line in testdf.iterrows():
    images = line['img_pair']
    image1,image2 = images.split("-")
    cv_image1 = cv2.imread(test_dir + image1,1)
    cv_image2 = cv2.imread(test_dir + image2,1)
            
    #resize the image
    cv_image1 = cv2.resize(cv_image1,(width,height))
    cv_image2 = cv2.resize(cv_image2,(width,height))
            
    # Add to array
    x1[index] = cv_image1/255
    x2[index] = cv_image2/255



prediction = model.predict([x1,x2])



# Find index of maximum value from 2D numpy array
#result = np.argmax(prediction,axis=1)
testdf.drop(['is_related'], axis=1)

print(prediction)

testdf['is_related'] = np.round(prediction,4)



testdf.to_csv("foo.csv",index=False)


