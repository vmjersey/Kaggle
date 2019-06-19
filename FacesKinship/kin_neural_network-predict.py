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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import LeakyReLU,concatenate
from keras.layers import Input,Dense, Dropout, Activation,LSTM,GRU,Conv2D,Flatten,MaxPooling2D,BatchNormalization
from keras.regularizers import l1,l2,l1_l2


# Definitions
data_dir = "/home/james/data/FacesKinship/"
train_dir = data_dir + "train/"

modeltype = "cnn"
batch_size=100
width=64
height=64


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



inputA = Input(shape=(height,width,3))
inputB = Input(shape=(height,width,3))
    
    
# Data from first part of segment
a = Conv2D(128,kernel_size=3,activation='relu')(inputA)
a = Dropout(.3)(a)
a = BatchNormalization()(a)
a = MaxPooling2D(pool_size=(2, 2))(a)

a = Conv2D(96,kernel_size=3,activation='relu')(a)
a = Dropout(.3)(a)
a = BatchNormalization()(a)

a = Conv2D(64,kernel_size=3,activation='relu')(a)
a = Dropout(.3)(a)
a = BatchNormalization()(a)

a = Conv2D(48,kernel_size=3,activation='relu')(a)
a = Dropout(.3)(a)
a = BatchNormalization()(a)

a = Conv2D(32,kernel_size=3,activation='relu')(a)
a = Dropout(.3)(a)
a = BatchNormalization()(a)

a = Conv2D(16,kernel_size=3,activation='relu')(a)
a = Dropout(.3)(a)
a = Flatten()(a)
a = Model(inputs=inputA,outputs=a)
 
# Data from middle part of segment
b = Conv2D(128,kernel_size=3,activation='relu')(inputB)
b = Dropout(.3)(b)
b = BatchNormalization()(b)
b = MaxPooling2D(pool_size=(2, 2))(b)

b = Conv2D(96,kernel_size=3,activation='relu')(b)
b = Dropout(.3)(b)
b = BatchNormalization()(b)

b = Conv2D(64,kernel_size=3,activation='relu')(b)
b = Dropout(.3)(b)
b = BatchNormalization()(b)

b = Conv2D(48,kernel_size=3,activation='relu')(b)
b = Dropout(.3)(b)
b = BatchNormalization()(b)

b = Conv2D(32,kernel_size=3,activation='relu')(b)
b = Dropout(.3)(b)
b = BatchNormalization()(b)

b = Conv2D(16,kernel_size=3,activation='relu')(b)
b = Dropout(.3)(b)
b = Flatten()(b)
b = Model(inputs=inputB,outputs=b)

combined = concatenate([a.output,b.output])

final = Dense(512,activation='relu')(combined)
final = Dropout(.4)(final)
final = Dense(256,activation='relu')(final)
final = Dropout(.4)(final)
final = Dense(128,activation='relu')(final)
final = Dropout(.4)(final)
final = Dense(64,activation='relu')(final)
final = Dropout(.4)(final)
final = Dense(2, activation="softmax")(final)
    
model = Model(inputs=[a.input,b.input], outputs=final)
  
optim = Adam(lr = 0.0001)

model.compile(loss='categorical_crossentropy', optimizer=optim,metrics=['acc'])
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
print(prediction)



# Find index of maximum value from 2D numpy array
#result = np.argmax(prediction,axis=1)
testdf.drop(['is_related'], axis=1)

print(prediction[:,1])
testdf['is_related'] = np.round(prediction[:,1],4)

testdf.to_csv("foo.csv",index=False)


