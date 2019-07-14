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
from keras.layers import LeakyReLU,concatenate,subtract
from keras.layers import Input,Dense, Dropout, Activation,LSTM,GRU,Conv2D,Flatten,MaxPooling2D,BatchNormalization
from keras.regularizers import l1,l2,l1_l2
from keras_vggface.vggface import VGGFace




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

vgg_features1 = VGGFace(include_top=False, input_shape=(height,width, 3), pooling='avg')
vgg_features2 = VGGFace(include_top=False, input_shape=(height,width, 3), pooling='avg')


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

combined = subtract([model1.output,model2.output])

# Defining Output operations
final = Dense(1024,activation='relu')(combined)
final = BatchNormalization()(final)
final = Dropout(.5)(final)
final = Dense(512,activation='relu')(final)
final = BatchNormalization()(final)
final = Dropout(.5)(final)
final = Dense(256,activation='relu')(final)
final = BatchNormalization()(final)
final = Dropout(.5)(final)
final = Dense(128,activation='relu')(final)
final = BatchNormalization()(final)
final = Dropout(.5)(final)
final = Dense(64,activation='relu')(final)
final = BatchNormalization()(final)
final = Dropout(.5)(final)
final = Dense(2, activation="sigmoid")(final)


model = Model(inputs=[model1.input,model2.input], outputs=final)
 

# Compile the model 
optim = Adam(lr = 0.001)
model.compile(loss='mse', optimizer=optim,metrics=['acc','mse'])




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

print(prediction[:,1])

#targets = prediction[:,1] 
#targets[targets > .5] = 1
#targets[targets < .5] = 0

#testdf['is_related'] = targets
testdf['is_related'] = np.round(prediction[:,1],4)



testdf.to_csv("foo.csv",index=False)


