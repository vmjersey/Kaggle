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
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import concatenate,subtract,add,average
from keras.layers import Input,Dense, Dropout, Activation,LSTM,GRU,Conv2D,Flatten,MaxPooling2D,BatchNormalization
from keras.regularizers import l1,l2,l1_l2
from keras.utils import to_categorical




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

        #print(index,":",start,":",end)
        #print(self.df.shape)
        #print(self.df.index)
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

            f.write(str(index) + ":" + image1+" - "+image2+"\r\n")
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

        y_binary = to_categorical(batchy)

        return [x1,x2],y_binary


# Definitions
data_dir = "/home/james/data/FacesKinship/"
train_dir = data_dir + "train/"

modeltype = "cnn"
batch_size=100
width=64
height=64

f = open("output.txt",'w')

traindf = pd.read_csv(data_dir+"finaltrain.csv",usecols=['p1','p2','related'])
traindf = traindf.drop_duplicates()


print(traindf.shape)

# We want to swap the image places so both heads can be trained on both images
traindf_reverse = pd.DataFrame()
traindf_reverse['p1'] = traindf['p2']
traindf_reverse['p2'] = traindf['p1']
traindf_reverse['related'] = traindf['related']

# Add Back to real array 
traindf = traindf.append(traindf_reverse,ignore_index=True,sort=False)

print(traindf.shape)

# shuffle the dataframe
traindf = traindf.sample(frac=1).reset_index(drop=True)

ydf = pd.DataFrame()
ydf = traindf['related']

xdf = pd.DataFrame()
xdf['p1'] = traindf['p1']
xdf['p2'] = traindf['p2']

x_train, x_test, y_train, y_test = train_test_split(xdf,ydf,test_size=0.33, random_state=42,shuffle=True)

x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


train_steps = int(x_train.shape[0]/batch_size)
valid_steps = int(x_test.shape[0]/batch_size)


train_gen = KinImageReader(x_train,y_train,batch_size,height,width)
valid_gen = KinImageReader(x_test,y_test,batch_size,height,width)

inputA = Input(shape=(height,width,3))
inputB = Input(shape=(height,width,3))




    
l=0.00001

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



from keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
from IPython.display import Image
Image(retina=True, filename='model.png')


best_model = data_dir + modeltype + '_weights_best.hdf5'
current_model = data_dir + modeltype + '_weights_current.hdf5'



checkpoint = ModelCheckpoint(best_model, 
                             monitor="acc",
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')


callbacks_list = [checkpoint]


print("Train Samples: ",x_train.shape[0])
print("Valid Samples: ",x_test.shape[0])

model.load_weights(best_model)

history = model.fit_generator(train_gen,
                    steps_per_epoch=train_steps, 
                    epochs=50,
                    verbose=1,
                    shuffle=True,
                    max_queue_size=500,
                    initial_epoch=5,
                    #pickle_safe=True,
                    use_multiprocessing=False,    
                    workers=1,         
                    callbacks=callbacks_list
                   )


model.save(current_model)
