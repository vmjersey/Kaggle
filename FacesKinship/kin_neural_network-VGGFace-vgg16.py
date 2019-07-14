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


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def image_augment(image):
    
    random_int = randint(0,8)

    if random_int == 0 or random_int == 1 or random_int == 2 or random_int == 3:
        # Introduce some noise
        sigma = randint(0, 50)
        temp_image = np.float64(np.copy(image))
        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * sigma

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:,:,0] = temp_image[:,:,0] + noise
            noisy_image[:,:,1] = temp_image[:,:,1] + noise
            noisy_image[:,:,2] = temp_image[:,:,2] + noise  
        
        image = noisy_image

    
    return image

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

            changed_image1 = image_augment(cv_image1)
            changed_image2 = image_augment(cv_image2)
            
            #changed_image1 = cv_image1
            #changed_image2 = cv_image2
 
            # Add to array
            x1[counter] = changed_image1/255
            x2[counter] = changed_image2/255
            
            counter+=1

        #y_binary = to_categorical(batchy)

        return [x1,x2],batchy



# Definitions
data_dir = "/home/james/data/FacesKinship/"
train_dir = data_dir + "train/"

modeltype = "cnn"
batch_size=20
width=197
height=197

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

x_train, x_test, y_train, y_test = train_test_split(xdf,ydf,test_size=0.20, random_state=42,shuffle=True)

x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


train_steps = int(x_train.shape[0]/batch_size)
valid_steps = int(x_test.shape[0]/batch_size)


train_gen = KinImageReader(xdf,ydf,batch_size,height,width)
valid_gen = KinImageReader(x_test,y_test,batch_size,height,width)



vgg_features1 = VGGFace(include_top=False, model='vgg16')
vgg_features2 = VGGFace(include_top=False, model='vgg16')


# Define Pretrained VGG16 Application from Keras
model1 = vgg_features1
model2 = vgg_features2

# Append label for each head so merge layer doesnt get confused
for layer in model1.layers:
    layer.name = layer.name + str("_1")
    #layer.trainable = False
    layer.trainable=True

for layer in model2.layers:
    layer.name = layer.name + str("_2")
    #layer.trainable = False
    layer.trainable = True


x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(model1.output), GlobalAvgPool2D()(model1.output)])
x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(model2.output), GlobalAvgPool2D()(model2.output)])

x3 = Subtract()([x1, x2])
x3 = Multiply()([x3, x3])

x1_ = Multiply()([x1, x1])
x2_ = Multiply()([x2, x2])
x4 = Subtract()([x1_, x2_])
x = Concatenate(axis=-1)([x4, x3])

x = Dense(100, activation="relu")(x)
x = Dropout(0.4)(x)
final = Dense(1, activation="sigmoid")(x)

    
model = Model(inputs=[model1.input,model2.input], outputs=final)
 


model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

# Print basic summary
model.summary()


# Create nice plot of what model looks like
from keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
from IPython.display import Image
Image(retina=True, filename='model.png')


# Define where to write model checkpoints
best_model = data_dir + modeltype + '_weights_best.hdf5'


reduce_on_plateau = ReduceLROnPlateau(monitor="acc", mode="max", factor=0.1, patience=20, verbose=1)

# Define checkpoint object
checkpoint = ModelCheckpoint(best_model, 
                             monitor="acc",
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

callbacks_list = [checkpoint, reduce_on_plateau]


class_weight = {0: 2.5,
                1: 1.}


model.load_weights(best_model)

history = model.fit_generator(train_gen,
                    steps_per_epoch=train_steps, 
                    epochs=15,
                    verbose=1,
                    shuffle=True,
                    max_queue_size=400,
                    #class_weight=class_weight,
                    #initial_epoch=3,
                    #pickle_safe=True,
                    use_multiprocessing=False,    
                    workers=1,         
                    callbacks=callbacks_list
                   )


model.load_weights(best_model)

test = model.evaluate_generator(valid_gen,valid_steps,verbose=1)
print("Validation: ",test)
