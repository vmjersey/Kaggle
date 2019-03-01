#!/usr/bin/env python
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "mxnet"
import sys
import keras as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Add
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import to_categorical


def image_loader(num_samples,train_labels,batch_size,height,width):
    # Keep feeding data into the fitting processes
    L = num_samples

    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            X = load_images(batch_size,height,width)
            
            # Shape data channels_first
            X = K.utils.to_channels_first(X)
            Y = train_labels[batch_start:limit]

            yield (X,Y) #a tuple with two numpy arrays with batch_size sample

            batch_start += batch_size   
            batch_end += batch_size


def load_images(batch_size,height,width):
    # Normally I would be reading images and doing manipulations
    # But here I am just going to generate random data in the shape of the images I am using.
    
    images = np.array([])
    #Preallocate the array.  This is much faster that numpy.append
    images = np.zeros((batch_size,height,width,3))
    i=0
    for file in range(batch_size):
        # Create a numpy array that looks like and image
        image = np.random.randint(0,255, size=(128,128,3))
        # Add it to numpy array
        images[i] = image         
        i+=1
    return images

height=128
width=128



dropout = 0.5

num_samples = 228249

integer_encoded = np.random.randint(0,5005, size=228249)
train_labels = K.utils.to_categorical(integer_encoded)



# Get number of classes
num_classes = train_labels.shape[1]
input_shape=(height,width,1)
reshape_size = height * width


input_shape=(3,height,width)

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Dropout(dropout))

model.add(Conv2D(64, (2, 2), activation='relu',padding='same'))
model.add(Dropout(dropout))

model.add(Conv2D(32, (2, 2), activation='relu',padding='same'))
model.add(Dropout(dropout))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(dropout))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=["accuracy"]
             )



epochs=10
batch_size=60

history = model.fit_generator(
                    image_loader(
                        num_samples,
                        train_labels, 
                        batch_size, 
                        height, 
                        width),                   
                    steps_per_epoch=int(num_samples/batch_size),  
                    epochs=epochs,
                    max_queue_size=350,
                    use_multiprocessing=True,
                    workers=2,
                    verbose=1,
                    shuffle=True)




