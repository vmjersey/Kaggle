#!/usr/bin/env python
import cv2
import numpy as np
import os
import pandas
import sys
import matplotlib.pyplot as plt
import keras as K
from keras.datasets import mnist
from keras.layers.merge import concatenate
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
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
import scipy


backend = "tensorflow"

if backend =='cntk':
    import cntk
    cntk.device.try_set_default_device(cntk.device.gpu(0)) 

def get_labels(filelist,labels):
    
    labels.index = labels['Image']
    select_labels = np.array([])
    labels_filenames = []
    for file in filelist:
        select_labels = np.append(select_labels,labels.loc[file]['Encoded_Id'])
        labels_filenames.append(file)
    
    return select_labels,labels_filenames


def encode_labels(labels_df):
    labels = labels_df['Id'].values
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_encode = le.transform(labels)    
    labels_df['Encoded_Id'] = labels_encode 
    return labels_df


def load_images(filenames,directory,height,width):
    # Read Images into a Numpy array
    # Get the array size
    array_size = len(filenames)
    images = np.array([])
    #Preallocate the array.  This is much faster that numpy.append
    if backend =='mxnet':
        images = np.zeros((array_size,height,width,3))
        channel = 1
    else:
        images = np.zeros((array_size,height,width,3))
        channel = 1

    i=0
    for file in filenames:
        
        image = cv2.imread(directory + "/" + file,channel)
        # Resize the image.
        resized_image = cv2.resize(image, (width,height))
        # Add it to numpy array
        images[i] = resized_image
        i+=1

    return images


def image_loader(files, labels, batch_size, directory, height, width):

    L = len(files)

    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            X = load_images(files[batch_start:limit],directory,height,width)
            
            # Shape data depending on backend
            if backend =='mxnet':
                X = K.utils.to_channels_first(X)
            else: 
                X = X.reshape((X.shape[0],height,width,3))
            
            Y = labels[batch_start:limit]

            yield (X,Y) #a tuple with two numpy arrays with batch_size sample

            batch_start += batch_size   
            batch_end += batch_size


height=128
width=128

traindir="/home/james/Kaggle/Whale/Data/train/augment"
labelsfile="/home/james/Kaggle/Whale/Data/train.csv"


labels = pandas.read_csv(labelsfile)
#labels.describe()
#with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(labels['Id'].value_counts())

labels = pandas.read_csv(labelsfile)
y=labels['Id']
values = np.array(y)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(values),
                                                 values)

#Create Encoded Labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
train_labels = K.utils.to_categorical(integer_encoded)



directory = traindir + "/combined"
validdir = traindir + "/valid"

filenames = os.listdir(directory)
filenames.sort()

#how many files before augmentation
num_orig_filenames = len(os.listdir(traindir + "/1"))
num_aug_filenames = len(filenames)
iterations = int(num_aug_filenames / num_orig_filenames)

augmented_labels = np.array([])
for it in range(iterations):
    if augmented_labels.shape[0] == 0:
        augmented_labels = train_labels
    else:
        augmented_labels = np.append(augmented_labels,train_labels,axis=0)



dropout = 0.5
    
# Get number of classes
num_classes = train_labels.shape[1]
input_shape=(height,width,1)
reshape_size = height * width


# Get number of classes
num_classes = train_labels.shape[1]

if backend =='mxnet':
    input_shape=(3,height,width)
else: 
    input_shape=(height,width,3)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Dropout(dropout))

model.add(Conv2D(128, (8, 8), activation='relu',padding='same'))
model.add(Dropout(dropout))

model.add(Conv2D(64, (2, 2), activation='relu',padding='same'))
model.add(Dropout(dropout))

model.add(Conv2D(32, (2, 2), activation='relu',padding='same'))
model.add(Dropout(dropout))
model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(dropout))


model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=["accuracy"]
             )


#model.summary()

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#Save image_models
filepath="/home/james/Kaggle/Whale/Data/whales-copy" + ".best.hdf5"

#del test_images
testdir = "/home/james/Kaggle/Whale/Data/test/4"
testfilenames = os.listdir(testdir)
testfilenames.sort()
testfiles = []

for testfilename in testfilenames:
    testfiles.append(testdir + "/" + testfilename)
    
test_images = load_images(testfiles,"",height,width)
# Reshape Array
test_images = test_images.reshape((test_images.shape[0],height,width,3))


model.load_weights(filepath)
# Make prediction
Predictions = model.predict(test_images)

test_csv = "/home/james/Kaggle/Whale/Data/test.csv"

test = testfilenames
col = ['Image']
test_df = pandas.DataFrame(test, columns=col)
test_df['Id'] = ''

for i, pred in enumerate(Predictions):
    ids = label_encoder.inverse_transform(pred.argsort()[-5:][::-1])
    ids = ids.tolist()
    sorted_idx = pred.argsort()[-5:][::-1]
    sorted_values = []
    for idx in sorted_idx:
        sorted_values.append(pred[idx])

    print(sorted_values)
    
    test_df.loc[i, 'Id'] = ' '.join(ids)

print(test_df.head(10))

test_df.to_csv(test_csv, index=False)
