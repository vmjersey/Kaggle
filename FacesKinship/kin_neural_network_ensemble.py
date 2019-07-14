#!/usr/bin/env python
import numpy as np
import pandas as pd
import timeit
import copy 
import gc
import scipy as sp
import cv2
import os
import random
import sys






# Definitions
submissions_dir = "/home/james/Kaggle/FacesKinship/submissions/"
vgg16file = submissions_dir + "VGGFACE-16_197x197.csv"
resnetfile = submissions_dir + "VGGFACE-resnet_197x197.csv"



vgg16df  = pd.read_csv(vgg16file)
resnetdf = pd.read_csv(resnetfile)

vgg16 = vgg16df['is_related'].to_numpy()
resnet = resnetdf['is_related'].to_numpy()

maximum = np.maximum(vgg16,resnet)
minimum = np.minimum(vgg16,resnet)

print(maximum)
print(minimum)
combined = np.ones((2,maximum.shape[0]))

combined[0] = maximum
combined[1] = maximum

average = np.average(combined,axis=0)
print(average)



testdf = pd.read_csv("/home/james/data/FacesKinship/" + "sample_submission.csv")


testdf.drop(['is_related'], axis=1)
testdf['is_related'] = maximum

testdf.to_csv("maximum.csv",index=False)



testdf.drop(['is_related'], axis=1)
testdf['is_related'] = minimum


testdf.to_csv("minimum.csv",index=False)



testdf.drop(['is_related'], axis=1)
testdf['is_related'] = average


testdf.to_csv("average.csv",index=False)



