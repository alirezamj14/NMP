import matplotlib.pyplot as plt 
import os
import joblib
import numpy as np
from scipy.io import loadmat
import time
import pickle
import argparse
from sklearn.pipeline import make_pipeline
import random
import pandas as pd
from MyFunctions import *
from compare import *
import tensorflow as tf
from tensorflow.keras import layers

def patch_filter_indices(rows=28, cols=28, radius=3, pad_width=0):
    # padding the edges 
    # data = np.pad(data, mode='edge', pad_width=pad_width) # pad_width -> search radius

    # Parameters
    R = radius  # Radius
    M1 = rows
    N1 = cols
    rowlen = N1 - R + 1
    collen = M1 - R + 1

    # Linear indices for the starting R x R block
    idx1 = np.arange(R)[:,None]*N1 + np.arange(R)

    # Offset (from the starting block indices) linear indices for all the blocks
    idx2 = np.arange(collen)[:,None]*N1 + np.arange(rowlen)

    # Finally, get the linear indices for all blocks
    all_idx = idx1.ravel()[None,None,:] + idx2[:,:,None] 

    # Index into padded for the final output
    # out = padded.ravel()[all_idx] 
    
    return all_idx

def debug_filter(image, indices):
    rows, cols, _ = indices.shape
    plt.ion() # turn on interactive mode
    for row in range(0,rows):
        for col in range(0,cols):
            mask = np.zeros(image.shape)
            
            feature_image = np.repeat(mask[..., np.newaxis], 3, -1)

            mask[indices[row,col]] = 1
            feature_image[:,0] = mask
            
            feature_image = np.reshape(feature_image,(28,28,3))
            feature_image = Image.fromarray((feature_image * 255).astype(np.uint8))

            image_ = image.T
            image_ = np.reshape(image_,(28,28))
            image_ = Image.fromarray((image_ * 255).astype(np.uint8))

            plt.imshow(image_)
            plt.imshow(feature_image, 'viridis', interpolation='nearest', alpha=0.6)
            plt.show()
            _ = input("Press [enter] to continue.")

X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
T_test =  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)

num_classes = 10
input_shape = (28, 28, 1)

# All flattened masks indices
all_idx = patch_filter_indices(rows=28, cols=28, radius=3)

debug_filter(X_train.T[20,:], all_idx)

'''
model = CNNModel(num_classes, input_shape)
# Mask at a particular row(feature) for the training and test data
i = 94
train_mask = np.zeros(X_train.shape)
test_mask = np.zeros(X_test.shape)
train_mask[i,:] = 1
test_mask[i,:] = 1
X_train = X_train*train_mask
X_test = X_test*test_mask

print(model.run_cnn_inference(X_train.T, T_train.T, X_test.T, T_test.T))
'''