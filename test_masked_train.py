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

def debug_filter(images, mask, rows=28, cols=28):
    
    for i in range(images.shape[1]):
        image = images[:,i]
        
        overlay = np.zeros(image.shape)
        feature_image = np.repeat(overlay[..., np.newaxis], 3, -1)
        feature_image[:,0] = mask[:,i]
        feature_image = np.reshape(feature_image,(rows,cols,3))
        feature_image = Image.fromarray((feature_image * 255).astype(np.uint8))
        
        image_ = np.reshape(image,(rows,cols)).T
        image_ = Image.fromarray((image_ * 255).astype(np.uint8))

        plt.imshow(image_)
        plt.imshow(feature_image, 'viridis', interpolation='nearest', alpha=0.6)
        plt.ion() # turn on interactive mode
        plt.show()
        plt.pause(0.01)
        plt.clf()
    
def train_patched_data(X_train, T_train, X_test, T_test, all_idx, rows=28, cols=28, debug=False):
    train_len = X_train.shape[1]
    test_len = X_test.shape[1]
    
    # Since, stride = 0 and padding = 0, Row len = rows-2 and Col len = cols - 2
    for row in range(0,rows-2):
        for col in range(0,cols-2):
            mask = np.zeros(rows*cols)
            mask[all_idx[row,col]] = 1
            train_mask = np.repeat(mask[..., np.newaxis],train_len, -1)
            test_mask = np.repeat(mask[..., np.newaxis],test_len, -1)

            curr_x_train = X_train*train_mask
            curr_x_test = X_test*test_mask

            if debug == True:
                debug_filter(X_train, train_mask, rows=28, cols=28)
            
            # Check if at least one element is other than zero, maybe define more efficient ways to skip arbitrary training
            if np.sum(curr_x_train) > 0.0:
                print("******The sum of the elements in the patch is: ", np.sum(curr_x_train))
                # Now train SSFN with NMP here

                # CNN model example
                num_classes = 10
                input_shape = (28, 28, 1)
                model = CNNModel(num_classes, input_shape)
                print(model.run_cnn_inference(curr_x_train.T, T_train.T, curr_x_test.T, T_test.T))
            else:
                print("All elements in training are zero....skipping training!")


X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
T_test =  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)

num_classes = 10
input_shape = (28, 28, 1)

# All flattened masks indices
all_idx = patch_filter_indices(rows=28, cols=28, radius=3)

# Selected only 100 train images. For this the train data size will expand to 26x26x100 in each feature patch selector
train_size = 100
train_patched_data (X_train[:,0:train_size], T_train[:,0:train_size], X_test, T_test, all_idx, debug=True)