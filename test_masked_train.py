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
#from compare import *
import math

def patch_filter_indices(rows=28, cols=28, radius=4, pad_width=0, stride=4):
    # padding the edges 
    # data = np.pad(data, mode='edge', pad_width=pad_width) # pad_width -> search radius

    # Parameters
    R = radius  # Radius
    M1 = rows
    N1 = cols
    patch_x_moves = math.floor((N1 - R)/stride) + 1
    patch_y_moves = math.floor((M1 - R)/stride) + 1

    # Linear indices for the starting R x R block
    idx1 = np.arange(R)[:,None]*N1 + np.arange(R)

    # Offset (from the starting block indices) linear indices for all the blocks
    idx2 = np.arange(patch_y_moves)[:,None]*N1*stride + np.arange(patch_x_moves)*stride

    # Finally, get the linear indices for all blocks
    all_idx = idx1.ravel()[None,None,:] + idx2[:,:,None] 

    # Index into padded for the final output
    # out = padded.ravel()[all_idx] 
    
    return all_idx

def debug_filter(images, mask, rows=28, cols=28):
    
    for i in range(images.shape[1]):
        image = images[:,i]
        
        overlay = np.zeros(mask[:,i].shape)
        feature_image = np.repeat(overlay[..., np.newaxis], 3, -1)
        feature_image[:,0] = mask[:,i]
        feature_image = np.reshape(feature_image,(rows,cols,3))
        feature_image = Image.fromarray((feature_image * 255).astype(np.uint8))
        
        image_ = np.reshape(image,(rows,cols)).T
        image_ = Image.fromarray((image_ * 255).astype(np.uint8))

        plt.imshow(image_)
        plt.imshow(feature_image, 'RdBu', interpolation='nearest', alpha=0.6)
        plt.ion() # turn on interactive mode
        plt.show()
        plt.pause(0.01)
        plt.clf()
    
def train_patched_data(X_train, T_train, X_test, T_test, all_idx, radius=4, stride=4,rows=28, cols=28, debug=False, train=False):
    train_len = X_train.shape[1]
    test_len = X_test.shape[1]
    
    data = []
    # Since, padding = 0, Row len = math.floor((rows-radius)/stride+1) and Col len = math.floor((cols-radius)/stride+1)
    for row in range(0,math.floor((rows-radius)/stride+1)):
        for col in range(0,math.floor((cols-radius)/stride+1)):
            mask = np.zeros(rows*cols)
            mask[all_idx[row,col]] = 1
            train_mask = np.repeat(mask[..., np.newaxis],train_len, -1)
            test_mask = np.repeat(mask[..., np.newaxis],test_len, -1)

            curr_x_train = X_train*train_mask
            curr_x_test = X_test*test_mask

            if debug == True:
                debug_filter(X_train, train_mask, rows=28, cols=28)
            
            # Book keeping for understanding the distribution of the weights
            normalized_avg_patch_sum = np.sum(curr_x_train)/(train_len*radius*radius)
            data.append(normalized_avg_patch_sum)
            
            if train == True: 
                # Check the distribution of the sum of pathches. Based on that the threshold is decided to choose around 300 features
                if normalized_avg_patch_sum > 0.05:
                    print("******The sum of the elements in the patch is: ", np.sum(curr_x_train))
                    # Now train SSFN with NMP here

                    # CNN model example
                    '''
                    num_classes = 10
                    input_shape = (28, 28, 1)
                    model = CNNModel(num_classes, input_shape)
                    print(model.run_cnn_inference(curr_x_train.T, T_train.T, curr_x_test.T, T_test.T))
                    '''
                else:
                    print("Elements sum in training patch is less than threshold....skipping training!")
    avg = np.array(data)  
    #plt.imshow(np.reshape(avg,(rows-radius+1, cols-radius+1)), cmap='viridis', interpolation='nearest')
    plt.plot(avg)
    plt.show(block=True)

def return_patched_data(X_train, X_test, row_ind, col_ind, radius=3, rows=28, cols=28):
    patch_size = radius
    # All flattened masks indices
    all_idx = patch_filter_indices(rows=28, cols=28, radius=patch_size)

    train_len = X_train.shape[1]
    test_len = X_test.shape[1]
    
    # Since, stride = 0 and padding = 0, Row len = rows-2 and Col len = cols - 2
    mask = np.zeros(rows*cols)
    mask[all_idx[row_ind,col_ind]] = 1
    train_mask = np.repeat(mask[..., np.newaxis],train_len, -1)
    test_mask = np.repeat(mask[..., np.newaxis],test_len, -1)

    curr_x_train = X_train*train_mask
    curr_x_test = X_test*test_mask

    return curr_x_train, curr_x_test

def main():
    X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
    X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
    T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
    T_test =  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)

    num_classes = 10
    input_shape = (28, 28, 1)
    patch_size = 4
    stride = 4

    # All flattened masks indices
    all_idx = patch_filter_indices(rows=28, cols=28, radius=patch_size, stride=stride)

    # Selected only 10 train images. For this the train data size will expand to patch_x_moves*patch_y_moves*10 in each feature patch selector
    # patch_x_moves = math.floor((rows - patch_size)/stride) + 1
    # patch_y_moves = math.floor((cols - patch_size)/stride) + 1
    train_size = 10
    train_patched_data (X_train[:,0:train_size], T_train[:,0:train_size], X_test, T_test, all_idx, radius=patch_size, stride=stride, debug=True, train=False)
    
if __name__ == '__main__':
    main()