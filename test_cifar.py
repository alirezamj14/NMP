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

def show_image_accuracy_cifar(samples, sorted_ind, save_name, test_accuracy):
    result_path = "./results/"
    image = {}
    percentage = [0.2,0.4,0.6,0.8,1] # [0.2,0.4,0.6,0.8,1] # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    percentage_label = [0.08,0.16,0.24,0.32,.40]
    fig = plt.figure()
    
    gs1 = gridspec.GridSpec(len(percentage), len(samples))
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes.
    no_of_samples = len(samples)

    i = 0
    feature_len = len(sorted_ind)
    for p in percentage:
        idx = 0
        for x in samples:
            y = np.zeros((1024))
            temp = int(round(p*feature_len))
            index = sorted_ind[0:temp]
            
            y[index] = 1 # Identify the feature positions
            feature = np.reshape(y, (32,32))
            feature = np.transpose(feature)
            sample = x
            zero = np.zeros(sample.shape)

            dummy_RGB_image = x
            feature_image = zero
            feature_image[:,:,0] = feature

            output = (1*dummy_RGB_image + 0.8*feature_image)
            ax = plt.subplot(gs1[i])

            ax.set_aspect('equal')
            plt.imshow(output)
            
            if idx == 0:
                ax.set_title(str(int(percentage_label[i%(no_of_samples-1)] * 100))+"%", x=-0.5,y=0.3, fontsize=20)
            if idx == no_of_samples-1:
                ax.set_title(str(test_accuracy[i%(no_of_samples-1)])+"%", color='blue', x=1.5,y=0.3, fontsize=20)
                #ax.set_title(, x=10,y=0.3)
            
            plt.axis('off')
            i = i + 1
            idx = idx + 1
    plt.show()

def choose_patched_features(size, rows, cols, patch_rows = 4, patch_cols = 4, img_size=32):
    S_hat = []
    for i in range(size):
        for j in range(patch_rows):
            for k in range(patch_cols):
                S_hat.append((patch_rows*rows[i] + j)*img_size + patch_cols*cols[i] + k%patch_cols) 
    return S_hat

# Just to compare what global features SHAP with DeepLift choose
X_original =  loadmat("./mat_files/CIFAR-10.mat")["train_x"].astype(np.float32)
X_rgb = np.reshape(X_original.T, (50000, 3, 32, 32))
X_rgb = np.swapaxes(X_rgb, 1,3)
X_rgb = np.swapaxes(X_rgb, 1,2)

x = np.array([X_rgb[4,...]/255,
              X_rgb[15,...]/255,
              X_rgb[3,...]/255,
              X_rgb[3000,...]/255,
              X_rgb[12321,...]/255,
              X_rgb[4556,...]/255,
              X_rgb[7687,...]/255,
              X_rgb[1231,...]/255,
              X_rgb[456,...]/255,
              X_rgb[8687,...]/255
              ])

rows = [4, 4, 0, 4, 7, 2, 5, 2, 7, 0, 7, 7, 5, 6, 4, 1, 2, 4, 0, 2, 2, 1, 5, 3, 6, 4]
cols = [2, 6, 7, 4, 5, 1, 6, 0, 7, 5, 1, 4, 2, 4, 1, 4, 7, 3, 4, 2, 6, 5, 7, 7, 6, 0]

#S_hat = choose_patched_features(49, patch_rows = 4, patch_cols = 4)
S_hat = choose_patched_features(26, rows, cols, patch_rows = 4, patch_cols = 4)

test_accuracy = [29.64, 33.99, 40,  41.49, 50.45]
show_image_accuracy_cifar(x, S_hat[0:300], "test_nmp_cnn_300_4x4", test_accuracy)
