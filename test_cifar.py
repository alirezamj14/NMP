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


def show_image_cifar(samples, sorted_ind, save_name):
    result_path = "./results/"
    image = {}
    percentage = [0.2,0.4,0.6,0.8,1] # [0.2,0.4,0.6,0.8,1] # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    percentage_label = [0.08,0.16,0.24,0.32,.40]
    fig = plt.figure()
    gs1 = gridspec.GridSpec(len(samples), len(percentage))
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.

    i = 0
    feature_len = len(sorted_ind)
    for x in samples:
        idx = 0
        for p in percentage:
            y = np.zeros((1024))
            temp = int(round(p*feature_len))
            index = sorted_ind[0:temp]
            
            y[index] = 1 # Identify the feature positions
            feature = np.reshape(y, (32,32))
            feature = np.transpose(feature)
            #feature[feature>0] = 1

            sample = np.reshape(x, (32,32,3))
            #sample = np.transpose(sample)
            #sample[sample>0] = 1
            zero = np.zeros(sample.shape)

            dummy_RGB_image = sample
            feature_image = zero
            feature_image[:,:,0] = feature
            #dummy_RGB_image[:,:,1] = sample
            #dummy_RGB_image[:,:,2] = sample
            
            image = Image.fromarray((dummy_RGB_image * 255).astype(np.uint8))
            f_image = Image.fromarray((feature_image * 255).astype(np.uint8))
            ax = plt.subplot(gs1[i])
            ax.set_aspect('equal')
            # ax = fig.add_subplot(3, len(percentage), i)
            params = {'interpolation': 'nearest'}
            imgplot = plt.imshow(image)
            plt.imshow(f_image, 'viridis', interpolation='nearest', alpha=0.5)
            ax.set_title(str(int(percentage_label[idx] * 100))+"%")
            plt.axis('off')
            i = i + 1
            idx = idx + 1

    plt.savefig(result_path +"Sample_image_CIFAR_"+save_name+".png")
    plt.close()

def choose_patched_features(size, rows, cols, patch_rows = 4, patch_cols = 4, img_size=32):
    S_hat = []
    for i in range(size):
        for j in range(patch_rows):
            for k in range(patch_cols):
                S_hat.append((patch_rows*rows[i] + j)*img_size + patch_cols*cols[i] + k%patch_cols) 
    return S_hat

# Just to compare what global features SHAP with DeepLift choose
X_train_ori =  loadmat("./mat_files/CIFAR-10.mat")["train_x"].astype(np.float32)
x = np.array([X_train_ori[:,4],
              X_train_ori[:,15],
              X_train_ori[:,3]])

rows = [4, 4, 0, 4, 7, 2, 5, 2, 7, 0, 7, 7, 5, 6, 4, 1, 2, 4, 0, 2, 2, 1, 5, 3, 6, 4]
cols = [2, 6, 7, 4, 5, 1, 6, 0, 7, 5, 1, 4, 2, 4, 1, 4, 7, 3, 4, 2, 6, 5, 7, 7, 6, 0]

#S_hat = choose_patched_features(49, patch_rows = 4, patch_cols = 4)
S_hat = choose_patched_features(26, rows, cols, patch_rows = 4, patch_cols = 4)

show_image_cifar(x, S_hat[0:300], "test_nmp_cnn_300_4x4")

