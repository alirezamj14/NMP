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

X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
T_test =  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)

num_classes = 10
input_shape = (28, 28, 1)
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