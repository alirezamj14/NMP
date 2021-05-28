from __future__ import division, print_function
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import os
import joblib
import numpy as np
from scipy.io import loadmat
import time
import pickle
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from MyFunctions import *
import tensorflow as tf
from tensorflow.keras import layers
import shap
from xbart import XBART
from pygam import GAM, s, te
# Lasso, E-Net
from sklearn import linear_model
from sklearn.linear_model import ElasticNet


class CNNModel:
    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def create_cnn_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        return model

    def run_cnn_inference(self, X_train, T_train, X_test, T_test):
        """
            X_train ([float]): [The matrix of training data. Each row contains one sample.]
            X_test ([float]): [The matrix of testing data. Each row contains one sample.]
            T_train ([float]): [The matrix of training target. Each row contains one sample.]
            T_test ([float]): [The matrix of testing target. Each row contains one sample.]
        """
        # Make sure images have shape (28, 28, 1)
        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 28, 28)
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        
        model = self.create_cnn_model()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, T_train, epochs=10, batch_size=128)
        # Sanity checks
        score_train = model.evaluate(X_train, T_train, verbose=0)
        score_test = model.evaluate(X_test, T_test, verbose=0)
        print("Test loss:", score_test[0])
        print("Test accuracy:", score_test[1])
        t_hat_test = model.predict(X_test).reshape(T_test.shape)
        t_hat = model.predict(X_train).reshape(T_train.shape)
        return compute_nme(T_train,t_hat), compute_nme(T_test,t_hat_test),calculate_accuracy(T_train.T,t_hat.T), calculate_accuracy(T_test.T,t_hat_test.T)

def define_parser():
    parser = argparse.ArgumentParser(description="Run NMP")
    parser.add_argument("--reduced_inference", default="True", help="If comparision subset is reduced according to paper")
    parser.add_argument("--J", default="1000", help="Comparision subset")
    parser.add_argument("--feature_percentage", default="100", help="Percentage of features used for performance measure")
    parser.add_argument("--data", default="MNIST", help="Input dataset available as the paper shows")
    parser.add_argument("--algo", default="BART", help="The algorithm used for feature selection")
    parser.add_argument("--tree_size", default="20", help="The number of trees used in BART or RF")
    parser.add_argument("--MC_Num", default="10", help="The number of MC simulations done")
    parser.add_argument("--deeplift_sample_size", default="10", help="The number of samples chosen from deeplift for explaining in each MC simulation")
    args = parser.parse_args()
    return args

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def prepare_data(args):
    """
        X_train ([float]): [The matrix of training data. Each row contains one sample.]
        X_test ([float]): [The matrix of testing data. Each row contains one sample.]
        T_train ([float]): [The matrix of training target. Each row contains one sample.]
        T_test ([float]): [The matrix of testing target. Each row contains one sample.]
    """

    if args.data=="MNIST":
        X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
        X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
        T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
        T_test =  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)
        with open('./parameters/MNIST_sorted_ind.pkl', 'rb') as f:
             indices = pickle.load(f) 

        if args.reduced_inference == "True":
            J_subset = np.random.choice(X_train.shape[1], int(args.J))
            X_train = X_train[:,J_subset]
            T_train = T_train[:,J_subset]

        if args.algo=="RF":
            return indices['sorted_ind'][:300], X_train.T, X_test.T, T_train.T, T_test.T
        elif args.algo=="BART" or args.algo=="GAM":
            T_train = np.asarray([np.argmax(t, axis=None, out=None) for t in T_train.T])/10.0
            T_test = np.asarray([np.argmax(t, axis=None, out=None) for t in T_test.T])/10.0
            return indices['sorted_ind'][:300], X_train.T, X_test.T, T_train, T_test
        else:
            return indices['sorted_ind'][:300], X_train.T, X_test.T, T_train.T, T_test.T

    if args.data=="AIRFOIL":
        X_train = loadmat("./mat_files/AIRFOIL.mat")["X_test"].astype(np.float32)
        T_train = loadmat("./mat_files/AIRFOIL.mat")["T_test"].astype(np.float32)
        X_test = loadmat("./mat_files/AIRFOIL.mat")["X_train"].astype(np.float32)
        T_test= loadmat("./mat_files/AIRFOIL.mat")["T_train"].astype(np.float32)

        X_train = (X_train - X_train.mean(axis=0)) / np.std(X_train, axis=0)
        X_test = (X_test - X_test.mean(axis=0)) / np.std(X_test, axis=0)
        
        S = [0, 1, 2, 3, 4]

        fExtra=500
        X_train = np.concatenate((X_train, np.random.randn(X_train.shape[0],fExtra)), axis=1)
        X_test = np.concatenate((X_test, np.random.randn(X_test.shape[0],fExtra)), axis=1)

        T_train = T_train.reshape(-1,1)
        T_test = T_test.reshape(-1,1)

        if args.algo=="BART":
            return S, X_train, X_test, T_train.flatten(), T_test.flatten()
        else:
            return S, X_train, X_test, T_train, T_test
 
    if args.data=="BOSTON":
        from sklearn.datasets import load_boston
        X, T = load_boston(return_X_y=True)
        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.33, random_state=21)
        S = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        fExtra=100
        X_train = np.concatenate((X_train, 10*np.random.rand(X_train.shape[0],fExtra)+10), axis=1)
        X_test = np.concatenate((X_test, 10*np.random.rand(X_test.shape[0],fExtra)+10), axis=1)
        T_train = T_train.reshape(-1,1)
        T_test = T_test.reshape(-1,1)

        if args.algo=="BART":
            return S, X_train, X_test, T_train.flatten(), T_test.flatten()
        else:
            return S, X_train, X_test, T_train, T_test


    if args.data=="GRAVITATION":
        N=10000         # Number of samples  
        Ntr=9000       # Number of training samples  
        fExtra=7           # Number of extra random features
        m1=10*np.random.rand(1, N)+10
        m2=10*np.random.rand(1, N)+10
        r=10*np.random.rand(1, N)+10
        F=np.multiply(m1, m2)/(r**2)
        S = [0, 1, 2] # set of true relevant features for gravitational law
        X_train = np.concatenate((m1[:,:Ntr], m2[:,:Ntr], r[:,:Ntr], 10*np.random.rand(fExtra,Ntr)+10), axis=0)
        T_train=F[:,:Ntr]
        X_test = np.concatenate((m1[:,Ntr:], m2[:,Ntr:], r[:,Ntr:], 10*np.random.rand(fExtra,N-Ntr)+10), axis=0)
        T_test=F[:,Ntr:]
        
        if args.algo=="BART":
            return S, X_train.T, X_test.T, T_train.T.flatten(), T_test.T.flatten()
        else:
            return S, X_train.T, X_test.T, T_train.T, T_test.T

    if args.data=="ARTIFICIAL":
        # Articial model from paper http://proceedings.mlr.press/v80/ye18b.html

        N = 600             # Number of samples  
        Ntr = 300           # Number of training samples 
        P = 5               # Number of input features
        PExtra=495          # Number of extra random features
        e = np.random.randn(1, N)
        Z = np.random.randn(P+PExtra, N)
        X = (Z + e)/2
        epsilon = np.random.randn(1, 1)
        S = [0, 1, 2, 3, 4]
        X_train = X[:,:Ntr]
        X_test = X[:,Ntr:]

        T_train = (10 * np.sin(np.maximum(X_train[0,:], X_train[1,:])) + (np.maximum(np.maximum(X_train[2,:], X_train[3,:]), X_train[4,:]))**3 )/( 1 + (X_train[0,:] + X_train[4,:])**2 ) \
                + np.sin(0.5 * X_train[2,:]) * (1 + np.exp(X_train[3,:] - 0.5 * X_train[2,:])) \
                + X_train[2,:]**2 + 2 * np.sin(X_train[3,:]) + 2 * X_train[4,:] + epsilon

        T_test = (10 * np.sin(np.maximum(X_test[0,:], X_test[1,:])) + (np.maximum(np.maximum(X_test[2,:], X_test[3,:]), X_test[4,:]))**3 )/( 1 + (X_test[0,:] + X_test[4,:])**2 ) \
                + np.sin(0.5 * X_test[2,:]) * (1 + np.exp(X_test[3,:] - 0.5 * X_test[2,:])) \
                + X_test[2,:]**2 + 2 * np.sin(X_test[3,:]) + 2 * X_test[4,:] + epsilon

        if args.algo=="BART":
            return S, X_train.T, X_test.T, T_train.T.flatten(), T_test.T.flatten()
        else:
            return S, X_train.T, X_test.T, T_train.T, T_test.T

def load_parameters(parameter_file):
    # Getting back the objects:
    with open(parameter_file, 'rb') as f:
        model_avg_card, \
        model_std_card, \
        model_avg_fpsr, \
        model_std_fpsr, \
        model_avg_fnsr, \
        model_std_fnsr, \
        model_avg_msfe, \
        model_std_msfe, \
        model_avg_mspe, \
        model_std_mspe, \
        model_avg_train_nme, \
        model_std_train_nme, \
        model_avg_test_nme, \
        model_std_test_nme = pickle.load(f)

    print ("Cardinality - Mean", model_avg_card) 
    print ("Cardinality - Std", model_std_card) 
    print ("FPSR - Mean", model_avg_fpsr) 
    print ("FPSR - Std", model_std_fpsr) 
    print ("FNSR - Mean", model_avg_fnsr) 
    print ("FNSR - Std", model_std_fnsr) 
    print ("MSFE (Fitting error) - Mean", model_avg_msfe) 
    print ("MSFE (Fitting error) - Std", model_std_msfe) 
    print ("MSPE (Prediction error) - Mean", model_avg_mspe) 
    print ("MSPE (Prediction error) - Std", model_std_mspe)
    print ("NME (Training) - Mean", model_avg_train_nme)
    print ("NME (Training) - Std", model_std_train_nme)
    print ("NME (Testing) - Mean", model_avg_test_nme)
    print ("NME (Testing) - Std", model_std_test_nme)

def save_parameters(parameter_file, model_fpsr, model_fnsr, model_msfe, model_mspe, model_card, model_nme_train, model_nme_test):
    model_avg_fpsr = np.mean(model_fpsr)
    model_std_fpsr = np.std(model_fpsr)
    
    model_avg_fnsr = np.mean(model_fnsr)
    model_std_fnsr = np.std(model_fnsr)
    
    model_avg_msfe = np.mean(model_msfe)
    model_std_msfe = np.std(model_msfe)
    
    model_avg_mspe = np.mean(model_mspe)
    model_std_mspe = np.std(model_mspe)

    model_avg_card = np.mean(model_card)
    model_std_card = np.std(model_card)

    model_avg_train_nme = np.mean(model_nme_train) 
    model_std_train_nme = np.std(model_nme_train)

    model_avg_test_nme = np.mean(model_nme_test) 
    model_std_test_nme = np.std(model_nme_test)
    
    print ("Cardinality - Mean", model_avg_card) 
    print ("Cardinality - Std", model_std_card) 
    print ("FPSR - Mean", model_avg_fpsr) 
    print ("FPSR - Std", model_std_fpsr) 
    print ("FNSR - Mean", model_avg_fnsr) 
    print ("FNSR - Std", model_std_fnsr) 
    print ("MSFE (Fitting error) - Mean", model_avg_msfe) 
    print ("MSFE (Fitting error) - Std", model_std_msfe) 
    print ("MSPE (Prediction error) - Mean", model_avg_mspe) 
    print ("MSPE (Prediction error) - Std", model_std_mspe)
    print ("NME (Training) - Mean", model_avg_train_nme)
    print ("NME (Training) - Std", model_std_train_nme)
    print ("NME (Testing) - Mean", model_avg_test_nme)
    print ("NME (Testing) - Std", model_std_test_nme)

    pickle.dump([model_avg_card,
                model_std_card,
                model_avg_fpsr,
                model_std_fpsr,
                model_avg_fnsr,
                model_std_fnsr,
                model_avg_msfe,
                model_std_msfe,
                model_avg_mspe,
                model_std_mspe,
                model_avg_train_nme,
                model_std_train_nme,
                model_avg_test_nme,
                model_std_test_nme], open(parameter_file, "wb"))

def create_model(args, file_path, model, X_train, T_train):
    if (os.path.exists(file_path)):
        print ("--- Loading from the model ---", args.algo)
        
        if args.algo == "DEEPLIFT":
            model = tf.keras.models.load_model(file_path)
            model.summary()
        else:
            model = joblib.load(file_path)
    else:
        print ("--- Creating the model ---", args.algo)
        
        if args.algo == "DEEPLIFT":
            model.fit(X_train, T_train, epochs=10, batch_size=128)
            model.save(file_path)
        else:
            model.fit(X_train, T_train)
            if args.algo != "BART":
                joblib.dump(model, open(file_path, 'wb'), compress=0)
    return model

def run_feature_selector_algo(args, S, X_train, X_test, T_train, T_test, i, model_fpsr, model_fnsr, model_msfe, model_mspe, model_card, model_nme_train, model_nme_test):
    log_params = False
    file_path_prefix = "./parameters/"
    feature_percentage = args.feature_percentage

    start_time = time.time()
    if args.algo=="RF":      
        file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".joblib"
        
        model = RandomForestRegressor(n_estimators=100) 
        model = create_model(args, file_path, model, X_train, T_train)
        importance_vals = model.feature_importances_
        
        # Choose features which has 1% importance according to paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6660200/ 
        S_hat = np.argwhere(importance_vals > 0.01).flatten()
        
        if args.data=="MNIST" or args.data=="CIFAR-10": # Take 40% features of MNIST only
            file_path = file_path_prefix + args.data + "/" + args.algo + "-" + feature_percentage + "_percent_features-" + str(i) + ".joblib"
            n_sub_feat_size = int(X_train.shape[1]*int(feature_percentage)/100)
            # For RF use this because of the already trained saved model in Sandipan's laptop
            # n_sub_feat_size = 315 
            S_hat = np.argsort(importance_vals)[::-1][:n_sub_feat_size].flatten() #40% features
            model = RandomForestRegressor(n_estimators=100) 
            model = create_model(args, file_path, model, X_train[:,S_hat], T_train)
            X_train = X_train[:,S_hat]
            X_test = X_test[:,S_hat]
        log_params = True

    elif args.algo == "DEEPLIFT": 
        # Implemented using DeepExplain in SHAP: https://github.com/slundberg/shap
        #-------------------------------------------------------------------------#
        x_train = X_train
        x_test = X_test

        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 28, 28)
        # Make sure images have shape (28, 28, 1)
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        print("X_train shape:", X_train.shape)
        print(X_train.shape[0], "train samples")
        print(X_test.shape[0], "test samples")

        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)
        """
        ## Build the model
        """

        model = CNNModel(num_classes, input_shape).create_cnn_model()
        model.summary()
                
        file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".h5"
        
        """
        ## Train the model
        """

        batch_size = 128
        epochs = 15

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model = create_model(args, file_path, model, X_train, T_train)
        
        # Sanity checks
        score_train = model.evaluate(X_train, T_train, verbose=0)
        score_test = model.evaluate(X_test, T_test, verbose=0)
        print("Test loss:", score_test[0])
        print("Test accuracy:", score_test[1]) 

        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        # explain predictions of the model on 10 images
        e = shap.DeepExplainer(model, background)
        
        x_test_sample = X_test[np.random.choice(X_test.shape[0], int(args.deeplift_sample_size), replace=False), :]

        shap_values = e.shap_values(x_test_sample)

        total_val = np.sum(np.sum(np.abs(shap_values), axis=0), axis=0).flatten()
        S_hat = total_val.argsort()[::-1]

        
        if args.data=="MNIST" or args.data=="CIFAR-10": # Take 40% features of MNIST only
            X_train = x_train[:,S_hat]
            X_test = x_test[:,S_hat]
            X_train = X_train.reshape(X_train.shape[0], 28, 28)
            X_test = X_test.reshape(X_test.shape[0], 28, 28)
            # Make sure images have shape (28, 28, 1)
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)
            file_path = file_path_prefix + args.data + "/" + args.algo + "-" + feature_percentage + "percent_features-" + str(i) + ".h5"
            n_sub_feat_size = int(X_train.shape[1]*int(feature_percentage)/100)
            S_hat = total_val.argsort()[::-1][:n_sub_feat_size] #40% features
            model_new = CNNModel(num_classes, input_shape).create_cnn_model()
            model_new.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            model = create_model(args, file_path, model_new, X_train, T_train)
        
        # Just to compare what global features SHAP with DeepLift choose
        # X_train_ori =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
        # show_image([X_train_ori[:,1],X_train_ori[:,20],X_train_ori[:,30]],S_hat[0:len(S)], (args.algo+str(i)))
        
        # show_image(x_train[1,:].flatten(),x_train[20,:].flatten(),x_train[30,:].flatten(),S_hat, (args.algo+str(i)))

        log_params = True

    elif args.algo=="BART":
        # Implemented using XBART: https://github.com/JingyuHe/XBART
        #----------------------------------------------------------#
        x_train = X_train
        x_test = X_test

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        # Ugly hack otherwise xbart fit does not work
        T_train = T_train.flatten()
        T_test = T_test.flatten()

        file_path = file_path_prefix + args.data + "/" + args.algo + str(args.tree_size) + "-" + str(i) + ".joblib"
        # model = XBART(num_trees = int(args.tree_size), num_sweeps = 20, burnin = 15, verbose = True, parallel = True)
        model = XBART(num_trees = int(args.tree_size), num_sweeps = 20, burnin = 15, verbose = True, parallel = True)
        model = create_model(args, file_path, model, X_train, T_train)

        S_hat = sorted(model.importance, key=model.importance.get)[::-1]
        imp_vals = np.array(S_hat)
        S_hat = imp_vals[imp_vals>0.01]
        
        
        if args.data=="MNIST" or args.data=="CIFAR-10": # Take 40% features of MNIST only
            file_path = file_path_prefix + args.data + "/" + args.algo + "-" + feature_percentage + "_percent_features-" + str(i) + ".joblib"
            n_sub_feat_size = int(X_train.shape[1]*int(feature_percentage)/100)
            S_hat = sorted(model.importance, key=model.importance.get)[::-1][:n_sub_feat_size] #40% features
            model = XBART(num_trees = int(args.tree_size), num_sweeps = 20, burnin = 15, verbose = True, parallel = True)
            X_train = pd.DataFrame(x_train[:,S_hat])
            X_test = pd.DataFrame(x_test[:,S_hat])
            model = create_model(args, file_path, model, X_train, T_train)
            

        # Ugly hack otherwise xbart predict does not work
        T_train = T_train.reshape(X_train.shape[0], 1)
        T_test = T_test.reshape(X_test.shape[0], 1)

        log_params = True

    elif args.algo=="POINTNET":
        import torch
        from torch.utils.data import DataLoader
        import kaolin as kal
        from kaolin import ClassificationEngine
        from kaolin.datasets import ModelNet
        from kaolin.models.PointNet import PointNetClassifier as PointNet
        import kaolin.transforms as tfs

        modelnet_path = './mat_files/ModelNet10'
        categories = ['chair', 'sofa']
        num_points = 1024
        device = 'cuda'

        transform = tfs.Compose([
        tfs.TriangleMeshToPointCloud(num_samples=num_points),
        tfs.NormalizePointCloud()
        ])

        train_loader = DataLoader(ModelNet(modelnet_path, 
                                  categories=categories,
                                  split='train', 
                                  transform=transform, 
                                  device=device),
                                  batch_size=12, shuffle=True)


    elif args.algo=="GAM": # Note GAM doesn't work on MNIST properly
        file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".joblib"
        thershold = 0.01
        
        
        gam_fn_form = s(0, n_splines=5)
        for feature in range(1, X_train.shape[1]):
            gam_fn_form += s(feature, n_splines=5)
        # Regression in GAM
        # https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html#Regression
        model = GAM(gam_fn_form, distribution='normal', link='identity', max_iter=10, tol=0.001)
        model = create_model(args, file_path, model, X_train, T_train)
        
        feature_vals = np.array(model.statistics_['p_values'])
        imp_vals = feature_vals[feature_vals > thershold]
        S_hat = np.argsort(imp_vals).flatten()
        
        #S_hat = np.argsort(model.statistics_['p_values'])
        
        log_params = True

    elif args.algo=="LASSO":
        file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".joblib"
        
        thershold = 0.01
        #T_train = np.argmax(T_train, axis=1)
        #T_test = np.argmax(T_test, axis=1)

        model = linear_model.Lasso(alpha=0.01, max_iter=5000)
        model = create_model(args, file_path, model, X_train, T_train)

        imp_vals = model.coef_[model.coef_ > thershold]
        S_hat = np.argsort(imp_vals).flatten()
        if args.data=="MNIST" or args.data=="CIFAR-10": # Take 40% features of MNIST only
            file_path = file_path_prefix + args.data + "/" + args.algo + "-" + feature_percentage + "_percent_features-" + str(i) + ".joblib"
            n_sub_feat_size = int(X_train.shape[1]*int(feature_percentage)/100)
            S_hat = np.argsort(model.coef_)[::-1][:n_sub_feat_size].flatten() #40% features           
            model = linear_model.Lasso(alpha=0.01, max_iter=5000)
            model = create_model(args, file_path, model, X_train[:,S_hat], T_train)
            X_train = X_train[:,S_hat]
            X_test = X_test[:,S_hat]

        # Ugly hack otherwise vector norm not calculated
        #T_train = T_train.reshape(X_train.shape[0], 1)
        #T_test = T_test.reshape(X_test.shape[0], 1)

        log_params = True
    
    elif args.algo=="E-NET":
        file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".joblib"

        T_train = np.argmax(T_train, axis=1)
        T_test = np.argmax(T_test, axis=1)
        
        model = ElasticNet(alpha=0.01, l1_ratio=0.7)
        model = create_model(args, file_path, model, X_train, T_train)

        S_hat = np.argsort(model.coef_)

        log_params = False

    elif args.algo=="CORR":
        thershold = 0.01
        importance_vals = abs(np.dot((X_train.T),T_train).T)[::-1]
        S_hat = np.argsort(importance_vals  > thershold ).flatten()
        model_fpsr[0,i] = FPSR(S,S_hat)
        model_fnsr[0,i] = FNSR(S,S_hat)

        log_params = False
    elif args.algo=="SPINN":
        # https://github.com/jjfeng/spinn
        log_params = False
        print ("Not yet implemented!")

    else:
        print("Sorry! No such evaluation exists.")

    if log_params:
        # Mean squared errors
        model_msfe[0,i] = compute_mse_compare(model.predict(X_train).reshape(T_train.shape), T_train)
        model_mspe[0,i] = compute_mse_compare(model.predict(X_test).reshape(T_test.shape), T_test)
        # Selection rate errors
        model_fpsr[0,i] = FPSR(S,S_hat)
        model_fnsr[0,i] = FNSR(S,S_hat)
        # Cardinality of the model
        model_card[0,i] = len(S_hat)
        # Normalized Error (NME)
        model_nme_train[0,i] = compute_nme(model.predict(X_train).reshape(T_train.shape), T_train)
        model_nme_test[0,i] = compute_nme(model.predict(X_test).reshape(T_test.shape), T_test)
                
        if args.algo == "BART":
            val = model.predict(X_train)
            normalized = (val-min(val))/(max(val)-min(val))
            accuracy = np.sum([abs(0.9*normalized - T_train.flatten()) < 0.2]) / len(T_train.flatten())
            print("**********The train accuracy is: ", accuracy)
        else:
            print("**********The train accuracy is: ", calculate_accuracy(model.predict(X_train).reshape(T_train.shape).T, T_train.T))

        if args.algo == "BART":
            val = model.predict(X_test)
            normalized = (val-min(val))/(max(val)-min(val))
            accuracy = np.sum([abs(0.9*normalized - T_test.flatten()) < 0.2]) / len(T_test.flatten())
            print("**********The test accuracy is: ", accuracy)
        else:
            print("**********The test accuracy is: ", calculate_accuracy(model.predict(X_test).reshape(T_test.shape).T, T_test.T))

    print ("Time taken for this MC iteration: ", time.time() - start_time)

             
def main():
    args = define_parser()

    S, X_train, X_test, T_train, T_test = prepare_data(args)
    
    model_fpsr = np.zeros((1, int(args.MC_Num)))
    model_fnsr = np.zeros((1, int(args.MC_Num)))
    model_msfe = np.zeros((1, int(args.MC_Num)))
    model_mspe = np.zeros((1, int(args.MC_Num)))
    model_card = np.zeros((1, int(args.MC_Num)))
    model_nme_train = np.zeros((1, int(args.MC_Num)))
    model_nme_test  = np.zeros((1, int(args.MC_Num)))

    file_path_prefix = "./parameters/"
    if args.algo == "BART":
        parameter_file = file_path_prefix + args.data + "/" + args.algo + str(args.tree_size) + "_params"
    else:
        parameter_file = file_path_prefix + args.data + "/" + args.algo + "_params"
 

    if (os.path.isfile(parameter_file)):
        print ("--- Loading from the parameters ---", args.algo, "on", args.data)
        load_parameters(parameter_file)
    else:
        for i in np.arange(0,int(args.MC_Num)):
            print ("Monte carlo simulation no: ", str(i))
            run_feature_selector_algo(args, S, X_train, X_test, T_train, T_test, i, model_fpsr, model_fnsr, model_msfe, model_mspe, model_card, model_nme_train, model_nme_test)
  
        save_parameters(parameter_file, model_fpsr, model_fnsr, model_msfe, model_mspe, model_card, model_nme_train, model_nme_test)


if __name__ == '__main__':
    main()
