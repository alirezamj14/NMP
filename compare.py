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
from bartpy.sklearnmodel import SklearnModel
from bartpy.features.featureselection import SelectNullDistributionThreshold
from sklearn.pipeline import make_pipeline

import pandas as pd
from MyFunctions import *


def define_parser():
    parser = argparse.ArgumentParser(description="Run NMP")
    parser.add_argument("--data", default="GRAVITATION", help="Input dataset available as the paper shows")
    parser.add_argument("--algo", default="RF", help="The algorithm used for feature selection")
    parser.add_argument("--tree_size", default="20", help="The number of trees used")
    parser.add_argument("--MC_Num", default="10", help="The number of MC simulations done")
    args = parser.parse_args()
    return args

def prepare_data(args):
    """
        X_train ([float]): [The matrix of training data. Each column contains one sample.]
        X_test ([float]): [The matrix of testing data. Each column contains one sample.]
        T_train ([float]): [The matrix of training target. Each column contains one sample.]
        T_test ([float]): [The matrix of testing target. Each column contains one sample.]
    """
    if args.data=="MNIST":
        X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
        X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
        T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
        T_test =  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)
        return X_train.T, X_test.T, T_train.T, T_test.T

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
        return S, X_train.T, X_test.T, T_train.T, T_test.T

def create_model(args, file_path, model, X_train, T_train):
    if (os.path.isfile(file_path)):
        print ("--- Loading from the model ---", args.algo)
        model = joblib.load(file_path)
    else:
        print ("--- Creating the model ---", args.algo)
        model.fit(X_train, T_train)
        joblib.dump(model, open(file_path, 'wb'), compress=0)
    return model

def load_parameters(parameter_file):
    # Getting back the objects:
    with open(parameter_file, 'rb') as f:
        model_avg_fpsr, model_std_fpsr, model_avg_fnsr, model_std_fnsr, model_avg_msfe, model_std_msfe, model_avg_mspe, model_std_mspe = pickle.load(f)

    print ("FPSR - Mean", model_avg_fpsr) 
    print ("FPSR - Std", model_std_fpsr) 
    print ("FNSR - Mean", model_avg_fnsr) 
    print ("FNSR - Std", model_std_fnsr) 
    print ("MFSE - Mean", model_avg_msfe) 
    print ("MFSE - Std", model_std_msfe) 
    print ("MSPE - Mean", model_avg_mspe) 
    print ("MSPE - Std", model_std_mspe)

def save_parameters(parameter_file, model_fpsr, model_fnsr, model_msfe, model_mspe):
    model_avg_fpsr = np.mean(model_fpsr)
    model_std_fpsr = np.std(model_fpsr)
    
    model_avg_fnsr = np.mean(model_fnsr)
    model_std_fnsr = np.std(model_fnsr)
    
    model_avg_msfe = np.mean(model_msfe)
    model_std_msfe = np.std(model_msfe)
    
    model_avg_mspe = np.mean(model_mspe)
    model_std_mspe = np.std(model_mspe)
    
    print ("FPSR - Mean", model_avg_fpsr) 
    print ("FPSR - Std", model_std_fpsr) 
    print ("FNSR - Mean", model_avg_fnsr) 
    print ("FNSR - Std", model_std_fnsr) 
    print ("MFSE - Mean", model_avg_msfe) 
    print ("MFSE - Std", model_std_msfe) 
    print ("MSPE - Mean", model_avg_mspe) 
    print ("MSPE - Std", model_std_mspe)

    pickle.dump([model_avg_fpsr,
                model_std_fpsr,
                model_avg_fnsr,
                model_std_fnsr,
                model_avg_msfe,
                model_std_msfe,
                model_avg_mspe,
                model_std_mspe], open(parameter_file, "wb"))


def run_feature_selector_algo(args, S, X_train, X_test, T_train, T_test):
    start_time = time.time()
    file_path_prefix = "./parameters/"
    parameter_file = file_path_prefix + args.data + "/" + args.algo + "_params"
    
    model_fpsr = np.zeros((1, int(args.MC_Num)))
    model_fnsr = np.zeros((1, int(args.MC_Num)))
    model_msfe = np.zeros((1, int(args.MC_Num)))
    model_mspe = np.zeros((1, int(args.MC_Num)))

    if (os.path.isfile(parameter_file)):
        print ("--- Loading from the parameters ---", args.algo, "on", args.data)
        load_parameters(parameter_file)
    else:
        if args.algo=="RF":
            for i in np.arange(0,int(args.MC_Num)):
                print ("Monte carlo simulation no: ", str(i))
                file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".joblib"
                
                rf = RandomForestRegressor(n_estimators=100) 
                rf = create_model(args, file_path, rf, X_train, T_train)
                importance_vals = rf.feature_importances_
                
                # Choose features which has 1% importance according to paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6660200/ 
                S_hat = np.argwhere(importance_vals > 0.01).flatten()
                
                # Mean squared errors
                model_msfe[0,i] = compute_mse(T_train, rf.predict(X_train).reshape(T_train.shape))
                model_mspe[0,i] = compute_mse(T_test, rf.predict(X_test).reshape(T_test.shape))
                # Selection rate errors
                model_fpsr[0,i] = FPSR(S,S_hat[0:len(S)])
                model_fnsr[0,i] = FNSR(S,S_hat[0:len(S)])
        
        if args.algo=="BART-20":
            pass

        if args.algo=="BART-30":
            pass

        if args.algo=="BART-50":
            pass

        if args.algo=="SPINN":
            pass

        if args.algo=="GAM":
            pass

        if args.algo=="L1-NN":
            pass

        save_parameters(parameter_file, model_fpsr, model_fnsr, model_msfe, model_mspe)
            
             
def main():
    args = define_parser()

    S, X_train, X_test, T_train, T_test = prepare_data(args)

    run_feature_selector_algo(args, S, X_train, X_test, T_train, T_test)
    
if __name__ == '__main__':
    main()