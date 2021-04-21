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
from bartpy.sklearnmodel import SklearnModel
from bartpy.features.featureselection import SelectNullDistributionThreshold
from sklearn.pipeline import make_pipeline
from tensorflow import keras
import random
import pandas as pd
from MyFunctions import *


def define_parser():
    parser = argparse.ArgumentParser(description="Run NMP")
    parser.add_argument("--data", default="MNIST", help="Input dataset available as the paper shows")
    parser.add_argument("--algo", default="DEEPLIFT", help="The algorithm used for feature selection")
    parser.add_argument("--tree_size", default="20", help="The number of trees used")
    parser.add_argument("--MC_Num", default="10", help="The number of MC simulations done")
    parser.add_argument("--deeplift_sample_size", default="10", help="The number of samples chosen from deeplift for explaining in each MC simulation")
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
        with open('./parameters/MNIST_sorted_ind.pkl', 'rb') as f:
             indices = pickle.load(f) 
        
        if args.algo=="RF":
            return indices['sorted_ind'][:300], X_train.T[:10000], X_test.T[:3000], T_train.T[:10000], T_test.T[:3000]
        else:
            return indices['sorted_ind'][:300], X_train.T, X_test.T, T_train.T, T_test.T

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
        model_avg_nme, \
        model_std_nme = pickle.load(f)

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
    print ("NME  - Mean", model_avg_nme)
    print ("NME  - Std", model_std_nme)

def save_parameters(parameter_file, model_fpsr, model_fnsr, model_msfe, model_mspe, model_card, model_nme):
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

    model_avg_nme = np.mean(model_nme)
    model_std_nme = np.std(model_nme)
    
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
    print ("NME  - Mean", model_avg_nme)
    print ("NME  - Std", model_std_nme)

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
                model_avg_nme,
                model_std_nme], open(parameter_file, "wb"))

def create_model(args, file_path, model, X_train, T_train):
    if (os.path.exists(file_path)):
        print ("--- Loading from the model ---", args.algo)
        
        if args.algo == "DEEPLIFT":
            model = keras.models.load_model(file_path)
            model.summary()
        else:
            model = joblib.load(file_path)
    else:
        print ("--- Creating the model ---", args.algo)
        
        if args.algo == "DEEPLIFT":
            model.fit(X_train, T_train, epochs=3, batch_size=128)
            model.save(file_path)
        else:
            model.fit(X_train, T_train)
            joblib.dump(model, open(file_path, 'wb'), compress=0)
    return model

def run_feature_selector_algo(args, S, X_train, X_test, T_train, T_test):
    log_params = False
    file_path_prefix = "./parameters/"
    parameter_file = file_path_prefix + args.data + "/" + args.algo + "_params"
    
    model_fpsr = np.zeros((1, int(args.MC_Num)))
    model_fnsr = np.zeros((1, int(args.MC_Num)))
    model_msfe = np.zeros((1, int(args.MC_Num)))
    model_mspe = np.zeros((1, int(args.MC_Num)))
    model_card = np.zeros((1, int(args.MC_Num)))
    model_nme  = np.zeros((1, int(args.MC_Num)))

    if (os.path.isfile(parameter_file)):
        print ("--- Loading from the parameters ---", args.algo, "on", args.data)
        load_parameters(parameter_file)
    else:
        if args.algo=="RF":
            for i in np.arange(0,int(args.MC_Num)):
                start_time = time.time()
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
                # Cardinality of the model
                model_card[0,i] = len(S_hat)
                # Normalized Error (NME)
                model_nme [0,i] = compute_nme(rf.predict(X_test).reshape(T_test.shape), T_test)

                print ("Time taken for this MC iteration: ", time.time() - start_time)
            log_params = True
        
        elif args.algo == "DEEPLIFT": # Implemented using DeepExplain in SHAP: https://github.com/slundberg/shap
            import shap
            from tensorflow import keras
            from tensorflow.keras import layers

            # Model / data parameters
            num_classes = 10
            input_shape = (28, 28, 1)

            # the data, split between train and test sets
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            # Scale images to the [0, 1] range
            x_train = x_train.astype("float32") / 255
            x_test = x_test.astype("float32") / 255
            # Make sure images have shape (28, 28, 1)
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            print("x_train shape:", x_train.shape)
            print(x_train.shape[0], "train samples")
            print(x_test.shape[0], "test samples")

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            """
            ## Build the model
            """

            model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )
            model.summary()
                        
            for i in np.arange(0,int(args.MC_Num)):
                start_time = time.time()
                print ("Monte carlo simulation no: ", str(i))
                file_path = file_path_prefix + args.data + "/" + args.algo + "-" + str(i) + ".h5"
                
                """
                ## Train the model
                """

                batch_size = 128
                epochs = 15

                model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
                model = create_model(args, file_path, model, x_train, y_train)
                
                # Sanity checks
                score_train = model.evaluate(x_train, y_train, verbose=0)
                score_test = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score_test[0])
                print("Test accuracy:", score_test[1])
                
    
                background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
                # explain predictions of the model on four images
                e = shap.DeepExplainer(model, background)
                
                x_test_sample = x_test[np.random.choice(x_test.shape[0], int(args.deeplift_sample_size), replace=False), :]

                shap_values = e.shap_values(x_test_sample)

                total_val = np.sum(np.sum(np.abs(shap_values), axis=0), axis=0).flatten()
                S_hat = total_val.argsort()[::-1]

                # Just to compare what global features SHAP with DeepLift choose
                X_train_ori =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
                show_image([X_train_ori[:,100],X_train_ori[:,200],X_train_ori[:,300]],S_hat[0:len(S)], (args.algo+str(i)))
                
                #show_image(x_train[1,:].flatten(),x_train[20,:].flatten(),x_train[30,:].flatten(),S_hat, (args.algo+str(i)))

                # Mean squared errors
                model_msfe[0,i] = compute_mse(y_train, model.predict(x_train).reshape(y_train.shape))
                model_mspe[0,i] = compute_mse(y_test, model.predict(x_test).reshape(y_test.shape))


                # Selection rate errors
                model_fpsr[0,i] = FPSR(S,S_hat[0:len(S)])
                model_fnsr[0,i] = FNSR(S,S_hat[0:len(S)])
                # Cardinality of the model
                model_card[0,i] = len(S_hat)

                # Normalized Error (NME)
                model_nme [0,i] = compute_nme(model.predict(x_test).reshape(y_test.shape), y_test)


                print ("Time taken for this MC iteration: ", time.time() - start_time)
            log_params = True

        elif args.algo=="BART-20":
            pass

        elif args.algo=="BART-30":
            pass

        elif args.algo=="BART-50":
            pass

        elif args.algo=="SPINN":
            pass

        elif args.algo=="GAM":
            pass


        else:
            print("Sorry! No such evaluation exists.")
        
        if log_params:
            save_parameters(parameter_file, model_fpsr, model_fnsr, model_msfe, model_mspe, model_card, model_nme)
            
             
def main():
    args = define_parser()

    S, X_train, X_test, T_train, T_test = prepare_data(args)

    run_feature_selector_algo(args, S, X_train, X_test, T_train, T_test)
  

if __name__ == '__main__':
    main()


'''
    if args.algo=="XBART":
        from xbart import XBART

        y_train = np.asarray([np.argmax(t, axis=None, out=None) for t in T_train])/10.0
        x_train = pd.DataFrame(X_train)
        x_test = pd.DataFrame(X_test)

        xbt = XBART(num_trees = 100, num_sweeps = 40, burnin = 15)
        xbt.fit(x_train,y_train)
        xbart_yhat_matrix = xbt.predict(x_test)  # Return n X num_sweeps matrix
        y_hat = xbart_yhat_matrix[:,15:].mean(axis=1) # Use mean a prediction estimate

    if args.algo=="BART":

        # bart = SklearnModel(n_trees=20, store_in_sample_predictions=False, n_jobs=1) # Use default parameters
        bart = SklearnModel(n_samples=200,
                            n_burn=50,
                            n_trees=20,
                            store_in_sample_predictions=False,
                            n_jobs=-1,
                            n_chains=1) # Use default parameters
        
        T_classes = np.asarray([np.argmax(t, axis=None, out=None) for t in T_train])/10.0
        X = pd.DataFrame(X_train)

        pipeline = make_pipeline(SelectNullDistributionThreshold(bart, n_permutations=20), bart)
        pipeline_model = create_model(args, file_path, pipeline, X_train, T_classes)
        pipeline = make_pipeline(SelectNullDistributionThreshold(pipeline_model, 0.75, "local"), pipeline_model)

        print("Feature Proportions", pipeline_model.named_steps["selectnulldistributionthreshold"].feature_proportions)

'''