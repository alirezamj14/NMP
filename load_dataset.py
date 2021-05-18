# -*- coding: utf-8 -*-

import numpy as np
from MyFunctions import relu
from scipy.io import loadmat

def prepare_artificial():
    # Articial model from paper http://proceedings.mlr.press/v80/ye18b.html

    N = 1300             # Number of samples  
    Ntr = 1000           # Number of training samples 
    P = 5               # Number of input features
    PExtra=495          # Number of extra random features
    e = np.random.randn(1, N)
    Z = np.random.randn(P+PExtra, N)
    X = (Z + e)/2
    epsilon = np.random.randn(1, 1)

    X_train = X[:,:Ntr]
    X_test = X[:,Ntr:]

    T_train = (10 * np.sin(np.maximum(X_train[0,:], X_train[1,:])) + (np.maximum(np.maximum(X_train[2,:], X_train[3,:]), X_train[4,:]))**3 )/( 1 + (X_train[0,:] + X_train[4,:])**2 ) \
            + np.sin(0.5 * X_train[2,:]) * (1 + np.exp(X_train[3,:] - 0.5 * X_train[2,:])) \
            + X_train[2,:]**2 + 2 * np.sin(X_train[3,:]) + 2 * X_train[4,:] + epsilon

    T_test = (10 * np.sin(np.maximum(X_test[0,:], X_test[1,:])) + (np.maximum(np.maximum(X_test[2,:], X_test[3,:]), X_test[4,:]))**3 )/( 1 + (X_test[0,:] + X_test[4,:])**2 ) \
            + np.sin(0.5 * X_test[2,:]) * (1 + np.exp(X_test[3,:] - 0.5 * X_test[2,:])) \
            + X_test[2,:]**2 + 2 * np.sin(X_test[3,:]) + 2 * X_test[4,:] + epsilon
    
    return X_train, X_test, T_train, T_test

def prepare_NN():
    # Ohm's law: https://en.wikipedia.org/wiki/Ohm%27s_law

    N = 10000         # Number of samples  
    Ntr = 9000       # Number of training samples 
    P = 50          # Number of input features
    n1 = 100        # Number of hidden neurons in the first layer
    n2 = 10         # Number of hidden neurons in the second layer
    X_train = 10*np.random.rand(P, Ntr)+10       # voltage
    X_test = 10*np.random.rand(P, N-Ntr)+10       # resistance
    W1 = np.random.randn(n1, P)
    W2 = np.random.randn(n2, n1)

    T_train = np.dot(W2 , relu(np.dot(W1, X_train)))
    T_test = np.dot(W2 , relu(np.dot(W1, X_test)))
    return X_train, X_test, T_train, T_test
    

def prepare_Ohm():
    # Ohm's law: https://en.wikipedia.org/wiki/Ohm%27s_law

    N=10000         # Number of samples  
    Ntr=9000       # Number of training samples 
    fExtra=0          # Number of extra random features
    V = 10*np.random.rand(1, N)+10       # voltage
    R = 10*np.random.rand(1, N)+10       # resistance
    I = V/R    # current
    
    X_train = np.concatenate((V[:,:Ntr], R[:,:Ntr], (10)*np.random.rand(fExtra,Ntr)+10), axis=0)
    T_train=I[:,:Ntr]
    X_test = np.concatenate((V[:,Ntr:], R[:,Ntr:], (10)*np.random.rand(fExtra,N-Ntr)+10), axis=0)
    T_test=I[:,Ntr:]
    return X_train, X_test, T_train, T_test


def prepare_Planck():
    # Planck's law: https://en.wikipedia.org/wiki/Planck%27s_law

    N=10000         # Number of samples  
    Ntr=9000       # Number of training samples
    fExtra=0           # Number of extra random features
    v = 10*np.random.rand(1, N)+10       # frequency
    T = 10*np.random.rand(1, N)+10       # absolute temperature
    B = 2 * (v**3) * (1/(np.exp(v/T)-1))    # spectral radiance of a body
    
    X_train = np.concatenate((v[:,:Ntr], T[:,:Ntr], (10)*np.random.rand(fExtra,Ntr)+10), axis=0)
    T_train=B[:,:Ntr]
    X_test = np.concatenate((v[:,Ntr:], T[:,Ntr:], (10)*np.random.rand(fExtra,N-Ntr)+10), axis=0)
    T_test=B[:,Ntr:]
    return X_train, X_test, T_train, T_test

def prepare_Gravitation():
    N=10000         # Number of samples  
    Ntr=9000       # Number of training samples  
    fExtra=0           # Number of extra random features
    m1=10*np.random.rand(1, N)+10
    m2=10*np.random.rand(1, N)+10
    r=10*np.random.rand(1, N)+10
    F=np.multiply(m1, m2)/(r**2)
    
    X_train = np.concatenate((m1[:,:Ntr], m2[:,:Ntr], r[:,:Ntr], 10*np.random.rand(fExtra,Ntr)+10), axis=0)
    T_train=F[:,:Ntr]
    X_test = np.concatenate((m1[:,Ntr:], m2[:,Ntr:], r[:,Ntr:], 10*np.random.rand(fExtra,N-Ntr)+10), axis=0)
    T_test=F[:,Ntr:]
    return X_train, X_test, T_train, T_test

def prepare_cifar10():
    cifar10 = loadmat("./mat_files/CIFAR-10.mat")
    X_train =  cifar10["train_x"].astype(np.float32)
    X_test =  cifar10["test_x"].astype(np.float32)
    T_train =  cifar10["train_y"].astype(np.float32)
    T_test=  cifar10["test_y"].astype(np.float32)
    return X_train, X_test, T_train, T_test


def prepare_satimage():
    X_train = loadmat("./mat_files/Satimage.mat")["train_x"].astype(np.float32)
    T_train = loadmat("./mat_files/Satimage.mat")["train_y"].astype(np.float32)
    X_test = loadmat("./mat_files/Satimage.mat")["test_x"].astype(np.float32)
    T_test= loadmat("./mat_files/Satimage.mat")["test_y"].astype(np.float32)
    return X_train, X_test, T_train,  T_test

def prepare_mnist():
    X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
    X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
    T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
    T_test=  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)
    return X_train, X_test, T_train, T_test

def prepare_vowel():
    X = loadmat("./mat_files/Vowel.mat")["featureMat"]
    T = loadmat("./mat_files/Vowel.mat")["labelMat"]
    X_train,X_test = X[:, :528].astype(np.float32), X[:, 528:].astype(np.float32)
    T_train, T_test = T[:, :528].astype(np.float32), T[:, 528:].astype(np.float32)
    return X_train, X_test, T_train, T_test

def prepare_norb():
    X_train = loadmat("./mat_files/NORB.mat")["train_x"].T.astype(np.float32)
    T_train = loadmat("./mat_files/NORB.mat")["train_y"].T.astype(np.float32)
    X_test = loadmat("./mat_files/NORB.mat")["test_x"].T.astype(np.float32)
    T_test= loadmat("./mat_files/NORB.mat")["test_y"].T.astype(np.float32)
    return X_train, X_test, T_train,  T_test

def prepare_shuttle():
    X_train = loadmat("./mat_files/Shuttle.mat")["train_x"].astype(np.float32)
    T_train = loadmat("./mat_files/Shuttle.mat")["train_y"].astype(np.float32)
    X_test = loadmat("./mat_files/Shuttle.mat")["test_x"].astype(np.float32)
    T_test= loadmat("./mat_files/Shuttle.mat")["test_y"].astype(np.float32)
    return X_train, X_test, T_train,  T_test

def prepare_caltech():
    train_num = 6000
    test_num = 3000
    X = loadmat("./mat_files/Caltech101.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Caltech101.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    X_train, X_test = X[:, random_train_lists], X[:, random_test_lists]
    T_train, T_test = T[:, random_train_lists], T[:, random_test_lists]
    return X_train, X_test, T_train,  T_test
   
def prepare_letter():
    train_num = 13333
    test_num = 6667
    X = loadmat("./mat_files/Letter.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Letter.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    X_train, X_test = X[:, random_train_lists], X[:, random_test_lists]
    T_train, T_test = T[:, random_train_lists], T[:, random_test_lists]
    return X_train, X_test, T_train,  T_test

def prepare_Boston():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    X, T = load_boston(return_X_y=True)
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.33, random_state=21)
    fExtra=100
    X_train = np.concatenate((X_train, 10*np.random.rand(X_train.shape[0],fExtra)+10), axis=1)
    X_test = np.concatenate((X_test, 10*np.random.rand(X_test.shape[0],fExtra)+10), axis=1)
    T_train = T_train.reshape(-1,1)
    T_test = T_test.reshape(-1,1)
    
    X_train = (X_train - X_train.mean(axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / np.std(X_test, axis=0)

    return X_train.T, X_test.T, T_train.T, T_test.T


def prepare_Airfoil():
    X_train = loadmat("./mat_files/Airfoil.mat")["X_train"].astype(np.float32)
    T_train = loadmat("./mat_files/Airfoil.mat")["T_train"].astype(np.float32)
    X_test = loadmat("./mat_files/Airfoil.mat")["X_test"].astype(np.float32)
    T_test= loadmat("./mat_files/Airfoil.mat")["T_test"].astype(np.float32)

    X_train = (X_train - X_train.mean(axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / np.std(X_test, axis=0)

    fExtra=500
    X_train = np.concatenate((X_train, np.random.randn(X_train.shape[0],fExtra)), axis=1)
    X_test = np.concatenate((X_test, np.random.randn(X_test.shape[0],fExtra)), axis=1)
    
    return X_train.T, X_test.T, T_train.T,  T_test.T