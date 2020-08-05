# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat


def prepare_Ohm():
    # Ohm's law: https://en.wikipedia.org/wiki/Ohm%27s_law

    N=10000         # Number of samples  
    Ntr=9000       # Number of training samples 
    fExtra=8          # Number of extra random features
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
    fExtra=8           # Number of extra random features
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
    fExtra=7           # Number of extra random features
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
    train_X = loadmat("./mat_files/Satimage.mat")["train_x"].astype(np.float32)
    train_T = loadmat("./mat_files/Satimage.mat")["train_y"].astype(np.float32)
    test_X = loadmat("./mat_files/Satimage.mat")["test_x"].astype(np.float32)
    test_T= loadmat("./mat_files/Satimage.mat")["test_y"].astype(np.float32)
    return train_X, test_X, train_T, test_T

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
    train_X = loadmat("./mat_files/NORB.mat")["train_x"].T.astype(np.float32)
    train_T = loadmat("./mat_files/NORB.mat")["train_y"].T.astype(np.float32)
    test_X = loadmat("./mat_files/NORB.mat")["test_x"].T.astype(np.float32)
    test_T= loadmat("./mat_files/NORB.mat")["test_y"].T.astype(np.float32)
    return train_X, test_X, train_T, test_T

def prepare_shuttle():
    train_X = loadmat("./mat_files/Shuttle.mat")["train_x"].astype(np.float32)
    train_T = loadmat("./mat_files/Shuttle.mat")["train_y"].astype(np.float32)
    test_X = loadmat("./mat_files/Shuttle.mat")["test_x"].astype(np.float32)
    test_T= loadmat("./mat_files/Shuttle.mat")["test_y"].astype(np.float32)
    return train_X, test_X, train_T, test_T

def prepare_caltech():
    train_num = 6000
    test_num = 3000
    X = loadmat("./mat_files/Caltech101.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Caltech101.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T
   
def prepare_letter():
    train_num = 13333
    test_num = 6667
    X = loadmat("./mat_files/Letter.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Letter.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T