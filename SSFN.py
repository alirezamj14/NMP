# -*- coding: utf-8 -*-

import numpy as np
import logging
from MyFunctions import *
from MyOptimizers import *
import json
import os

def SSFN(X_train, X_test, T_train, T_test, SSFN_hparameters):
    """[Implements SSFN]

    Args:
        X_train ([float]): [The matrix of training data. Each column contains one sample.]
        X_test ([float]): [The matrix of testing data. Each column contains one sample.]
        T_train ([float]): [The matrix of training target. Each column contains one sample.]
        T_test ([float]): [The matrix of testing target. Each column contains one sample.]
        SSFN_hparameters ([dic]): [The dictionary of hyperparameters of SSFN.]

    Returns:
        [float]: [Training and testing error in dB.]
    """
    data = SSFN_hparameters["data"]
    lam = SSFN_hparameters["lam"]
    mu = SSFN_hparameters["mu"]
    kMax = SSFN_hparameters["kMax"]
    ni = SSFN_hparameters["NodeNum"]
    L = SSFN_hparameters["LayerNum"]

    parameters_path = "./parameters/"
    create_directory(parameters_path)

    P=X_train.shape[0]
    Q=T_train.shape[0]
    VQ=np.concatenate([np.eye(Q), (-1) * np.eye(Q)], axis=0)
    eps_o = 2 * np.sqrt(2*Q);

    train_error=[]
    test_error=[]
    test_accuracy=[]
    train_accuracy=[]

    O_ls = LS(X_train, T_train, lam)
    t_hat = np.dot(O_ls, X_train)
    t_hat_test = np.dot(O_ls, X_test)

    train_error.append(compute_nme(T_train,t_hat))
    test_error.append(compute_nme(T_test,t_hat_test))
    train_accuracy.append(calculate_accuracy(T_train,t_hat))
    test_accuracy.append(calculate_accuracy(T_test,t_hat_test))

    #   Initializing the algorithm for the first time
    Yi=X_train;
    Pi=P;
    Yi_test=X_test;

    for layer in range(1, L+1):
        # _logger.info("Begin to optimize layer {}".format(layer))
        Ri = 2 * np.random.rand(ni, Pi) - 1

        Zi_part1=np.dot(VQ, t_hat)
        Zi_part2=np.dot(Ri,Yi)
        Zi_part2 = Zi_part2 / np.linalg.norm(Zi_part2, axis=0)
        Zi=np.concatenate([Zi_part1, Zi_part2], axis=0)
        Yi_temp=activation(Zi)
        
        Oi=LS_ADMM(Yi_temp, T_train, eps_o, mu, kMax)    #   The ADMM solver for constrained least square
        t_hat=np.dot(Oi,Yi_temp)

        ##########  Test
        #  Following the same procedure for test data
        Zi_part1_test = np.dot(VQ, t_hat_test)
        Zi_part2_test = np.dot(Ri,Yi_test)
        Zi_part2_test = Zi_part2_test / np.linalg.norm(Zi_part2_test, axis=0)
        Zi_test=np.concatenate([Zi_part1_test, Zi_part2_test], axis=0)
        Yi_test_temp=activation(Zi_test)
        t_hat_test=np.dot(Oi,Yi_test_temp)

        train_error.append(compute_nme(T_train,t_hat))
        test_error.append(compute_nme(T_test,t_hat_test))
        train_accuracy.append(calculate_accuracy(T_train,t_hat))
        test_accuracy.append(calculate_accuracy(T_test,t_hat_test))

        train_listsP = [ '%.2f' % elem for elem in train_error ]
        test_listsP = [ '%.2f' % elem for elem in test_error ]


        Yi = Yi_temp
        Yi_test=Yi_test_temp
        Pi=Yi.shape[0]

    return compute_nme(T_train,t_hat), compute_nme(T_test,t_hat_test)
