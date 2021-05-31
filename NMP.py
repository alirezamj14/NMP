# -*- coding: utf-8 -*-

import logging 
import argparse
import numpy as np
from SSFN import SSFN
from CNN import my_CNN
from MLP import MLP
from MyFunctions import *
from load_dataset import *
import multiprocessing
from joblib import Parallel, delayed
from test_masked_train import return_patched_data

def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="Gravitation", help="Input dataset available as the paper shows")
    parser.add_argument("--lam", type=float, default=10**(2), help="Reguralized parameters on the least-square problem")
    parser.add_argument("--mu", type=float, default=10**(3), help="Parameter for ADMM")
    parser.add_argument("--kMax", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--NodeNum", type=int, default=100, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum", type=int, default=5, help="Parameter for ADMM")
    parser.add_argument("--J", type=int, default=50, help="Sample Size")
    parser.add_argument("--Pextra", type=int, default=7, help="Number of extra random features")
    args = parser.parse_args()
    return args

def define_logger():
    _logger = logging.getLogger(__name__)
    logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
    return _logger

def define_dataset(args):
    if args.data == "Gravitation":
        X_train, X_test, T_train,  T_test  = prepare_Gravitation()
    elif args.data == "MNIST":
        X_train, X_test, T_train,  T_test  = prepare_mnist()
    elif args.data == "Vowel":
        X_train, X_test, T_train,  T_test  = prepare_vowel()
    elif args.data == "Planck":
        X_train, X_test, T_train,  T_test  = prepare_Planck()
    elif args.data == "Ohm":
        X_train, X_test, T_train,  T_test  = prepare_Ohm()
    elif args.data == "NN":
        X_train, X_test, T_train,  T_test  = prepare_NN()
    elif args.data == "Artificial":
        X_train, X_test, T_train,  T_test  = prepare_artificial()
    elif args.data == "Modelnet10":
        X_train, X_test, T_train,  T_test  = prepare_Modelnet10()
    return X_train, X_test, T_train, T_test

def set_hparameters(args):
    SSFN_hparameters = {"data": args.data, "lam": args.lam, "mu": args.mu, \
            "kMax": args.kMax, "NodeNum": args.NodeNum, "LayerNum": args.LayerNum\
                , "J": args.J, "Pextra": args.Pextra}
    return SSFN_hparameters

def LookAhead(test_error_array, sorted_ind, search_ind, X_train, X_test, T_train, T_test, SSFN_hparameters):
    """[This function is used for look-ahead variant of NMP.
        It proceed one more step for the two best indices and choose the best index according to the next step evaluation.]

    Args:
        test_error_array ([float]): [Array of test error for each of the features in a step of NMP.]
        sorted_ind ([int]): [The sorted set S according to the paper.]
        search_ind ([int]): [The search set A according to the paper.]
        X_train ([float]): [The matrix of training data. Each column contains one sample.]
        X_test ([float]): [The matrix of testing data. Each column contains one sample.]
        T_train ([float]): [The matrix of training target. Each column contains one sample.]
        T_test ([float]): [The matrix of testing target. Each column contains one sample.]
        SSFN_hparameters ([dic]): [The dictionary of hyperparameters of SSFN.]

    Returns:
        [int]: [Best index according to look-ahead strategy.]
    """
    ind_array = np.argsort(test_error_array)

    sorted_ind0 = np.append(sorted_ind, search_ind[ind_array[0]])
    search_ind0 = np.delete(search_ind, ind_array[0])

    test_error_temp = np.array([])
    for i in search_ind0:
        if len(sorted_ind0)>=1:
            X_tr = X_train[np.append(sorted_ind0,i),:]
            X_ts = X_test[np.append(sorted_ind0,i),:]
        else:
            X_tr = X_train[[i],:]
            X_ts = X_test[[i],:]
        _, test_error, _, _ = SSFN(X_tr, X_ts, T_train, T_test, SSFN_hparameters)
        test_error_temp = np.append(test_error_temp, test_error)
    myMin0 = np.min(test_error_temp)  

    sorted_ind1 = np.append(sorted_ind, search_ind[ind_array[1]])
    search_ind1 = np.delete(search_ind, ind_array[1])

    test_error_temp = np.array([])
    for i in search_ind1:
        if len(sorted_ind1)>=1:
            X_tr = X_train[np.append(sorted_ind1,i),:]
            X_ts = X_test[np.append(sorted_ind1,i),:]
        else:
            X_tr = X_train[[i],:]
            X_ts = X_test[[i],:]
        _, test_error, _, _ = SSFN(X_tr, X_ts, T_train, T_test, SSFN_hparameters)
        test_error_temp = np.append(test_error_temp, test_error)
    myMin1 = np.min(test_error_temp)  

    if myMin0 <= myMin1:
        LookAhead_ind = ind_array[0]
    else:
        LookAhead_ind = ind_array[1]

    return LookAhead_ind

def Err_vs_feat_window(X_train, X_test, T_train, T_test, args):
    """[This function plots training and testing error versus number of features |S| for an image dataset.]

    Args:
        args ([parser]): [It contains the inputs specifies by the user such as name of the dataset, and hyperparameters of the NN.]
    """
    SSFN_hparameters = set_hparameters(args)

    eta = args.eta

    J = SSFN_hparameters["J"]
    Pextra = SSFN_hparameters["Pextra"]
    data = SSFN_hparameters["data"]
    LayerNum = SSFN_hparameters["LayerNum"]
    NodeNum = SSFN_hparameters["NodeNum"]

    # X_train, X_test, T_train, T_test = define_dataset(args)
    # X_train = X_train[:,:int(round(0.9*J))] 
    # T_train = T_train[:,:int(round(0.9*J))]
    # X_test = X_test[:,:int(round(0.1*J))] 
    # T_test = T_test[:,:int(round(0.1*J))]

    Ntr = X_train.shape[1]
    Nts = X_test.shape[1]

    if data!="MNIST":
        X_train = np.concatenate((X_train, (10)*np.random.rand(Pextra,Ntr)+10), axis=0)
        X_test = np.concatenate(( X_test, (10)*np.random.rand(Pextra,Nts)+10), axis=0)

    # if data=="MNIST":
    #     X_train = 1 - X_train
    #     X_test = 1 - X_test
        
    parameters_path = "./parameters/"
    result_path = "./results/"
    LA = "None"

    # train_error, test_error = SSFN( X_train, X_test, T_train, T_test, SSFN_hparameters)
    rows, cols, depth = (28, 28, 1)
    radius = 4
    R_num = rows // radius
    C_num = cols // radius
    search_ind = np.arange(R_num * C_num)

    P = X_train.shape[0]
    train_nme_sorted = np.array([])
    test_nme_sorted = np.array([])
    train_mse_sorted = np.array([])
    test_mse_sorted = np.array([])
    train_acc_sorted = np.array([])
    test_acc_sorted = np.array([])
    sorted_ind = np.array([],  dtype=int)
    sorted_ind_row = np.array([],  dtype=int)
    sorted_ind_col = np.array([],  dtype=int)

    while len(search_ind) > 0 :
        train_nme_array = np.array([])
        test_nme_array = np.array([])
        train_mse_array = np.array([])
        test_mse_array = np.array([])
        train_acc_array = np.array([])
        test_acc_array = np.array([])
        for i in search_ind: 
            row = i // C_num
            col = i % C_num
            if len(sorted_ind_row)>=1 and len(sorted_ind_col)>=1 :
                X_tr, X_ts = return_patched_data(X_train, X_test, np.append(sorted_ind_row,row), np.append(sorted_ind_col,col), radius=radius, rows=28, cols=28)
            elif len(sorted_ind_row)>=1 and len(sorted_ind_col)==0 :
                X_tr, X_ts = return_patched_data(X_train, X_test, np.append(sorted_ind_row,row), col, radius=radius, rows=28, cols=28)
            elif len(sorted_ind_row)>=0 and len(sorted_ind_col)==1 :
                X_tr, X_ts = return_patched_data(X_train, X_test, row, np.append(sorted_ind_col,col), radius=radius, rows=28, cols=28)
            else:
                X_tr, X_ts = return_patched_data(X_train, X_test, row, col, radius=radius, rows=28, cols=28)
            # train_nme, test_nme, train_mse, test_mse, train_acc, test_acc = SSFN( X_tr, X_ts, T_train, T_test, SSFN_hparameters)
            train_nme, test_nme, train_mse, test_mse, train_acc, test_acc = my_CNN( X_tr, X_ts, T_train, T_test)
            train_nme_array = np.append(train_nme_array, train_nme)
            test_nme_array = np.append(test_nme_array, test_nme)
            train_mse_array = np.append(train_mse_array, train_mse)
            test_mse_array = np.append(test_mse_array, test_mse)
            train_acc_array = np.append(train_acc_array, train_acc)
            test_acc_array = np.append(test_acc_array, test_acc)
            # print(test_acc_array)

        i = np.argmin(test_nme_array)

        best_ind = search_ind[i]
        best_ind_row = search_ind[i] // C_num
        best_ind_col = search_ind[i] % C_num

        train_nme_sorted = np.append(train_nme_sorted, train_nme_array[i])
        test_nme_sorted = np.append(test_nme_sorted, test_nme_array[i])
        train_mse_sorted = np.append(train_mse_sorted, train_mse_array[i])
        test_mse_sorted = np.append(test_mse_sorted, test_mse_array[i])
        train_acc_sorted = np.append(train_acc_sorted, train_acc_array[i])
        test_acc_sorted = np.append(test_acc_sorted, test_acc_array[i])

        search_ind = np.delete(search_ind, i)
        sorted_ind = np.append(sorted_ind, best_ind)
        sorted_ind_row = np.append(sorted_ind_row, best_ind_row)
        sorted_ind_col = np.append(sorted_ind_col, best_ind_col)

        print("rows: "+str(sorted_ind_row))
        print("cols: "+str(sorted_ind_col))
        print("Test NME:" + str(test_nme_sorted))
        print("Test ACC:" + str(test_acc_sorted))

        if len(test_nme_sorted) >= 0.4 * R_num * C_num:
            break


    # MyFPSR = FPSR([0, 1, 2],sorted_ind[0:3]) 
    # print("FPSR: " + str(MyFPSR))
    # MyFNSR = FNSR([0, 1, 2],sorted_ind[0:3]) 
    # print("FNSR: " + str(MyFNSR))

    output_dic = {}
    output_dic["sorted_ind"]=sorted_ind 
    output_dic["sorted_ind_row"]=sorted_ind_row 
    output_dic["sorted_ind_col"]=sorted_ind_col 
    output_dic["test_acc_sorted"]=test_acc_sorted 
    output_dic["train_acc_sorted"]=train_acc_sorted
    output_dic["test_nme_sorted"]=test_nme_sorted 
    output_dic["train_nme_sorted"]=train_nme_sorted 
    output_dic["test_mse_sorted"]=test_mse_sorted 
    output_dic["train_mse_sorted"]=train_mse_sorted 
    save_dic(output_dic, parameters_path, data, "sorted_CNN_window4")

    FontSize = 18
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(np.arange(1,len(test_nme_sorted)+1), test_nme_sorted, 'r-', label="Test", linewidth=3)
    plt.plot(np.arange(1,len(test_nme_sorted)+1), train_nme_sorted, 'b-', label="Train", linewidth=2)
    plt.legend(loc='best', fontsize=FontSize)
    plt.grid()
    plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("Normalized error (dB)",fontdict=csfont, fontsize=FontSize)
    # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"Err_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+"_CNN_window4.png",dpi=600)
    plt.close()

    # if data=="MNIST":
    #     csfont = {'fontname':'sans-serif'}
    #     plt.subplots()
    #     plt.plot(np.arange(1,P+1), test_acc_sorted, 'r-', label="Test Accuracy", linewidth=2)
    #     plt.plot(np.arange(1,P+1), train_acc_sorted, 'b-', label="Train Accuracy", linewidth=2)
    #     plt.legend(loc='best')
    #     plt.grid()
    #     plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    #     plt.ylabel("Classification accuracy",fontdict=csfont, fontsize=FontSize)
    #     # plt.title(data+", SSFN", loc='center')
    #     plt.xticks(fontsize=FontSize)
    #     plt.yticks(fontsize=FontSize)
    #     plt.tight_layout()
    #     plt.savefig(result_path +"Acc_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+".png",dpi=600)
    #     plt.close()

    return (sorted_ind_row,sorted_ind_col),  train_nme_sorted[-1], test_nme_sorted[-1], train_mse_sorted[-1], test_mse_sorted[-1]

def Err_vs_feat(X_train, X_test, T_train, T_test, args):
    """[This function plots training and testing error versus number of features |S|, refer to Figure 1 and 2 in the Overleaf.]

    Args:
        args ([parser]): [It contains the inputs specifies by the user such as name of the dataset, and hyperparameters of the NN.]
    """
    SSFN_hparameters = set_hparameters(args)

    eta = args.eta

    if args.flag == "given_order":
        best_ind_given = args.best_ind

    best_ind_true = [1, 0, 3, 2, 4] 
    best_ind_reversed = [4, 2, 3, 0, 1]
    best_ind_random = np.random.choice([0,1,2,3,4], 5, replace=False)
    k = 0

    J = X_train.shape[1]
    print("J: "+ str(J))
    Pextra = SSFN_hparameters["Pextra"]
    data = SSFN_hparameters["data"]
    LayerNum = SSFN_hparameters["LayerNum"]
    NodeNum = SSFN_hparameters["NodeNum"]

    # X_train, X_test, T_train, T_test = define_dataset(args)
    # X_train = X_train[:,:int(round(0.9*J))] 
    # T_train = T_train[:,:int(round(0.9*J))]
    # X_test = X_test[:,:int(round(0.1*J))] 
    # T_test = T_test[:,:int(round(0.1*J))]

    Ntr = X_train.shape[1]
    Nts = X_test.shape[1]

    if data!="MNIST":
        X_train = np.concatenate((X_train, (10)*np.random.rand(Pextra,Ntr)+10), axis=0)
        X_test = np.concatenate(( X_test, (10)*np.random.rand(Pextra,Nts)+10), axis=0)

    if data=="MNIST":
        X_train = 1 - X_train
        X_test = 1 - X_test
        
    parameters_path = "./parameters/"
    result_path = "./results/"
    LA = "None"

    # train_error, test_error = SSFN( X_train, X_test, T_train, T_test, SSFN_hparameters)

    P=X_train.shape[0]
    search_ind = np.arange(P)
    train_acc_sorted = np.array([])
    test_acc_sorted = np.array([])
    train_nme_sorted = np.array([])
    test_nme_sorted = np.array([])
    train_mse_sorted = np.array([])
    test_mse_sorted = np.array([])
    sorted_ind = np.array([],  dtype=int)

    while len(search_ind) > 0:

        
        train_acc_array = np.array([])
        test_acc_array = np.array([])
        train_nme_array = np.array([])
        test_nme_array = np.array([])
        train_mse_array = np.array([])
        test_mse_array = np.array([])
        for i in search_ind:
            if len(sorted_ind)>=1:
                X_tr = X_train[np.append(sorted_ind,i),:]
                X_ts = X_test[np.append(sorted_ind,i),:]
            else:
                X_tr = X_train[[i],:]
                X_ts = X_test[[i],:]

            # if i == 36:
            #     pass
            if np.mean(X_train[[i],:], axis=1 ) <= 0.995:
                train_nme, test_nme, train_mse, test_mse, train_acc, test_acc = SSFN( X_tr, X_ts, T_train, T_test, SSFN_hparameters)
                # train_nme, test_nme, train_mse, test_mse = MLP( X_tr, X_ts, T_train, T_test, data)
            else:
                train_nme, test_nme, train_mse, test_mse, train_acc, test_acc = 10**10, 10**10, 10**10, 10**10, 10**10, 10**10

            train_acc_array = np.append(train_acc_array, train_acc)
            test_acc_array = np.append(test_acc_array, test_acc)
            train_nme_array = np.append(train_nme_array, train_nme)
            test_nme_array = np.append(test_nme_array, test_nme)
            train_mse_array = np.append(train_mse_array, train_mse)
            test_mse_array = np.append(test_mse_array, test_mse)

        if LA == "LookAhead":
            if len(test_nme_array)>1:
                i = LookAhead(test_nme_array, sorted_ind, search_ind, X_train, X_test, T_train, T_test, SSFN_hparameters)
            else:
                i = np.argmin(test_nme_array)
        else:
            i = np.argmin(test_nme_array)
            if args.flag == "given_order":
                i = np.where(search_ind == best_ind_given[k])
                k = k + 1
            # i = np.where(search_ind == best_ind_reversed[k])
            # i = np.where(search_ind == best_ind_random[k])
            
        if len(test_nme_sorted) >= 0.4*784:
            break
            if len(sorted_ind) == len(args.S):
                break
            if np.abs(test_nme_array[i] - test_nme_sorted[-1])/np.abs(test_nme_sorted[-1]) < eta or np.abs(test_nme_array[i]) <= np.abs(test_nme_sorted[-1]):
            # if np.abs(test_nme_array[i] - test_nme_sorted[-1])/np.abs(test_nme_sorted[-1]) > eta and np.abs(test_nme_array[i]) <= np.abs(test_nme_sorted[-1]):
                # break
                pass

        best_ind = search_ind[i]
        train_acc_sorted = np.append(train_acc_sorted, train_acc_array[i])
        test_acc_sorted = np.append(test_acc_sorted, test_acc_array[i])
        train_nme_sorted = np.append(train_nme_sorted, train_nme_array[i])
        test_nme_sorted = np.append(test_nme_sorted, test_nme_array[i])
        train_mse_sorted = np.append(train_mse_sorted, train_mse_array[i])
        test_mse_sorted = np.append(test_mse_sorted, test_mse_array[i])
        sorted_ind = np.append(sorted_ind, best_ind)
        search_ind = np.delete(search_ind, i)
    
        print(sorted_ind)
        
        # print(str(round(len(sorted_ind)/P * 100,2))+"%")

    # MyFPSR = FPSR([0, 1, 2],sorted_ind[0:3]) 
    # print("FPSR: " + str(MyFPSR))
    # MyFNSR = FNSR([0, 1, 2],sorted_ind[0:3]) 
    # print("FNSR: " + str(MyFNSR))
    print(test_acc_sorted[-1])

    output_dic = {}
    output_dic["sorted_ind"]=sorted_ind 
    output_dic["test_acc_sorted"]=test_acc_sorted 
    output_dic["train_acc_sorted"]=train_acc_sorted
    output_dic["test_nme_sorted"]=test_nme_sorted 
    output_dic["train_nme_sorted"]=train_nme_sorted 
    output_dic["test_mse_sorted"]=test_mse_sorted 
    output_dic["train_mse_sorted"]=train_mse_sorted 
    save_dic(output_dic, parameters_path, data, "sorted_threshold_0995")

    FontSize = 18
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(np.arange(1,len(test_nme_sorted)+1), test_nme_sorted, 'r-', label="Test", linewidth=3)
    plt.plot(np.arange(1,len(test_nme_sorted)+1), train_nme_sorted, 'b-', label="Train", linewidth=2)
    plt.legend(loc='best', fontsize=FontSize)
    plt.grid()
    plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("Normalized error (dB)",fontdict=csfont, fontsize=FontSize)
    # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"Err_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+"_SSFN.png",dpi=600)
    plt.close()

    # if data=="MNIST":
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(np.arange(1,len(test_acc_sorted)+1), test_acc_sorted, 'r-', label="Test Accuracy", linewidth=2)
    plt.plot(np.arange(1,len(train_acc_sorted)+1), train_acc_sorted, 'b-', label="Train Accuracy", linewidth=2)
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("Classification accuracy (%)",fontdict=csfont, fontsize=FontSize)
    # plt.title(data+", SSFN", loc='center')
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"Acc_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+".png",dpi=600)
    plt.close()

    return sorted_ind, train_nme_sorted[-1], test_nme_sorted, train_mse_sorted[-1], test_mse_sorted[-1]

def MonteCarlo_NMP(J,Pextra,LA,args):
    """[This function is used for parallel processing when doing Monte Carlo trials. 
    It counts the number of correct index detection across several trials and returns the detection accuracy of NMP for a fixed J and Pextra.]

    Args:
        J ([int]): [Sample size: total number of training and testing samples.]
        Pextra ([int]): [Number of extra random features concatenated with the true features.]
        LA ([str]): [If set to "LookAhead" it will implement LA-NMP. Otherwise, set it to "none".]
        args ([parser]): [It contains the inputs specifies by the user such as name of the dataset, and hyperparameters of the NN.]

    Returns:
        [float]: [Returns correct detection accuracy of NMP.]
    """
    MC_Num = 100
    SSFN_hparameters = set_hparameters(args)
    data = SSFN_hparameters["data"]
    
    if data == "Gravitation":
        true_ind = [0,1,2]
    elif data == "Planck":
        true_ind = [0,1]
    elif data == "Ohm":
        true_ind = [0,1]
    elif data == "NN":
        true_ind = np.arange(0,50)
    elif data == "Artificial":
        true_ind = np.arange(0,50)

    miss_count = 0 

    for iteration in np.arange(1,MC_Num+1):
        # print("J = "+str(J)+", Itration "+str(iteration))
        X_train, X_test, T_train, T_test = define_dataset(args)

        J_subset = np.random.choice(X_train.shape[1], J)
        X_train = X_train[:,J_subset] 
        T_train = T_train[:,J_subset]
        # X_test = X_test[:,:int(round(0.1*J))] 
        # T_test = T_test[:,:int(round(0.1*J))]

        Ntr = X_train.shape[1]
        Nts = X_test.shape[1]
        X_train = np.concatenate((X_train, (10)*np.random.rand(Pextra,Ntr)+10), axis=0)
        X_test = np.concatenate(( X_test, (10)*np.random.rand(Pextra,Nts)+10), axis=0)

        P=X_train.shape[0]
        search_ind = range(P)
        sorted_ind = np.array([],  dtype=int)

        while len(search_ind) > 0:
            train_error_array = np.array([])
            test_error_array = np.array([])
            for i in search_ind:
                if len(sorted_ind)>=1:
                    X_tr = X_train[np.append(sorted_ind,i),:]
                    X_ts = X_test[np.append(sorted_ind,i),:]
                else:
                    X_tr = X_train[[i],:]
                    X_ts = X_test[[i],:]
                train_error, test_error, _, _, _, _ = SSFN(X_tr, X_ts, T_train, T_test, SSFN_hparameters)
                train_error_array = np.append(train_error_array, train_error)
                test_error_array = np.append(test_error_array, test_error)
                
            if LA == "LookAhead":
                if len(test_error_array)>1:
                    i = LookAhead(test_error_array, sorted_ind, search_ind, X_train, X_test, T_train, T_test, SSFN_hparameters)
                else:
                    i = np.argmin(test_error_array)
            else:
                i = np.argmin(test_error_array)
            
            best_ind = search_ind[i]
            sorted_ind = np.append(sorted_ind, best_ind)
            search_ind = np.delete(search_ind, i)
            
            if len(true_ind) == len(sorted_ind):
                diff_ind = np.setdiff1d(true_ind, sorted_ind)
                if len(diff_ind) > 0:
                    miss_count = miss_count + 1
                    # print(sorted_ind)
                    print("For J = "+str(J)+" , Pextra = "+str(Pextra)+" -> Miss count = "+str(miss_count)+" / "+str(iteration))
                break
    
    accuracy = (1 - miss_count/MC_Num) * 100
    return accuracy

def acc_vs_J(_logger,args):
    """[This function plots NMP detection accuracy versus sample size J, refer to Figure 3 and 5 in the Overleaf.]

    Args:
        _logger ([str]): [It can be used to print relative information during the training.]
        args ([parser]): [It contains the inputs specifies by the user such as name of the dataset, and hyperparameters of the NN.]
    """
    parameters_path = "./parameters/"
    result_path = "./results/"

    SampleSize = np.arange(50,1050,50)  # [100,1000,10000]
    LA = "None"

    # Pextra = 8
    # args.data = "Ohm"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Ohm = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for J in SampleSize)
    
    # Pextra = 8
    # args.data = "Planck"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Planck = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for J in SampleSize)
    
    # Pextra = 7
    # args.data = "Gravitation"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Gravitation = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for J in SampleSize)

    Pextra = 0
    args.data = "Artificial"
    _logger.info("The dataset we use is {}".format(args.data))
    accuracy_Artificial = Parallel(n_jobs=1)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for J in SampleSize)
    print(accuracy_Artificial)
    # Pextra = 50
    # args.data = "NN"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_NN = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for J in SampleSize)

    # LA = "LookAhead"

    # args.data = "Ohm"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Ohm_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,_logger,args) for J in SampleSize)
    
    # args.data = "Planck"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Planck_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,_logger,args) for J in SampleSize)
    
    # args.data = "Gravitation"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Gravitation_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,args) for J in SampleSize)

    # Pextra = 50
    # args.data = "NN"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_NN_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for J in SampleSize)
    
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(SampleSize, accuracy_Ohm, 'r-', label="Ohm's law")
    # plt.plot(SampleSize, accuracy_Planck, 'b-', label="Planck's law")
    # # plt.plot(SampleSize, accuracy_NN_LA, 'g:', label="NN model LookAhead")
    # plt.plot(SampleSize, accuracy_Gravitation, 'g:', label="Gravitation law")
    # plt.legend(loc='best')
    # plt.grid()
    # plt.xlabel("Sample Size (J)",fontdict=csfont)
    # plt.ylabel("Detection Accuracy (%)",fontdict=csfont)
    # # plt.title("SSFN", loc='center')
    # # plt.savefig(result_path +"Acc_vs_J_"+args.data+".png")
    # plt.savefig(result_path +"Acc_vs_J.png")
    # plt.close()

    # output_dic = {}
    # output_dic["accuracy_Ohm"]=accuracy_Ohm 
    # output_dic["accuracy_Planck"]=accuracy_Planck
    # output_dic["accuracy_Gravitation"]=accuracy_Gravitation
    # save_dic(output_dic, parameters_path, "three_laws", "accuracy")

def acc_vs_P(_logger,args):
    """[This function plots NMP detection accuracy versus total number of input features (P), refer to Figure 4 in the Overleaf.]

    Args:
        _logger ([str]): [It can be used to print relative information during the training.]
        args ([parser]): [It contains the inputs specifies by the user such as name of the dataset, and hyperparameters of the NN.]
    """
    parameters_path = "./parameters/"
    result_path = "./results/"
    J = 100 
    LA = "None"
    Sweep = np.arange(0,200,10)  # [100,1000,10000]
    
    args.data = "Ohm"
    _logger.info("The dataset we use is {}".format(args.data))
    accuracy_Ohm = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for Pextra in Sweep)
    
    args.data = "Planck"
    _logger.info("The dataset we use is {}".format(args.data))
    accuracy_Planck = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for Pextra in Sweep)
    
    args.data = "Gravitation"
    _logger.info("The dataset we use is {}".format(args.data))
    accuracy_Gravitation = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,Pextra,LA,args) for Pextra in Sweep)
    
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(Sweep+10, accuracy_Ohm, 'r-', label="Ohm's law")
    plt.plot(Sweep+10, accuracy_Planck, 'b-', label="Planck's law")
    plt.plot(Sweep+10, accuracy_Gravitation, 'g-', label="Gravitation law")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Total number of features (P)",fontdict=csfont)
    plt.ylabel("Detection Accuracy (%)",fontdict=csfont)
    plt.title("SSFN", loc='center')
    plt.savefig(result_path +"Acc_vs_P.png", dpi=600)
    plt.close()

def plot_MNIST(_logger,args):
    SSFN_hparameters = set_hparameters(args)

    J = SSFN_hparameters["J"]
    Pextra = SSFN_hparameters["Pextra"]

    X_train, X_test, T_train, T_test = define_dataset(args)
    X_train = X_train[:,:int(round(0.9*J))] 
    T_train = T_train[:,:int(round(0.9*J))]
    X_test = X_test[:,:int(round(0.1*J))] 
    T_test = T_test[:,:int(round(0.1*J))]

    Ntr = X_train.shape[1]
    Nts = X_test.shape[1]
    P=X_train.shape[0]

    parameters_path = "./parameters/"
    result_path = "./results/"
    LA = "None"

    data = SSFN_hparameters["data"]
    LayerNum = SSFN_hparameters["LayerNum"]
    NodeNum = SSFN_hparameters["NodeNum"]
    
    save_name = "sorted_threshold_0995_J_60000"
    my_dic = load_dic( parameters_path, data, save_name)
    print("Read sorted indices")
    # sorted_ind = [195,404,408,272,239,322,296,489,400,341,290,299,464,485,398,570,379,377,596,745,250,214,471,368,447,620,325,109,754,673,92,636,357,383,348,514,566,211,657,511,655,634,352,445,672,306,392,263,430,237,267,327,597,433
    # ,321,80,359,270,455,143,228,292,563,5,179,558,554,304,307,492,468,126,285,380,189,146,374,291,762,598,372,205,287,556,349,293,625,168,569,302,621,34,662,422,680,209,212,312,255,629,68,235,274,677,691,152,772,586
    # ,190,20,429,282,460,550,632,46,722,648,202,319,276,764,17,43,353,727,604,443,575,529,457,18,53,333,651,561,264,580,99,759,421,405,258,82,252,4,50,705,473,573,495,305,572,665,364,527,60,481,58,725,337,746
    # ,131,663,335,480,148,150,462,224,403,286,56,416,265,339,318,323,326,453,342,47,147,366,688,51,459,579,373,531,681,233,667,627,72,28,289,712,155,350,222,22,504,129,395,243,470,35,334,709,175,137,645,779,675,706
    # ,654,1,9,332,161,107,502,269,10,417,338,254,381,442,613,595,605,630,85,647,8,431,160,91,753,266,206,316,780,734,75,410,241,248,36,125,313,521,232,256,637,407,388,513,773,415,423,113,133,721,145,747,389,498
    # ,257,559,710,628,571,661,121,406,452,361,199,699,94,63,678,242,114,783,87,666,494,543,262,724,277,67,491,775,112,78,546,345,397,41,221,726,356,157,541,171,537,340,766,524]

    # sorted_ind = [697,405,490,380,397,400,292,299,239,331,486,464,91,532,258,423,273,318,111,268,541,395,381,473,398,665,655,297,322,217,650,590,209,196,347,178,222,261,430,350,173,713,459,153,316,384,711,555,482,682,597,314,404,213
    # ,485,65,377,25,768,630,557,544,36,122,440,782,500,339,675,389,434,105,327,741,270,501,17,29,176,224,608,188,749,677,11,109,5,136,621,19,382,102,171,668,686,340,663,328,722,117,654,523,80,624,455,723,181,779
    # ,687,238,104,681,696,520,148,4,210,319,220,594,699,453,474,202,545,204,408,6,646,436,443,598,698,192,509,205,680,476,291,460,229,410,174,424,317,259,69,719,300,540,778,452,628,656,263,694,641,371,406,56,256,769
    # ,510,670,228,353,589,266,162,321,236,362,569,576,143,20,643,525,636,60,278,70,364,82,615,489,252,286,511,752,103,403,633,94,275,363,602,386,635,0,57,584,566,134,746,401,755,465,553,492,573,560,407,355,683,8
    # ,186,493,132,542,709,44,672,726,206,674,64,625,365,582,669,262,435,736,415,475,141,330,480,780,163,298,716,388,402,592,561,100,600,552,744,651,97,107,99,197,603,695,191,495,183,168,283,325,79,505,195,308,753,419
    # ,378,470,673,306,429,59,659,248,199,413,310,346,463,648,634,160,748,289,219,198,491,113,535,420,730,295,92,565,101,502,169,471,554,34,498,562,732,619,285,721,527,740,337,515,774,187,55,75,412,763,479,309,374,685,758,267,123,290,237,720,481,446,418,154,35,282,345,71,108,484,67,411
    # ,158,73,246,165,549,45,324,269,530,667,550,369,280,506,609,358,180,354,507,368,124,469,593,451,243,68,620,3,366,271,760,359,649,144,631,156,147,487,31,89,216,293,754,33,372,513,211,627,496,139,223,738,320,296
    # ,254,166,568,342,757,586,348,47,445,313,48,53,93,428,112,214,24,26,563,605,13,151,645,533,287,234,208,288,444,612,129,393,241,559,46,303,184,585,190,114,232,27,392,390,231,537,274,610,704,548,771,333,664,567
    # ,142,336,599,770,127,431,623,85,611,701,7,218,194,449,642,751,536,133,307,118,22,764,72,438,399,61,193,652,647,416,727,227,733,185,756,349,522,676,448,468,394,84,488,767,302,626,373,249,235,546,334,737,577,383
    # ,517,78,312,777,539,572,688,272,175,110,88,323,472,115,591,357,226,708,693,332,521,376,28,477,742,529,580,225,276,215,439,149,182,120,613,551,155,250,457,714,772,130,422,700,518,614,409,579,32,62,255,461,725,375
    # ,12,745,43,244,427,503,63,137,558,707,734,52,706,710,661,703,456,466,432,761,240,622,750,379,145,279,499,341,140,189,87,762,157,632,177,338,257,417,534,543,478,426,335,421,1,58,575,604,783,653,16,776,367,678
    # ,526,729,494,458,170,775,570,679,765,583,596,657,131,81,519,106,773,425,660,264,76,437,387,747,352,644,9,304,658,684,305,759,37,281,2,531,361,607,637,54,414,512,315,343,617,116,450,718,724,467,95,172,692,125
    # ,638,516,528,74,98,588,581,245,587,207,547,462,146,524,23,454,260,164,86,574,629,119,14,10,705,326,616,671,666,38,203,50,766,731,538,351,18,242,90,618,356,578,497,391,640,442,702,128,360,200,344,284,167,96
    # ,152,41,277,126,138,739,691,247,483,743,77,447,717,689,595,735,301,121,370,606,30,230,564,639,201,441,715,21,396,212,161,662,40,508,514,39,251,253,150,728,556,179,221,781,571,233,311,51,265,15,433,712,42,294
    # ,49,601,385,83,690,504,159,135,329,66]

    sorted_ind = [717,405,490,380,272,291,397,400,239,402,296,348,299,514,218,381,322,384,440,567,371,571,430,651,408,628,230,158,458,442,235,179,121,188,358,326,428,451,437,483,340,404,435,399,177,212,271,351,343,547,260,433,181,486
    ,331,634,474,468,655,187,478,516,526,550,624,709,316,273,345,355,457,292,491,249,193,275,310,677,539,359,494,632,294,377,105,417,387,598,413,313,124,579,528,131,612,146,333,481,454,145,298,515,443,679,512,493,529,509
    ,264,285,627,502,686,570,577,318,367,290,321,434,540,546,300,406,556,374,157,706,277,372,215,255,527,95,184,621,164,178,543,563,473,394,631,681,638,690,209,133,665,211,535,269,228,456,663,276,176,524,320,378,582,373
    ,266,388,328,469,349,311,552,553,460,499,237,247,422,411,155,361,470,537,240,370,453,484,441,175,286,620,262,312,232,160,707,375,593,128,163,410,554,354,219,500,633,161,342,495,245,415,429,244,423,357,323,604,510,265
    ,607,385,352,508,545,692,425,159,248,270,648,536,125,284,261,382,622,492,338,459,134,718,122,376,288,596,656,317,565,283,479,658,680,174,584,485,132,305,623,229,609,688,278,542,149,103,401,398,221,657,431,600,210,519
    ,599,217,314,256,661,341,129,203,325,544,201,487,126,664,206,517,569,666,578,438,274,534,436,362,683,200,339,574,147,156,154,472,682,409,414,293,511,685,662,530,189,123,445,306] # for threshold of 0.005 (0.995) for the whole MNIST J = 60000


    show_image((X_train[:,1],X_train[:,20],X_train[:,30]),sorted_ind, save_name)
    # save_name = "random"
    # random_ind = np.arange(0,784)
    # np.random.shuffle(random_ind)
    # show_image(X_train[:,1],X_train[:,20],X_train[:,30], random_ind, save_name)

    ############################################################################################################################################
    ############################################################################################################################################
    ############################################################################################################################################

    train_mse_sorted = my_dic["train_mse_sorted"]
    train_nme_sorted = my_dic["train_nme_sorted"]
    test_mse_sorted = my_dic["test_mse_sorted"]
    test_nme_sorted = my_dic["test_nme_sorted"]
    test_acc_sorted = my_dic["test_acc_sorted"]
    train_acc_sorted = my_dic["train_acc_sorted"]

    print("F-MSE: "+str(train_mse_sorted[-1]))
    print("F-NME: "+str(train_nme_sorted[-1]))
    print("P-MSE: "+str(test_mse_sorted[-1]))
    print("P-NME: "+str(test_nme_sorted[-1]))
    print("Test Acc: "+str(test_acc_sorted[-1]))
    print("Train Acc: "+str(train_acc_sorted[-1]))

    FontSize = 14
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(np.arange(1,P+1), test_error_sorted, 'r-', label="Test", linewidth=2)
    plt.plot(np.arange(1,P+1), train_error_sorted, 'b-', label="Train", linewidth=2)
    plt.legend(loc='best', fontsize=FontSize)
    plt.grid()
    plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("Normalized error (dB)",fontdict=csfont, fontsize=FontSize)
    # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"Err_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+".png",dpi=600)
    plt.close()

    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(np.arange(1,P+1), test_acc_sorted * 100, 'r-', label="Test", linewidth=2)
    plt.plot(np.arange(1,P+1), train_acc_sorted * 100, 'b-', label="Train", linewidth=2)
    plt.legend(loc='best', fontsize=FontSize)
    plt.grid()
    plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("Classification accuracy (%)",fontdict=csfont, fontsize=FontSize)
    # plt.title(data+", SSFN", loc='center')
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"Acc_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+".png",dpi=600)
    plt.close()

def my_plot(_logger,args):
    SampleSize = np.arange(50,1050,50)  # [100,1000,10000]
    parameters_path = "./parameters/"
    result_path = "./results/"

    my_dic = load_dic( parameters_path, "three_laws", "accuracy")
    accuracy_Ohm = my_dic["accuracy_Ohm"]
    accuracy_Planck = my_dic["accuracy_Planck"]
    accuracy_Gravitation = my_dic["accuracy_Gravitation"]

    FontSize = 14
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(SampleSize, accuracy_Ohm, 'r-', label="Ohm law", linewidth=2)
    plt.plot(SampleSize, accuracy_Planck, 'b--', label="Planck law", linewidth=2)
    # plt.plot(SampleSize, accuracy_NN_LA, 'g:', label="NN model LookAhead")
    plt.plot(SampleSize, accuracy_Gravitation, 'g:', label="Gravitation law", linewidth=3)
    plt.legend(loc='best', fontsize=FontSize)
    plt.grid()
    plt.xlabel("Sample size (J)",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("Detection accuracy (%)",fontdict=csfont, fontsize=FontSize)
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"Acc_vs_J.png", dpi=600)
    plt.close()


def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    SSFN_hparameters = set_hparameters(args)
    
    _logger.info("Construct SSFN")
    Err_vs_feat(args)
    # acc_vs_J(_logger,args)
    # acc_vs_P(_logger,args)
    # plot_MNIST(_logger,args)
    # my_plot(_logger,args)

def NMP_train(_logger, X_train, X_test, T_train, T_test, args):
    # args = define_parser()

    # sorted_ind, train_NME, test_NME, train_mse, test_mse = Err_vs_feat_window(X_train, X_test, T_train, T_test, args)
    # sorted_ind, train_NME, test_NME, train_mse, test_mse = Err_vs_feat(X_train, X_test, T_train, T_test, args)
    # acc_vs_J(_logger,args)
    # acc_vs_P(_logger,args)
    plot_MNIST(_logger,args)
    # my_plot(_logger,args)

    return sorted_ind, train_NME, test_NME, train_mse, test_mse 

if __name__ == '__main__':
    main()