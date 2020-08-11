# -*- coding: utf-8 -*-

import logging 
import argparse
from SSFN import SSFN
from MyFunctions import *
from load_dataset import *
import multiprocessing
from joblib import Parallel, delayed

def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="NN", help="Input dataset available as the paper shows")
    parser.add_argument("--lam", type=float, default=10**(2), help="Reguralized parameters on the least-square problem")
    parser.add_argument("--mu", type=float, default=10**(3), help="Parameter for ADMM")
    parser.add_argument("--kMax", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--NodeNum", type=int, default=100, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum", type=int, default=5, help="Parameter for ADMM")
    parser.add_argument("--J", type=int, default=1000, help="Sample Size")
    parser.add_argument("--Pextra", type=int, default=50, help="Number of extra random features")
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
        X_train,X_test, T_train,  T_test  = prepare_mnist()
    elif args.data == "Vowel":
        X_train,X_test, T_train,  T_test  = prepare_vowel()
    elif args.data == "Planck":
        X_train,X_test, T_train,  T_test  = prepare_Planck()
    elif args.data == "Ohm":
        X_train,X_test, T_train,  T_test  = prepare_Ohm()
    elif args.data == "NN":
        X_train,X_test, T_train,  T_test  = prepare_NN()
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
        _, test_error = SSFN(X_tr, X_ts, T_train, T_test, SSFN_hparameters)
        test_error_temp = np.append(test_error_temp, test_error)
    myMin0 = np.min(test_error_temp)  

    sorted_ind1 = np.append(sorted_ind, search_ind[ind_array[1]])
    search_ind1 = np.delete(search_ind, ind_array[1])

    train_error_array = np.array([])
    test_error_temp = np.array([])
    for i in search_ind1:
        if len(sorted_ind1)>=1:
            X_tr = X_train[np.append(sorted_ind1,i),:]
            X_ts = X_test[np.append(sorted_ind1,i),:]
        else:
            X_tr = X_train[[i],:]
            X_ts = X_test[[i],:]
        _, test_error = SSFN(X_tr, X_ts, T_train, T_test, SSFN_hparameters)
        test_error_temp = np.append(test_error_temp, test_error)
    myMin1 = np.min(test_error_temp)  

    if myMin0 <= myMin1:
        LookAhead_ind = ind_array[0]
    else:
        LookAhead_ind = ind_array[1]

    return LookAhead_ind

def Err_vs_feat(args):
    """[This function plots training and testing error versus number of features |S|, refer to Figure 1 and 2 in the Overleaf.]

    Args:
        args ([parser]): [It contains the inputs specifies by the user such as name of the dataset, and hyperparameters of the NN.]
    """
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
    X_train = np.concatenate((X_train, (10)*np.random.rand(Pextra,Ntr)+10), axis=0)
    X_test = np.concatenate(( X_test, (10)*np.random.rand(Pextra,Nts)+10), axis=0)
        
    parameters_path = "./parameters/"
    result_path = "./results/"
    LA = "None"

    data = SSFN_hparameters["data"]
    LayerNum = SSFN_hparameters["LayerNum"]
    NodeNum = SSFN_hparameters["NodeNum"]
    
    P=X_train.shape[0]
    search_ind = range(P)
    train_error_sorted = np.array([])
    test_error_sorted = np.array([])
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
            train_error, test_error = SSFN( X_tr, X_ts, T_train, T_test, SSFN_hparameters)
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
        train_error_sorted = np.append(train_error_sorted, train_error_array[i])
        test_error_sorted = np.append(test_error_sorted, test_error_array[i])
        sorted_ind = np.append(sorted_ind, best_ind)
        search_ind = np.delete(search_ind, i)
        print(sorted_ind)


    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(np.arange(1,P+1), test_error_sorted, 'r-', label="Test NME")
    plt.plot(np.arange(1,P+1), train_error_sorted, 'b-', label="Train NME")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Number of input features",fontdict=csfont)
    plt.ylabel("Normalized Loss (dB)",fontdict=csfont)
    plt.title(data+", SSFN", loc='center')
    plt.savefig(result_path +"LA_Err_vs_index_J"+str(J)+"_L"+str(LayerNum)+"_node"+str(NodeNum)+"_"+data+".png")
    plt.close()

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

    miss_count = 0 

    for iteration in np.arange(1,MC_Num+1):
        # print("P = "+str(Pextra)+", Itration "+str(iteration))
        X_train, X_test, T_train, T_test = define_dataset(args)
        X_train = X_train[:,:int(round(0.9*J))] 
        T_train = T_train[:,:int(round(0.9*J))]
        X_test = X_test[:,:int(round(0.1*J))] 
        T_test = T_test[:,:int(round(0.1*J))]

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
                train_error, test_error = SSFN(X_tr, X_ts, T_train, T_test, SSFN_hparameters)
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
                    print(sorted_ind)
                    print("For J = "+str(J)+" , Pextra = "+str(Pextra)+" -> Miss count = "+str(miss_count))
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

    # args.data = "Ohm"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Ohm = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,_logger,args) for J in SampleSize)
    
    # args.data = "Planck"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Planck = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,_logger,args) for J in SampleSize)
    
    args.data = "Gravitation"
    _logger.info("The dataset we use is {}".format(args.data))
    accuracy_Gravitation = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,args) for J in SampleSize)

    args.data = "NN"
    _logger.info("The dataset we use is {}".format(args.data))
    accuracy_NN = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,args) for J in SampleSize)

    LA = "LookAhead"

    # args.data = "Ohm"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Ohm_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,_logger,args) for J in SampleSize)
    
    # args.data = "Planck"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Planck_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,_logger,args) for J in SampleSize)
    
    # args.data = "Gravitation"
    # _logger.info("The dataset we use is {}".format(args.data))
    # accuracy_Gravitation_LA = Parallel(n_jobs=20)(delayed(MonteCarlo_NMP)(J,0,LA,args) for J in SampleSize)
    
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    # plt.plot(SampleSize, accuracy_Ohm, 'r-', label="Ohm's law")
    # plt.plot(SampleSize, accuracy_Planck, 'b-', label="Planck's law")
    plt.plot(SampleSize, accuracy_NN, 'g-', label="NN model")
    # plt.plot(SampleSize, accuracy_Gravitation_LA, 'g:', label="Gravitation law LookAhead")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Sample Size (J)",fontdict=csfont)
    plt.ylabel("Detection Accuracy (%)",fontdict=csfont)
    plt.title("SSFN", loc='center')
    plt.savefig(result_path +"LA_Acc_vs_J_"+arg.data+".png")
    plt.close()

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
    plt.savefig(result_path +"Acc_vs_P.png")
    plt.close()

def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    SSFN_hparameters = set_hparameters(args)
    
    _logger.info("Construct SSFN")
    # Err_vs_feat(args)
    acc_vs_J(_logger,args)
    # acc_vs_P(_logger,args)

if __name__ == '__main__':
    main()