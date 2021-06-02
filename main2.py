import logging 
import argparse
from matplotlib.pyplot import axis
import numpy as np
from numpy import array
from SSFN import SSFN
from NMP import NMP_train
from test_masked_train import return_patched_data
from MyFunctions import *
from load_dataset import *
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt 

def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="CIFAR10", help="Input dataset available as the paper shows")
    parser.add_argument("--lam", type=float, default=10**(2), help="Reguralized parameters on the least-square problem")
    parser.add_argument("--mu", type=float, default=10**(3), help="Parameter for ADMM")
    parser.add_argument("--kMax", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--NodeNum", type=int, default=100, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum", type=int, default=1, help="Parameter for ADMM")
    parser.add_argument("--J", type=int, default=60000, help="Sample Size")
    parser.add_argument("--Pextra", type=int, default=0, help="Number of extra random features")
    parser.add_argument("--eta", type=float, default=0.06, help="Stopping criterion for NMP")
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
    elif args.data == "Boston":
        X_train, X_test, T_train,  T_test  = prepare_Boston()
    elif args.data == "Airfoil":
        X_train, X_test, T_train,  T_test  = prepare_Airfoil()
    elif args.data == "CIFAR10":
        X_train, X_test, T_train,  T_test  = prepare_cifar10()
    elif args.data == "Modelnet10":
        X_train, X_test, T_train,  T_test  = prepare_Modelnet10()
    return X_train, X_test, T_train, T_test

def parallel_Acc_vs_J(_logger, J, args):
    print("J: "+ str(J))
    MC_Num=args.MC_Num

    NMP_NME_matrix = np.zeros((5, MC_Num))
    NMP_NME_matrix_given = np.zeros((5, MC_Num))
    NMP_FPSR = np.zeros((1, MC_Num))
    NMP_FNSR = np.zeros((1, MC_Num))
    NMP_PNME = np.zeros((1, MC_Num))
    NMP_FNME = np.zeros((1, MC_Num))
    NMP_PMSE = np.zeros((1, MC_Num))
    NMP_FMSE = np.zeros((1, MC_Num))

    miss_count = 0

    for i in np.arange(0,MC_Num):
        X_train, X_test, T_train, T_test = define_dataset(args)
        J_train = np.random.choice(X_train.shape[1], int(round(0.9*J)), replace=False)
        J_test = np.random.choice(X_test.shape[1], int(round(0.1*J)), replace=False)
        args.flag = "Not_given"
        S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(_logger, X_train[:,:], X_test[:,:], T_train[:,:], T_test[:,:], args)       # set of selected features
        # S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(_logger, X_train, X_test, T_train, T_test, args)       # set of selected features
        # args.flag = "given_order"
        # args.best_ind = np.flip(S_hat)
        # args.best_ind = np.random.choice([0,1,2,3,4], 5, replace=False)
        # S_hat, _, test_NME_given, _, _  = NMP_train(_logger, X_train[:,J_train], X_test[:,:], T_train[:,J_train], T_test[:,:], args)
        # S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(X_train, X_test, T_train, T_test, args)       # set of selected features  
        # print(S_hat)
        # print(S_hat_given)

        # NMP_FPSR[0,i] = FPSR(S,S_hat)
        NMP_FNSR[0,i] = FNSR(args.S,S_hat)
        # NMP_PNME[0,i] = test_NME[-1]
        # NMP_FNME[0,i] = train_NME
        # NMP_PMSE[0,i] = test_mse
        # NMP_FMSE[0,i] = train_mse

        # NMP_NME_matrix[:,i] = test_NME
        # NMP_NME_matrix_given[:,i] = test_NME_given
        # print(NMP_NME_matrix)
        # print(NMP_NME_matrix_given)

        # print(NMP_FPSR)
        # print(NMP_FNSR)
        # print(NMP_PNME)
        # print(NMP_FNME)
        # print(NMP_PMSE)
        # print(NMP_FMSE)

        if len(args.S) == len(S_hat):
            diff_ind = np.setdiff1d(args.S, S_hat, assume_unique=True)
            if len(diff_ind) > 0:
                miss_count = miss_count + 1
                # print(S_hat)
                # print("For J = "+str(J)+" -> Miss count = "+str(miss_count)+" / "+str(i+1))

    accuracy = (1 - miss_count/MC_Num) * 100

    return (accuracy, np.mean(NMP_FNSR))

def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    # _logger.info("The dataset we use is {}".format(args.data))

    
    
    S = [0, 1] # 5, 6, 7, 8, 9, 10, 11, 12] # set of true relevant features
    sweep_eta = 0.01 * np.arange(1,11)
    sweep_J = np.arange(1000, 1050, 50)

    MC_Num = 1
    args.MC_Num = MC_Num
    datasets = ["CIFAR10"]

    for data in datasets:
        args.data = data
        print("The dataset we use is " + data)

        if data == "Planck":
            args.S = [0,1]
        elif data == "Ohm":
            args.S = [0,1]
        elif data == "Gravitation":
            args.S = [0,1,2]
        elif data == "Artificial":
            args.S = [0,1,2,3,4]

        NMP_avg_FPSR = np.array([])
        NMP_avg_FNSR = np.array([])    
        NMP_avg_PNME = np.array([])
        NMP_avg_FNME = np.array([])    
        NMP_avg_PMSE = np.array([])
        NMP_avg_FMSE = np.array([])
        NMP_accuracy = np.array([]) 

        for J in sweep_J:
            args.eta = 0.06
            # print("eta: "+ str(eta))
            # print("J: "+ str(J))

            NMP_NME_matrix = np.zeros((5, MC_Num))
            NMP_NME_matrix_given = np.zeros((5, MC_Num))
            NMP_FPSR = np.zeros((1, MC_Num))
            NMP_FNSR = np.zeros((1, MC_Num))
            NMP_PNME = np.zeros((1, MC_Num))
            NMP_FNME = np.zeros((1, MC_Num))
            NMP_PMSE = np.zeros((1, MC_Num))
            NMP_FMSE = np.zeros((1, MC_Num))

            miss_count = 0

            for i in np.arange(0,MC_Num):
                X_train, X_test, T_train, T_test = define_dataset(args)

                # X_train_mean = np.mean(X_train, axis=1 )
                # X_test_mean = np.mean(X_test, axis=1 )
                # X_train_mean = np.sort(X_train_mean)
                # X_test_mean = np.sort(X_test_mean)
                # FontSize = 18
                # result_path = "./results/"
                # csfont = {'fontname':'sans-serif'}
                # plt.subplots()
                # plt.plot(X_train_mean, 'r-', label="Train", linewidth=3)
                # plt.plot(X_test_mean, 'b-', label="Test", linewidth=2)
                # plt.legend(loc='best', fontsize=FontSize)
                # plt.grid()
                # plt.xlabel("Feature index",fontdict=csfont, fontsize=FontSize)
                # plt.ylabel("Sum",fontdict=csfont, fontsize=FontSize)
                # plt.xticks(fontsize=FontSize)
                # plt.yticks(fontsize=FontSize)
                # plt.tight_layout()
                # plt.savefig(result_path + data +"_sum_grey_new"+".png",dpi=600)
                # plt.close()

                J_train = np.random.choice(X_train.shape[1], int(round(J)), replace=False)
                J_test = np.random.choice(X_test.shape[1], int(round(0.1*J)), replace=False)
                args.flag = "Not_given"
                S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(_logger, X_train[:,:], X_test[:,:], T_train[:,:], T_test[:,:], args)       # set of selected features
                # S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(_logger, X_train, X_test, T_train, T_test, args)       # set of selected features
                # args.flag = "given_order"
                # args.best_ind = np.flip(S_hat)
                # args.best_ind = np.random.choice([0,1,2,3,4], 5, replace=False)
                # S_hat, _, test_NME_given, _, _  = NMP_train(_logger, X_train[:,J_train], X_test[:,:], T_train[:,J_train], T_test[:,:], args)
                # S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(X_train, X_test, T_train, T_test, args)       # set of selected features  
                # print(S_hat)
                # print(S_hat_given)

                # NMP_FPSR[0,i] = FPSR(args.S,S_hat)
                # NMP_FNSR[0,i] = FNSR(args.S,S_hat)
                # NMP_PNME[0,i] = test_NME[-1]
                # NMP_FNME[0,i] = train_NME
                # NMP_PMSE[0,i] = test_mse
                # NMP_FMSE[0,i] = train_mse

                # NMP_NME_matrix[:,i] = test_NME
                # NMP_NME_matrix_given[:,i] = test_NME_given
                # print(NMP_NME_matrix)
                # print(NMP_NME_matrix_given)

                # print(NMP_FPSR)
                # print(NMP_FNSR)
                # print(NMP_PNME)
                # print(NMP_FNME)
                # print(NMP_PMSE)
                # print(NMP_FMSE)

            #     if len(args.S) == len(S_hat):
            #         diff_ind = np.setdiff1d(args.S, S_hat, assume_unique=True)
            #         if len(diff_ind) > 0:
            #             miss_count = miss_count + 1
            #             # print(S_hat)
            #             # print("For J = "+str(J)+" -> Miss count = "+str(miss_count)+" / "+str(i+1))

            # accuracy = (1 - miss_count/MC_Num) * 100         
            # NMP_accuracy = np.append(NMP_accuracy, accuracy)
            
            # NMP_avg_FPSR = np.append(NMP_avg_FPSR, np.mean(NMP_FPSR))
            # NMP_avg_FNSR = np.append(NMP_avg_FNSR, np.mean(NMP_FNSR))
            # NMP_avg_PNME = np.append(NMP_avg_PNME, np.mean(NMP_PNME))
            # NMP_avg_FNME = np.append(NMP_avg_FNME, np.mean(NMP_FNME))
            # NMP_avg_PMSE = np.append(NMP_avg_PMSE, np.mean(NMP_PMSE))
            # NMP_avg_FMSE = np.append(NMP_avg_FMSE, np.mean(NMP_FMSE))

            # NMP_NME_avg = np.mean(NMP_NME_matrix, axis = 1)
            # NMP_NME_avg_given = np.mean(NMP_NME_matrix_given, axis = 1)
            # print(NMP_NME_avg)
            # print(NMP_NME_avg_given)

        #########################################################################################################################
        #########################################################################################################################
        #########################################################################################################################

        # _logger.info("The dataset we use is {}".format(args.data))
        # (NMP_accuracy, NMP_avg_FNSR) = zip(*Parallel(n_jobs=20)(delayed(parallel_Acc_vs_J)(_logger, J,args) for J in sweep_J))

        #########################################################################################################################
        #########################################################################################################################
        #########################################################################################################################   

        # print("Average FPSR of NMP: " + str(NMP_avg_FPSR))
        # print("Average FNSR of NMP: " + str(NMP_avg_FNSR))
        # print("Average PNME of NMP: " + str(NMP_avg_PNME))
        # print("Average FNME of NMP: " + str(NMP_avg_FNME))
        # print("Average PMSE of NMP: " + str(NMP_avg_PMSE))
        # print("Average FMSE of NMP: " + str(NMP_avg_FMSE))

        # print("Average dACC of NMP: " + str(NMP_accuracy))

    # FontSize = 18
    # result_path = "./results/"
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(sweep_J, NMP_avg_FPSR, 'r-', label="FPSR", linewidth=3)
    # plt.plot(sweep_J, NMP_avg_FNSR, 'b-', label="FNSR", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.grid()
    # # plt.xlabel("Stopping threshold (eta)",fontdict=csfont, fontsize=FontSize)
    # plt.xlabel("Number of samples (J)",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("False selection rate",fontdict=csfont, fontsize=FontSize)
    # # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path +"FPSR_&_FNSR_vs_J"+".png",dpi=600)
    # plt.close()

    
    # test_NME_ture = [ -4.42711219,  -7.09753944,  -9.24551936, -10.89538164, -12.12525705]
    # test_NME_reversed = [-3.15987956,  -4.55117262,  -6.05654299,  -7.69701609, -12.20626252]
    # test_NME_random = [-3.28400436,  -4.91835142,  -6.72199312,  -8.40703431, -11.85620162]

    # FontSize = 18
    # result_path = "./results/"
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot([1,2,3,4,5], test_NME_reversed, 'r--', label="Reversed", linewidth=2)
    # plt.plot([1,2,3,4,5], test_NME_random, 'g:', label="Random", linewidth=2)
    # plt.plot([1,2,3,4,5], test_NME_ture, 'b-', label="NGP", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.grid()
    # # plt.xlabel("Stopping threshold (eta)",fontdict=csfont, fontsize=FontSize)
    # plt.xlabel("Number of input features",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Normalized Error (dB)",fontdict=csfont, fontsize=FontSize)
    # # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path +"NGP_vs_reversed_vs_random"+".png",dpi=600)
    # plt.close()

    
    
    # NMP_avg_FNSR_Artificial = [0.5, 0.14, 0.02, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # for a full test set
    # NMP_avg_FNSR_Gravitation = [0.29, 0.1733, 0.0633, 0.04, 0.033, 0.033, 0, 0, 0, 0, 0, 0, 0, 0]
    # NMP_avg_FNSR_Planck = [0.15, 0.04, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # NMP_avg_FNSR_Ohm = [0.16, 0.105, 0.05, 0.03, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0]
    # # NMP_avg_FNSR_Planck = [0.095, 0.175, 0.115, 0.045, 0.02, 0.03, 0.065, 0.015, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.01, 0.0, 0.01, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # for J in np.arange(50, 510, 10)
    # NMP_avg_FNSR_Ohm = [0.155, 0.09, 0.07, 0.08, 0.05, 0.045, 0.0, 0.03, 0.0, 0.015, 0.03, 0.0, 0.0, 0.01, 0.02, 0.0, 0.01, 0.01, 0.0, 0.0, 0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # NMP_avg_FNSR_Gravitation = [0.23333333333333328, 0.24, 0.22, 0.17, 0.2033333333333333, 0.17666666666666667, 0.12666666666666665, 0.09666666666666666, 0.12999999999999998, 0.07666666666666666, 0.07999999999999999, 0.09, 0.07333333333333333, 0.06666666666666667, 0.05333333333333333, 0.04333333333333333, 0.03333333333333333, 0.026666666666666665, 0.03666666666666666, 0.019999999999999997, 0.05, 0.026666666666666665, 0.016666666666666666, 0.03666666666666667, 0.03333333333333333, 0.01, 0.026666666666666665, 0.01, 0.006666666666666666, 0.016666666666666666, 0.026666666666666665, 0.0, 0.0, 0.013333333333333332, 0.006666666666666666, 0.013333333333333332, 0.0, 0.006666666666666666, 0.006666666666666666, 0.006666666666666666, 0.006666666666666666, 0.0, 0.0, 0.013333333333333332, 0.0, 0.003333333333333333]
    # NMP_avg_FNSR_Artificial = [0.148, 0.066, 0.066, 0.07, 0.05, 0.030000000000000006, 0.028000000000000004, 0.032, 0.027999999999999997, 0.026000000000000002, 0.022000000000000002, 0.024000000000000004, 0.008, 0.018000000000000002, 0.014000000000000002, 0.014000000000000002, 0.016, 0.012, 0.006000000000000001, 0.013999999999999999, 0.002, 0.006000000000000001, 0.004, 0.008, 0.004, 0.008, 0.006000000000000001, 0.004, 0.002, 0.004, 0.002, 0.006000000000000001, 0.002, 0.004, 0.012, 0.002, 0.008, 0.002, 0.01, 0.006000000000000001, 0.006000000000000001, 0.0, 0.002, 0.0, 0.0, 0.004]
    
    # FontSize = 18
    # result_path = "./results/"
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(sweep_J, NMP_avg_FNSR_Planck, 'm--', label="Planck", linewidth=2)
    # plt.plot(sweep_J, NMP_avg_FNSR_Ohm, 'r-', label="Ohm", linewidth=2)
    # plt.plot(sweep_J, NMP_avg_FNSR_Gravitation, 'b:', label="Gravitation", linewidth=2)
    # plt.plot(sweep_J, NMP_avg_FNSR_Artificial, 'g-.', label="Artificial", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.grid()
    # # plt.xlabel("Stopping threshold (eta)",fontdict=csfont, fontsize=FontSize)
    # plt.xlabel("Number of samples (J)",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("FNSR",fontdict=csfont, fontsize=FontSize)
    # # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path +"FNSR_vs_J"+".png",dpi=600)
    # plt.close()

    # sweep_J = np.arange(50, 710, 10)
    # # NMP_FullR_Ohm = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1]
    # # NMP_FullR_Gravitation = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # NMP_FullR_Planck = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # NMP_FullR_Artificial = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # NMP_FullR_Planck =        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # NMP_FullR_Ohm =           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # NMP_FullR_Gravitation =   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # NMP_FullR_Artificial =    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    # FontSize = 18
    # result_path = "./results/"
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(sweep_J, NMP_FullR_Planck, 'm--', label="Planck", linewidth=2)
    # plt.plot(sweep_J, NMP_FullR_Ohm, 'r-', label="Ohm", linewidth=2)
    # plt.plot(sweep_J, NMP_FullR_Gravitation, 'b:', label="Gravitation", linewidth=2)
    # plt.plot(sweep_J, NMP_FullR_Artificial, 'g-.', label="Artificial", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.grid()
    # # plt.xlabel("Stopping threshold (eta)",fontdict=csfont, fontsize=FontSize)
    # plt.xlabel("Number of samples (J)",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Full Recovery",fontdict=csfont, fontsize=FontSize)
    # # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path +"FullR_vs_J"+".png",dpi=600)
    # plt.close()

    
    # NMP_ACC_Planck = [89.0, 80.0, 87.0, 95.0, 97.0, 97.0, 93.0, 98.0, 98.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 100.0, 99.0, 99.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # NMP_ACC_Ohm = [83.0, 89.0, 93.0, 92.0, 94.0, 95.0, 100.0, 97.0, 100.0, 98.0, 96.0, 100.0, 100.0, 99.0, 98.0, 100.0, 99.0, 99.0, 100.0, 100.0, 99.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # NMP_ACC_Gravitation = [48.0, 50.0, 54.0, 67.0, 61.0, 65.0, 73.0, 80.0, 75.0, 84.0, 85.0, 81.0, 85.0, 87.0, 88.0, 91.0, 94.0, 96.0, 92.0, 96.0, 91.0, 95.0, 97.0, 93.0, 94.0, 98.0, 96.0, 97.0, 99.0, 97.0, 96.0, 100.0, 100.0, 98.0, 99.0, 97.0, 100.0, 99.0, 99.0, 99.0, 99.0, 100.0, 100.0, 98.0, 100.0, 99.0]
    # NMP_ACC_Artificial = [39.0, 70.0, 71.0, 68.0, 77.0, 85.0, 87.0, 85.0, 86.0, 87.0, 90.0, 88.0, 96.0, 91.0, 93.0, 94.0, 92.0, 94.0, 97.0, 93.0, 99.0, 97.0, 98.0, 96.0, 98.0, 96.0, 97.0, 98.0, 99.0, 98.0, 99.0, 97.0, 99.0, 98.0, 94.0, 99.0, 96.0, 99.0, 95.0, 97.0, 97.0, 100.0, 99.0, 100.0, 100.0, 98.0]

    # FontSize = 18
    # result_path = "./results/"
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(sweep_J, NMP_ACC_Planck, 'm--', label="Planck", linewidth=2)
    # plt.plot(sweep_J, NMP_ACC_Ohm, 'r-', label="Ohm", linewidth=2)
    # plt.plot(sweep_J, NMP_ACC_Gravitation, 'b:', label="Gravitation", linewidth=2)
    # plt.plot(sweep_J, NMP_ACC_Artificial, 'g-.', label="Artificial", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.grid()
    # # plt.xlabel("Stopping threshold (eta)",fontdict=csfont, fontsize=FontSize)
    # plt.xlabel("Number of samples (J)",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Detection Accuracy (%)",fontdict=csfont, fontsize=FontSize)
    # # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path +"dACC_vs_J"+".png",dpi=600)
    # plt.close()


if __name__ == '__main__':
    main()