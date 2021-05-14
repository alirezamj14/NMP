import logging 
import argparse
import numpy as np
from numpy import array
from SSFN import SSFN
from NMP import NMP_train
from test_masked_train import return_patched_data
from MyFunctions import *
from load_dataset import *
import multiprocessing
from joblib import Parallel, delayed

def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="Boston", help="Input dataset available as the paper shows")
    parser.add_argument("--lam", type=float, default=10**(2), help="Reguralized parameters on the least-square problem")
    parser.add_argument("--mu", type=float, default=10**(3), help="Parameter for ADMM")
    parser.add_argument("--kMax", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--NodeNum", type=int, default=100, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum", type=int, default=1, help="Parameter for ADMM")
    parser.add_argument("--J", type=int, default=1000, help="Sample Size")
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
    return X_train, X_test, T_train, T_test


def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    S = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # set of true relevant features
    sweep_eta = 0.01 * np.arange(1,11)
    sweep_J = np.arange(1000, 1050, 50)

    NMP_avg_FPSR = np.array([])
    NMP_avg_FNSR = np.array([])    
    NMP_avg_PNME = np.array([])
    NMP_avg_FNME = np.array([])    
    NMP_avg_PMSE = np.array([])
    NMP_avg_FMSE = np.array([])    

    MC_Num=10

    for J in sweep_J:
        args.eta = 0.005
        # print("eta: "+ str(eta))
        print("J: "+ str(J))

        NMP_FPSR = np.zeros((1, MC_Num))
        NMP_FNSR = np.zeros((1, MC_Num))
        NMP_PNME = np.zeros((1, MC_Num))
        NMP_FNME = np.zeros((1, MC_Num))
        NMP_PMSE = np.zeros((1, MC_Num))
        NMP_FMSE = np.zeros((1, MC_Num))

        for i in np.arange(0,MC_Num):
            X_train, X_test, T_train, T_test = define_dataset(args)
            J_subset = np.random.choice(X_train.shape[1], J)
            # S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(X_train[:,J_subset], X_test, T_train[:,J_subset], T_test, args)       # set of selected features  
            S_hat, train_NME, test_NME, train_mse, test_mse  = NMP_train(X_train, X_test, T_train, T_test, args)       # set of selected features  
            # print(S_hat)

            NMP_FPSR[0,i] = FPSR(S,S_hat)
            NMP_FNSR[0,i] = FNSR(S,S_hat)
            NMP_PNME[0,i] = test_NME
            NMP_FNME[0,i] = train_NME
            NMP_PMSE[0,i] = test_mse
            NMP_FMSE[0,i] = train_mse

            print(NMP_FPSR)
            print(NMP_FNSR)
            print(NMP_PNME)
            print(NMP_FNME)
            print(NMP_PMSE)
            print(NMP_FMSE)

        NMP_avg_FPSR = np.append(NMP_avg_FPSR, np.mean(NMP_FPSR))
        NMP_avg_FNSR = np.append(NMP_avg_FNSR, np.mean(NMP_FNSR))
        NMP_avg_PNME = np.append(NMP_avg_PNME, np.mean(NMP_PNME))
        NMP_avg_FNME = np.append(NMP_avg_FNME, np.mean(NMP_FNME))
        NMP_avg_PMSE = np.append(NMP_avg_PMSE, np.mean(NMP_PMSE))
        NMP_avg_FMSE = np.append(NMP_avg_FMSE, np.mean(NMP_FMSE))

        print("Average FPSR of NMP: " + str(NMP_avg_FPSR))
        print("Average FNSR of NMP: " + str(NMP_avg_FNSR))
        print("Average PNME of NMP: " + str(NMP_avg_PNME))
        print("Average FNME of NMP: " + str(NMP_avg_FNME))
        print("Average PMSE of NMP: " + str(NMP_avg_PMSE))
        print("Average FMSE of NMP: " + str(NMP_avg_FMSE))

    FontSize = 18
    result_path = "./results/"
    csfont = {'fontname':'sans-serif'}
    plt.subplots()
    plt.plot(sweep_J, NMP_avg_FPSR, 'r-', label="FPSR", linewidth=3)
    plt.plot(sweep_J, NMP_avg_FNSR, 'b-', label="FNSR", linewidth=2)
    plt.legend(loc='best', fontsize=FontSize)
    plt.grid()
    # plt.xlabel("Stopping threshold (eta)",fontdict=csfont, fontsize=FontSize)
    plt.xlabel("Number of samples (J)",fontdict=csfont, fontsize=FontSize)
    plt.ylabel("False selection rate",fontdict=csfont, fontsize=FontSize)
    # plt.title(data+", SSFNN", loc='center', fontsize=FontSize)
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.tight_layout()
    plt.savefig(result_path +"FPSR_&_FNSR_vs_J"+".png",dpi=600)
    plt.close()


if __name__ == '__main__':
    main()