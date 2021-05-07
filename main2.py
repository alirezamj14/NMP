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
    parser.add_argument("--data", default="artificial", help="Input dataset available as the paper shows")
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
    elif args.data == "artificial":
        X_train, X_test, T_train,  T_test  = prepare_artificial()
    return X_train, X_test, T_train, T_test


def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    S = [0, 1, 2, 3, 4] # set of true relevant features
    sweep_eta = 0.01 * np.arange(1,11)
    sweep_J = np.arange(50, 1050, 50)

    NMP_avg_FPSR = np.array([])
    NMP_avg_FNSR = np.array([])    

    MC_Num=5

    for J in sweep_J:
        args.eta = 0.06
        # print("eta: "+ str(eta))
        print("J: "+ str(J))

        NMP_FPSR = np.zeros((1, MC_Num))
        NMP_FNSR = np.zeros((1, MC_Num))

        for i in np.arange(0,MC_Num):
            X_train, X_test, T_train, T_test = define_dataset(args)
            S_hat = NMP_train(X_train[:,0:J], X_test, T_train[:,0:J], T_test, args)       # set of selected features  
            # print(S_hat)

            NMP_FPSR[0,i] = FPSR(S,S_hat)
            NMP_FNSR[0,i] = FNSR(S,S_hat)

            # print(NMP_FPSR)
            # print(NMP_FNSR)

        NMP_avg_FPSR = np.append(NMP_avg_FPSR, np.mean(NMP_FPSR))
        NMP_avg_FNSR = np.append(NMP_avg_FNSR, np.mean(NMP_FNSR))

        print("Average FPSR of NMP: " + str(NMP_avg_FPSR))
        print("Average FNSR of NMP: " + str(NMP_avg_FNSR))
    
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