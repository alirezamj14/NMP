import logging 
import argparse
from SSFN import SSFN
from NMP import NMP
from MyFunctions import *
from load_dataset import *
import multiprocessing
from joblib import Parallel, delayed

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
    return X_train, X_test, T_train, T_test


def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    S = [0, 1, 2] # set of true relevant features for gravitational law

    MC_Num=10
    NMP_FPSR = np.zeros((1, MC_Num))
    NMP_FNSR = np.zeros((1, MC_Num))

    for i in np.arange(0,MC_Num):
        X_train, X_test, T_train, T_test = define_dataset(args)
        S_hat = NMP(X_train, X_test, T_train, T_test) # set of selected features  

        NMP_FPSR[0,i] = FPSR(S,S_hat[0:len(S)])
        NMP_FNSR[0,i] = FNSR(S,S_hat[0:len(S)])

    NMP_avg_FPSR = np.mean(NMP_FPSR)
    NMP_avg_FNSR = np.mean(NMP_FNSR)

    print("Average FPSR of NMP: " + str(NMP_avg_FPSR))
    print("Average FNSR of NMP: " + str(NMP_avg_FNSR))
            

if __name__ == '__main__':
    main()