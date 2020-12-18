import argparse
import datetime
import multiprocessing
import timeit

import numpy as np

import datasets
import spectral


def run_dataset(args):
    # generate random seed explicitly each time
    np.random.seed()
    
    dataset, eps = args.dataset, args.epsilon

    # find and run the function whose name matches `dataset` in the datasets module
    A, labels = getattr(datasets, dataset)()
    k = max(labels) + 1
    
    start = timeit.default_timer()
    
    if eps >= 0:
        A = spectral.perturb_symmetric(A, eps)
    
    labels = spectral.recover_labels(A, k)
    end = timeit.default_timer()
    
    accuracy = spectral.label_accuracy(labels, truth)
    
    return [datetime.datetime.now(), dataset, eps, accuracy, end-start]

def print_result(result):
    print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run private spectral clustering on dataset")
parser.add_argument("--dataset", type=str, help="Name of dataset (function in datasets module)")
parser.add_argument("--runs", type=int, help="Number of simulations to run")
parser.add_argument("--epsilon", type=float, help="Privacy budget (>=0 will use privacy, <0 will use no privacy)")

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    for run in range(args.runs):
        pool.apply_async(run_dataset, (args,), callback=print_result)

    pool.close()
    pool.join()
