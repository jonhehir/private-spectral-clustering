import argparse
import datetime
import multiprocessing
import sys
import timeit

import numpy as np

import datasets, spectral


def run_dataset(args):
    # generate random seed explicitly each time
    np.random.seed()
    
    dataset, eps = args.dataset, args.epsilon
    
    # allow arguments to pass to dataset function
    fn_parts = dataset.split(":")
    fn = fn_parts.pop(0)

    # find and run the function whose name matches `dataset` in the datasets module
    A, true_labels = getattr(datasets, fn)(*fn_parts)
    k = max(true_labels) + 1
    
    start = timeit.default_timer()
    
    if eps >= 0:
        A = spectral.perturb_symmetric(A, eps)
    
    labels = spectral.recover_labels(A, k)
    end = timeit.default_timer()
    
    accuracy = spectral.label_accuracy(labels, true_labels)
    
    return [datetime.datetime.now(), dataset, eps, accuracy, end-start]

def print_result(result):
    print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run private spectral clustering on dataset")
parser.add_argument("--dataset", type=str, help="Name of dataset (function in datasets module)")
parser.add_argument("--runs", type=int, help="Number of simulations to run")
parser.add_argument("--epsilon", type=float, help="Privacy budget (>=0 will use privacy, <0 will use no privacy)")

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    jobs = [pool.apply_async(run_dataset, (args,), callback=print_result)
            for _ in range(args.runs)]
    for job in jobs:
        job.get() # for the sake of re-raising any exceptions in the child process

    pool.close()
    pool.join()
