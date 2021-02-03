import argparse
import datetime
import multiprocessing
import timeit

import numpy as np

import spectral


def sparsify(p, n):
    return p * n**(-.3)

def run_simulation(args):
    # generate random seed explicitly each time
    np.random.seed()
    
    n, eps = args.nodes, args.epsilon
    
    k = 2
    p = sparsify(1.5, n)
    r = sparsify(.15, n)

    A = spectral.generate_symmetric_sbm(n, k, p, r)
    
    start = timeit.default_timer()
    
    if eps >= 0:
        A = spectral.perturb_symmetric(A, eps)
    
    labels = spectral.recover_labels(A, k)
    end = timeit.default_timer()
    
    true_lengths = [n // k] * k # eek, this is awful
    accuracy = spectral.simulation_label_accuracy(labels, true_lengths)
    
    return [datetime.datetime.now(), n, k, p, r, eps, accuracy, end-start]

def print_result(result):
    print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run simulations for (private) spectral clustering")
parser.add_argument("--nodes", type=int, help="Number of nodes in each simulated network")
parser.add_argument("--runs", type=int, help="Number of simulations to run")
parser.add_argument("--epsilon", type=float, help="Privacy budget (>=0 will use privacy, <0 will use no privacy)")

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    jobs = [pool.apply_async(run_simulation, (args,), callback=print_result)
            for _ in range(args.runs)]
    for job in jobs:
        job.get() # for the sake of re-raising any exceptions in the child process

    pool.close()
    pool.join()
