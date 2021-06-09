import argparse
import datetime
import itertools
import multiprocessing
import timeit

import numpy as np

import generation, spectral


def sparsify(p, n, sparsity):
    return p * n**(-sparsity)

def run_simulation(name, n, k, p, r, a, eps, sparsity):
    # generate random seed explicitly each time
    np.random.seed()
    
    if sparsity > 0:
        p = sparsify(p, n, sparsity)
        r = sparsify(r, n, sparsity)

    if a < 1:
        A = generation.generate_symmetric_dcbm(n, k, p, r, a)
        strategy = spectral.cluster_normalized_kmedians
    else:
        A = generation.generate_symmetric_sbm(n, k, p, r)
        strategy = spectral.cluster_kmeans
    
    start = timeit.default_timer()
    
    if eps >= 0:
        A = spectral.preprocess_recenter(spectral.perturb_symmetric(A, eps), eps)
    
    labels = spectral.recover_labels(A, k, strategy=strategy)
    end = timeit.default_timer()
    
    true_lengths = [n // k] * k # eek, this is awful
    accuracy = spectral.simulation_label_accuracy(labels, true_lengths)
    
    return [name, datetime.datetime.now(), n, k, p, r, a, sparsity, eps, accuracy, end-start]

def print_result(result):
    print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run simulations for (private) spectral clustering")
parser.add_argument("--name", type=str, help="Name of simulation setting", default="simulation")
parser.add_argument("--runs", type=int, help="Number of simulations to run", default=1)
parser.add_argument("--n", type=int, nargs="+", help="Number of nodes in each simulated network")
parser.add_argument("--k", type=int, help="Number of blocks in each simulated network")
parser.add_argument("--p", type=float, help="Edge probability parameter 'p'")
parser.add_argument("--r", type=float, help="Edge probability parameter 'r'")
parser.add_argument("--a", type=float, help="For DCBM, specify a value between 0 and 1", default=1.0)
parser.add_argument("--sparsity", type=float, help="If set, p and r will be multiplied by n^-sparsity", default=0.0)
parser.add_argument("--epsilon", nargs="+", type=float, help="Privacy budget (>=0 will use privacy, <0 will use no privacy)", default=[-1.0])

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    jobs = []
    
    for (n, eps, _) in itertools.product(args.n, args.epsilon, range(args.runs)):
        a = (args.name, n, args.k, args.p, args.r, args.a, eps, args.sparsity)
        jobs.append(pool.apply_async(run_simulation, a, callback=print_result))

    for job in jobs:
        job.get() # for the sake of re-raising any exceptions in the child process

    pool.close()
    pool.join()
