import argparse
import datetime
import multiprocessing
import spectral
import timeit


def run_simulation(args):
    n, k, p, r, eps = args.nodes, args.blocks, args.p, args.r, args.epsilon

    A = spectral.generate_symmetric_sbm(n, k, p, r)
    
    start = timeit.default_timer()
    
    if eps >= 0:
        A = spectral.perturb_symmetric(A, eps)
    
    labels = spectral.recover_labels(A, k)
    end = timeit.default_timer()
    
    true_lengths = [n // k] * k # eek, this is awful
    accuracy = spectral.label_accuracy(labels, true_lengths)
    
    return [datetime.datetime.now(), n, k, p, r, eps, accuracy, end-start]

def print_result(result):
    print("\t".join([str(x) for x in result]))


parser = argparse.ArgumentParser(description="Run simulations for (private) spectral clustering")
parser.add_argument("--nodes", type=int, help="Number of nodes in each simulated network")
parser.add_argument("--blocks", type=int, help="Number of communities in each network")
parser.add_argument("--r", type=float, help="Probability of edge across different blocks")
parser.add_argument("--p", type=float, help="Additional probability of edge within block (p + r)")
parser.add_argument("--runs", type=int, help="Number of simulations to run")
parser.add_argument("--epsilon", type=float, help="Privacy budget (>=0 will use privacy, <0 will use no privacy)")

args = parser.parse_args()

with multiprocessing.Pool() as pool:
    for run in range(args.runs):
        pool.apply_async(run_simulation, (args,), callback=print_result)

    pool.close()
    pool.join()
