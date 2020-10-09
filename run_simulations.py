import argparse
import spectral
import timeit


def run_simulation(args):
    eps = -1 if args.no_privacy else args.epsilon

    A = spectral.generate_symmetric_sbm(args.nodes, args.blocks, args.p, args.r)
    
    start = timeit.default_timer()
    
    if not args.no_privacy:
        A = spectral.perturb_symmetric(A, eps)
    
    labels = spectral.recover_labels(A, args.k)
    end = timeit.default_timer()
    
    true_lengths = [args.n / args.k] * args.k # eek
    accuracy = spectral.label_accuracy(labels, true_lengths)
    
    print(f"{args.nodes}\t{args.k}\t{args.p}\t{args.r}\t{eps}\t{accuracy}\t{end-start}\n")


parser = argparse.ArgumentParser(description="Run simulations for (private) spectral clustering")
parser.add_argument("--nodes", type=int, help="Number of nodes in each simulated network")
parser.add_argument("--blocks", type=int, help="Number of communities in each network")
parser.add_argument("--r", type=float, help="Probability of edge across different blocks")
parser.add_argument("--p", type=float, help="Additional probability of edge within block (p + r)")
parser.add_argument("--runs", type=int, help="Number of simulations to run")
parser.add_argument("--no-privacy", action='store_true', help="If specified, no privacy will be used")
parser.add_argument("--epsilon", type=float, help="Privacy budget (>0)")

args = parser.parse_args()

for run in range(args.runs):
    run_simulation(args)

