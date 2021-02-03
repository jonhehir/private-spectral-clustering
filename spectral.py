import itertools
from math import exp

from scipy import sparse, stats
from sklearn import cluster, metrics


def generate_block(size, prob, symmetric=False):
    """
    Generates a random block of binary entries where each entry is 1 w.p. prob
    If symmetric=True, returns a symmetric block with a zero on the diagonal.
    """
    density = stats.binom.rvs(size[0] * size[1], prob, size=1).item() / (size[0] * size[1])
    m = sparse.random(size[0], size[1], density)
    m.data[:] = 1
    
    if symmetric:
        if size[0] != size[1]:
            raise RuntimeError("symmetric matrix must be square")
        m = sparse.triu(m, k=1) + sparse.triu(m, k=1).transpose()
    
    return m

def generate_sbm(block_sizes, block_probs):
    """
    Generate a stochastic block model using fixed block sizes and connectivity matrix
    """
    k = len(block_sizes)
    blocks = [[None for i in range(k)] for j in range(k)]
    
    for i in range(k):
        for j in range(i, k):
            blocks[i][j] = generate_block(
                (block_sizes[i], block_sizes[j]),
                block_probs[i][j],
                symmetric=(i == j)
            )
            if i < j:
                blocks[j][i] = blocks[i][j].transpose()
    
    return sparse.bmat(blocks)

def generate_symmetric_sbm(n, k, p, r):
    if n % k > 0:
        raise RuntimeError("n must be divisble by k to have equal-sized blocks")
    
    block_probs = [
        [ p + r if i == j else r for i in range(k) ]
        for j in range(k)
    ]
    block_sizes = [ n // k ] * k
    
    return generate_sbm(block_sizes, block_probs)
    
def perturb_prob(eps):
    """
    P(perturb edge)
    """
    return 1 / (exp(eps) + 1)

def perturb_symmetric(m, eps):
    """
    Perturb a symmetric adjacency matrix using edge flips
    Note: This retains a zero on the diagonal.
    """
    p = perturb_prob(eps)
    error = generate_block(m.get_shape(), p, symmetric = True)
    return abs(m - error)

def recover_labels(A, k):
    """
    Employ spectral clustering to recover labels for A
    """
    eigs = sparse.linalg.eigsh(A, k)[1]
    kmeans = cluster.KMeans(n_clusters=k)
    return kmeans.fit(A).labels_

def simulation_label_accuracy(labels, lengths):
    """
    max(accuracy) over the set of all label permutations for a simulation,
    where nodes are ordered by block (so just knowing the length of each block is sufficient)
    """
    k = len(lengths)
    
    truth = []
    for i in range(k):
        truth.extend([i] * lengths[i]) # perhaps not perfectly pythonic
    
    return label_accuracy(labels, truth)

def label_accuracy(labels, truth):
    """
    max(accuracy) over the set of all label permutations
    truth should be an list of 0-indexed integer labels of length n
    """
    accuracy = 0
    k = max(truth) + 1 # number of labels
    
    # This is not optimal, but we're using small k, so it's no biggie.
    for p in itertools.permutations(range(k)):
        compare = [p[t] for t in truth]
        accuracy = max(accuracy, metrics.accuracy_score(labels, compare))
    
    return accuracy
