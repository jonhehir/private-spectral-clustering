import itertools
from math import exp

import numpy as np
from pyclustering.cluster.kmedians import kmedians
from scipy import sparse, stats
from sklearn import cluster, metrics


# Hardcoded number of random initializations to try for k-medians
N_KMEDIANS_INITIALIZATIONS = 10


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
    """
    A special case of SBM where:
    - blocks are equally sized (n/k nodes each)
    - within-block edge probability = p + r
    - across-block edge probability = r
    """
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

def normalize_rows(U):
    """
    Normalize rows of U to have unit norm
    If a given row has norm == 0, it is left alone
    """
    n = U.shape[0]
    row_norms = np.linalg.norm(U, axis=1).reshape((n, 1))
    
    # safely divide by norms, leaving original entries if row norm == 0
    return np.divide(U, row_norms, out=U, where=(row_norms > 0))

def preprocess_to_laplacian(A, eps, reg=False):
    """
    Returns a Laplacian (L = D^-0.5 A D^-0.5) given a symmetric adjacency matrix A
    """
    degrees = np.squeeze(np.asarray(A.sum(axis=1)))
    if reg:
        degrees += np.mean(degrees)
    D = sparse.diags(degrees ** -0.5)
    return D @ A @ D

def preprocess_recenter(M, eps):
    """
    Subtracts 1/(e^eps + 1) from off-diagonals of M
    """
    M = M.todense()
    p = perturb_prob(eps)
    return M - p + np.diag([p] * M.shape[0])

def cluster_kmeans(U, k):
    """
    Cluster U by simple k-means
    Return labels
    """
    kmeans = cluster.KMeans(n_clusters=k)
    return kmeans.fit(U).labels_

def cluster_normalized_kmeans(U, k):
    """
    Cluster U by k-means over row-normalized version of U
    Return labels
    """
    return cluster_kmeans(normalize_rows(U), k)

def cluster_normalized_kmedians(U, k):
    """
    Cluster U by k-medians over row-normalized version of U
    Return labels
    """
    n = U.shape[0]
    U_norm = normalize_rows(U)
    best_labels = None
    best_error = float("inf")

    # Run k-medians multiple times with random initializations, then return best result
    for i in range(N_KMEDIANS_INITIALIZATIONS):
        # for initial centers, choose k points at random
        indices = np.random.choice(n, k, replace=False)
        initial_centers = U_norm[indices, :]
        
        # run k-medians
        instance = kmedians(U_norm.tolist(), initial_centers.tolist())
        instance.process()
        
        # get cluster labels (in the format we want them)
        labels = [None] * n
        i = 0
        for c in instance.get_clusters():
            for j in c:
                labels[j] = i
            i += 1
        
        # calculate error
        medians = np.array(instance.get_medians())
        error = np.sum(np.linalg.norm(U_norm - medians[labels, :], axis=1))

        if error < best_error:
            best_labels = labels
            best_error = error

    return best_labels

def recover_labels(A, k, strategy=cluster_kmeans):
    """
    Employ spectral clustering to recover labels for A
    """
    U = sparse.linalg.eigsh(A, k)[1]
    return strategy(U, k)

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
