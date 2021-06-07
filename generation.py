import numpy as np
from scipy import sparse, stats


# Stochastic Block Model
# Note: These functions were originally implemented with sparse matrix methods.
# The original versions of the functions remain (e.g., `generate_sparse_sbm`),
# but they're no longer called directly.

def generate_sparse_block(size, prob, symmetric=False):
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

def generate_sparse_sbm(block_sizes, block_probs):
    """
    Generate a stochastic block model using fixed block sizes and connectivity matrix
    """
    k = len(block_sizes)
    blocks = [[None for i in range(k)] for j in range(k)]
    
    for i in range(k):
        for j in range(i, k):
            blocks[i][j] = generate_sparse_block(
                (block_sizes[i], block_sizes[j]),
                block_probs[i][j],
                symmetric=(i == j)
            )
            if i < j:
                blocks[j][i] = blocks[i][j].transpose()
    
    return sparse.bmat(blocks)

def generate_sparse_symmetric_sbm(n, k, p, r):
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
    
    return generate_sparse_sbm(block_sizes, block_probs)

def generate_block(size, prob, symmetric=False):
    return generate_sparse_block(size, prob, symmetric=symmetric).toarray()

def generate_sbm(block_sizes, block_probs):
    """
    Like `generate_sparse_sbm` but returns a dense array
    """
    return generate_sparse_sbm(block_sizes, block_probs).toarray()

def generate_symmetric_sbm(n, k, p, r):
    """
    Like `generate_sparse_symmetric_sbm` but returns a dense array
    """
    return generate_sparse_symmetric_sbm(n, k, p, r).toarray()

# Degree-Corrected Block Model

def _sbm_to_dcbm(sbm, weights):
    """
    Given an array of `weights` corresponding to each node,
    convert an SBM to a DCBM with these weights.
    """

    # Given a network Y ~ SBM(theta, B),
    # we can construct a network Y' ~ DCBM(theta, B, psi)
    # by taking Y'_ij = Bernoulli(psi_i * psi_j) * Y_ij .

    W = np.random.default_rng().binomial(1, np.outer(weights, weights))
    return np.multiply(W, sbm)

def generate_symmetric_dcbm(n, k, p, r, a):
    """
    Generates a random DCBM where:
    - blocks are equally sized (n/k nodes each)
    - max. within-block edge probability = p + r
    - max. across-block edge probability = r
    - node probability weights are randomly drawn from Uniform(a, 1)
    """

    sbm = generate_symmetric_sbm(n, k, p, r)

    # In order to ensure the maximum node weight for each block is 1, we set
    # the first node to have weight 1, then draw the rest from Uniform(a, 1).
    weights = np.random.default_rng().uniform(a, 1, n)
    for i in range(k):
        weights[i * n // k] = 1
    
    return _sbm_to_dcbm(sbm, weights)
