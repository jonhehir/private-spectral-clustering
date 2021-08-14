# Privacy-Preserving Spectral Clustering

## Overview

This code accompanies [a paper by a similar (but longer) title](https://arxiv.org/abs/2105.12615). The main file of interest is `spectral.py`, which includes functions to generate networks from the stochastic block model and the degree-corrected block model, apply the edge-flip privacy mechanism to a network, apply spectral clustering to a network, and measure the accuracy of the clustering results given ground-truth labels. The `run_*.py` files are batch files for running the simulations used in the paper.

## Spectral Clustering Methods

This code works exclusively with *adjacency spectral methods* (i.e., leveraging the eigenvalues of the adjacency matrix, not the graph Laplacian). Specifically, the function `spectral.recover_labels` accepts a `strategy` parameter that works with the following functions:

- `spectral.cluster_kmeans`: Simple *k*-means over the spectral embeddings
    - This is the method used for stochastic block model (SBM) clustering in Lei and Rinaldo.
    - This is also the default method. If you do not specify a strategy, this will be used.
- `spectral.cluster_normalized_kmeans`: *k* means over spectral embeddings, normalized to all have unit norm
- `spectral.cluster_normalized_kmedians`: *k*-medians clustering over spectral embeddings, normalized to all have unit norm
    - This is the method used for degree-corrected block model (DCBM) clustering in Lei and Rinaldo.

## Sources

The methods used in this paper are a modified version of:

> Lei, J., & Rinaldo, A. (2015). *Consistency of spectral clustering in stochastic block models.* Annals of Statistics, 43(1), 215-237.

The datasets included in this repository are sourced from:

**fb100**:

> Traud, A. L., Mucha, P. J., & Porter, M. A. (2012). *Social structure of Facebook networks.* Physica A: Statistical Mechanics and its Applications, 391(16), 4165-4180.

**hansell**:

> Hansell, S. (1984). *Cooperative groups, weak ties, and the integration of peer friendships.* Social Psychology Quarterly, 316-328.
>
> Wang, Y. J., & Wong, G. Y. (1987). *Stochastic blockmodels for directed graphs.* Journal of the American Statistical Association, 82(397), 8-19.

**house/senate**:

See [congress-voting-networks](https://github.com/jonhehir/congress-voting-networks).

**political_blogs**:

> Adamic, L. A., & Glance, N. (2005). *The political blogosphere and the 2004 US election: divided they blog.* In Proceedings of the 3rd international workshop on Link discovery (pp. 36-43).

**sampson**:

> Sampson, S. F. (1968). *A novitiate in a period of change: An experimental and case study of social relationships.* Cornell University.
>
> Hunter, D. R., Handcock, M. S., Butts, C. T., Goodreau, S. M., & Morris, M. (2008). *[`ergm`](https://github.com/statnet/ergm): A package to fit, simulate and diagnose exponential-family models for networks.* Journal of statistical software, 24(3), nihpa54860.

## Dependencies

Python 3 with the following modules:

- pyclustering
- scikit-learn
- SciPy

See requirements.txt for a more formal list.

## Privacy Warning

This code is **not intended for production use**. The random numbers used in these simulations are merely pseudo-random.

