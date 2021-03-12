# Privacy-Preserving Spectral Clustering

## Overview

This code accompanies a paper by a similar (but longer) title. The main file of interest is `spectral.py`, which includes functions to generate networks from a stochastic block model, apply the edge-flip privacy mechanism to a network, apply spectral clustering to a network, and measure the accuracy of the clustering results given ground-truth labels. The `run_*.py` files are batch files for running the simulations used in the paper.

## Primary Reference

The methods used in this paper are from:

> Lei, J., & Rinaldo, A. (2015). *Consistency of spectral clustering in stochastic block models.* Annals of Statistics, 43(1), 215-237.

## Spectral Clustering Methods

This code works exclusively with *adjacency spectral methods* (i.e., leveraging the eigenvalues of the adjacency matrix, not the graph Laplacian). Specifically, the function `spectral.recover_labels` accepts a `strategy` parameter that works with the following functions:

- `spectral.cluster_kmeans`: Simple *k*-means over the spectral embeddings
    - This is the method used for stochastic block model (SBM) clustering in Lei and Rinaldo.
    - This is also the default method. If you do not specify a strategy, this will be used.
- `spectral.cluster_normalized_kmeans`: *k* means over spectral embeddings, normalized to all have unit norm
- `spectral.cluster_normalized_kmedians`: *k*-medians clustering over spectral embeddings, normalized to all have unit norm
    - This is the method used for degree-corrected block model (DCBM) clustering in Lei and Rinaldo.

## Dependencies

Python 3 with the following modules:

- pyclustering
- scikit-learn
- SciPy

See requirements.txt for a more formal list.

## Privacy Warning

This code is **not intended for production use**. The random numbers used in these simulations are merely pseudo-random.

