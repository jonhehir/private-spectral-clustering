# Privacy-Preserving Spectral Clustering

## Overview

This code accompanies a paper by a similar (but longer) title. The main file of interest is `spectral.py`, which includes functions to generate networks from a stochastic block model, apply the edge-flip privacy mechanism to a network, apply spectral clustering to a network, and measure the accuracy of the clustering results given ground-truth labels. The `run_*.py` files are batch files for running the simulations used in the paper.

## Dependencies

Python 3, SciPy, scikit-learn

## Privacy Warning

This code is **not intended for production use**. The random numbers used in these simulations are merely pseudo-random.

