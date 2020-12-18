import pathlib

import numpy
from scipy import sparse


def path_to_file(file):
    return pathlib.Path(__file__).parent.absolute().joinpath(file)

def digit(s):
    n = int(s)
    if str(n) != s:
        return None
    return n

def read_adj(file, n):
    """
    Read a whitespace delimited adjacency matrix
    Ignore lines that aren't of the right format (e.g., comments or headers)
    """
    
    A = sparse.csr_matrix((n, n))
    
    with open(path_to_file(file)) as f:
        for line in f:
            digits = [digit(s) for s in line.strip().split()]
            
            # only use valid lines
            if any([d is None for d in digits]) or len(d) != 2:
                continue
                
            A[d[0], d[1]] = 1
    
    return A

def read_labels(file):
    """
    Read a file with one string label per line and
    return a list of label IDs (1-indexed)
    """
    
    lookup = {}
    labels = []
    
    with open(path_to_file(file)) as f:
        for line in f:
            line = line.strip()
            
            # skip blank lines
            if line == "":
                continue
                
            # assign an ID to this label if it doesn't exist
            if line not in lookup:
                lookup[line] = len(lookup) + 1
            
            # lookup label; append it
            labels.append(lookup[line])
    
    return labels

def political_blogs():
    labels = read_labels("datasets/political_blogs/blogs-orientation.txt")
    A = read_adj("datasets/political_blogs/blogs.txt", len(labels))
    return A, labels

