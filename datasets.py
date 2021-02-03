import pathlib

import numpy
from scipy import sparse


def path_to_file(file):
    return pathlib.Path(__file__).parent.absolute().joinpath(file)

def digit(s):
    try:
        n = int(s)
    except ValueError:
        n = None
    
    if str(n) != s:
        return None
    return n

def read_adj(file, n, first_node_index=1):
    """
    Read a whitespace delimited adjacency matrix
    Ignore lines that aren't of the right format (e.g., comments or headers)
    By default, assumes first node index is 1. To use 0-index,
    set `first_node_index` accordingly.
    """
    
    A = sparse.lil_matrix((n, n))
    
    with open(path_to_file(file)) as f:
        for line in f:
            digits = [digit(s) for s in line.strip().split()]
            
            # only use valid lines
            if any([d is None for d in digits]) or len(digits) != 2:
                continue
                
            A[digits[0] - first_node_index, digits[1] - first_node_index] = 1
    
    return A

def symmetrize(A):
    r, c = A.nonzero()
    A[c, r] = 1
    return A

def read_labels(file):
    """
    Read a file with one string label per line and
    return a list of label IDs (0-indexed)
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
                lookup[line] = len(lookup)
            
            # lookup label; append it
            labels.append(lookup[line])
    
    return labels
    
# The datasets:
def fb100(school):
    labels = read_labels(f"datasets/fb100/{school}-nodes.txt")
    A = symmetrize(read_adj(f"datasets/fb100/{school}-edges.txt", len(labels)))
    return A, labels

def hansell():
    labels = read_labels("datasets/hansell/nodes.txt")
    A = symmetrize(read_adj("datasets/hansell/edges.txt", 27))
    return A, labels

def house(congress):
    congress = int(congress)
    labels = read_labels(f"datasets/house/nodes-{congress:03}.tsv")
    A = symmetrize(read_adj(f"datasets/house/edges-{congress:03}.tsv", len(labels)))
    return A, labels

def political_blogs():
    labels = read_labels("datasets/political_blogs/blogs-orientation.txt")
    A = read_adj("datasets/political_blogs/blogs.txt", len(labels))
    return A, labels

def sampson():
    labels = read_labels("datasets/sampson/nodes.txt")
    A = read_adj("datasets/sampson/edges.txt", len(labels))
    return A, labels

def senate(congress):
    congress = int(congress)
    labels = read_labels(f"datasets/senate/nodes-{congress:03}.tsv")
    A = symmetrize(read_adj(f"datasets/senate/edges-{congress:03}.tsv", len(labels)))
    return A, labels
