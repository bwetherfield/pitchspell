import numpy as np
from .pullback import pullback

def skip_adjacency(hops, size):
    indices = np.indices((size, size))
    abs_differences = np.abs(indices[0] - indices[1])
    return np.equal(abs_differences, hops).astype('int')