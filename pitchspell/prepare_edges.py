import numpy as np


def hop_adjacency(hops, size):
    indices = np.indices((size, size))
    abs_differences = np.abs(indices[0] - indices[1])
    return np.equal(abs_differences, hops).astype('int')


def concurrencies(starts, ends, size=None):
    if size is None:
        size = len(starts)
    indices = np.indices((size, size))
    overlaps = np.logical_or(
        np.logical_and(
            starts[indices[0]] >= starts[indices[1]],
            starts[indices[0]] < ends[indices[1]]
        ),
        np.logical_and(
            starts[indices[1]] >= starts[indices[0]],
            starts[indices[1]] < ends[indices[0]]
        ))
    return overlaps.astype('int') - np.eye(size, dtype='int')
