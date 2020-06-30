import numpy as np


def pullback(f, shape, arr):
    idx = np.indices(shape)
    mapped_idx = f(idx)
    return arr(tuple(mapped_idx))


def pullback(mapping, arr=None):
    if arr is None:
        n = mapping.max() + 1
        arr = np.eye(n, dtype='int')
    N = len(mapping)
    return arr[tuple(mapping[np.indices((N, N), dtype='int')])]
