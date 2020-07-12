import numpy as np


def f_inverse(f, shape, arr):
    """
    Returns a numpy array of the specified dimensions, defined by mapping
    each index to an index of `arr` using `f`.

    Parameters
    ----------
    f: Callable
    shape: tuple
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    idx = np.indices(shape)
    mapped_idx = f(idx)
    return arr[tuple(mapped_idx)]


def pullback(mapping, arr=None):
    """
    Returns a numpy array by mapping each index to its value in `mapping`,
    using these values to index into `arr` (or an identity matrix with size
    inferred if no array is supplied).

    Parameters
    ----------
    mapping: Indexable (e.g. 1d numpy array)
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    if arr is None:
        n = mapping.max() + 1
        arr = np.eye(n, dtype='int')
    N = len(mapping)
    return arr[tuple(mapping[np.indices((N, N), dtype='int')])]


def pad(abs_idx, rel_idx, length, arr):
    arr_plus = np.insert(np.insert(arr,
                                   arr.shape[1], 0, axis=1),
                         arr.shape[0], 0, axis=0)
    mapping = np.full(length, -1, dtype='int')
    mapping[abs_idx] = rel_idx
    return pullback(mapping, arr_plus)

