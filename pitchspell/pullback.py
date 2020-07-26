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


def pad(new_shape, arr, abs_idx, rel_idx=None):
    """
    Map relative indices `rel_idx` to new positions in a larger matrix of shape
    `(new_size, new_size)` with the remaining entries 0 (padded).

    Parameters
    ----------
    new_shape: int or tuple of ints
    arr: numpy.ndarray
    abs_idx: numpy.ndarray
    rel_idx: numpy.ndarray or None

    Returns
    -------
    numpy.ndarray

    """
    if rel_idx is None:
        rel_idx = np.indices(arr.shape)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    new_arr[abs_idx] = arr[rel_idx]
    return new_arr


def stretch(arr):
    """
    Take an NxM matrix to an NxNM matrix such that the new matrix
    multiplied by an NM column of variables produces the same constraints
    as the original matrix multiplied by an MxN matrix of variables.

    Parameters
    ----------
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    rows = arr.shape[0]
    columns = arr.shape[1]
    stretched_arr = np.zeros((rows, rows * columns), dtype='int')
    for i in range(rows):
        stretched_arr[i, i * np.indices((columns,))] = arr[i]
    return stretched_arr
