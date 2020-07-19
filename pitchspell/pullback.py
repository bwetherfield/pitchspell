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


def pad(abs_idx, rel_idx, new_size, arr):
    """
    Map relative indices `rel_idx` to new positions in a larger matrix of shape
    `(new_size, new_size)` with the remaining entries 0 (padded).

    Parameters
    ----------
    abs_idx: numpy.ndarray
    rel_idx: numpy.ndarray
    new_size: int
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    arr_plus = np.insert(np.insert(arr,
                                   arr.shape[1], 0, axis=1),
                         arr.shape[0], 0, axis=0)
    mapping = np.full(new_size, -1, dtype='int')
    mapping[abs_idx] = rel_idx
    return pullback(mapping, arr_plus)


def square_index(arr):
    """
    Take an NxN matrix to an N^2xN^2 matrix such that the new matrix
    multiplied by an N^2 column of variables produces the same constraints
    as the original matrix multiplied by an NxN matrix of variables. Also
    produces NxN - N zero constraints. Row i is mapped to position (i*N+i,i*N).

    Parameters
    ----------
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    N = arr.shape[0]
    N_sq = N*N
    squared_arr = np.zeros((N_sq, N_sq), dtype='int')
    for i in range(N):
        mask = np.zeros((N, N), dtype='int')
        mask[i] = np.ones(N)
        idx = np.indices((N,))[0]
        squared_arr += pad(idx + N * i, idx, N_sq, arr * mask)
    return squared_arr