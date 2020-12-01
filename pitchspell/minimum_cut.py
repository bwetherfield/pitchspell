import numpy as np

def generate_complete_cut(arr):
    """
    Return matrix that is 1 at (i,j) only if arr[i] == 0 and arr[j] == 1.

    Parameters
    ----------
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    N = len(arr)
    indices = np.indices((N, N), sparse=True)
    return np.clip(arr[indices[1]] - arr[indices[0]], 0, 1)