import numpy as np


def hop_adjacencies(hops, size):
    """
    Generate adjacency matrix where every nodes with indices that are `hops`
    distance apart are connected.

    Parameters
    ----------
    hops: int
    size: int

    Returns
    -------
    numpy.ndarray

    """
    indices = np.indices((size, size))
    abs_differences = np.abs(indices[0] - indices[1])
    return np.equal(abs_differences, hops).astype('int')


def concurrencies(starts, ends, size=None):
    """
    Generate adjacency array based on events that overlap, given their start
    and end times.

    Parameters
    ----------
    starts: numpy.ndarray
    ends: numpy.ndarray
    size: int or None

    Returns
    -------

    """
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


def add_node(arr, pos=None, in_edges=0, out_edges=0):
    """
    Add node to an adjacency matrix given its position (assumed at the end
    of the list of nodes), its incident edges and its outgoing edges.

    Parameters
    ----------
    arr: numpy.ndarray
    pos: int
    in_edges: int or numpy.ndarray
    out_edges: int or numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    if pos is None:
        pos = arr.shape[0]
    output = np.insert(arr, pos, out_edges, axis=0)
    return np.insert(output, pos, np.append(in_edges, 0), axis=1)
