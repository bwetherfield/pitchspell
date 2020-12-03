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


def add_node(arr, in_edges=None, out_edges=None):
    """
    Add node to an adjacency matrix given its position (assumed at the end
    of the list of nodes), its incident edges and its outgoing edges.

    Parameters
    ----------
    arr: numpy.ndarray
    in_edges: None or numpy.ndarray
    out_edges: None or numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    out_edges = [0] if out_edges is None else np.append(out_edges, 0)
    in_edges = [0] if in_edges is None else np.append(in_edges, [0, 0])
    output = np.zeros(np.array(arr.shape) + 1, dtype=arr.dtype)
    output[:-1, :-1] = arr
    output[-1] = out_edges
    output[:, -1] = in_edges
    return output
