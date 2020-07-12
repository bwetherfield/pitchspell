import numpy as np


def hop_adjacencies(hops, size):
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


def add_node(arr, row_pos=None, col_pos=None, in_edges=0, out_edges=0):
    if row_pos is None and col_pos is not None:
        row_pos = col_pos
    elif col_pos is None and row_pos is not None:
        col_pos = row_pos
    elif row_pos is None and col_pos is None:
        row_pos = arr.shape[0]
        col_pos = arr.shape[1]
    output = np.insert(arr, row_pos, out_edges, axis=0)
    return np.insert(output, col_pos, np.append(in_edges, 0), axis=1)
