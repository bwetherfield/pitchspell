import numpy as np

from pitchspell.minimum_cut import generate_complete_cut
from pitchspell.prepare_edges import add_node, hop_adjacencies, concurrencies
from pitchspell.pullback import pullback, f_inverse

n_pitch_classes = 12
n_pitch_class_internal_nodes = 24  # n_pitch_classes * 2
n_pitch_class_nodes = 26  # n_pitch_class_internal_nodes + 2
n_pitch_class_edges = 676  # pow(pitch_class_nodes, 2)


def generate_bounds(pre_calculated_weights, internal_scheme, source_edge_scheme,
                    sink_edge_scheme, n_variables):
    """
    Generate the upper and lower bounds allowable for each variable.
    Pitch-based edge weights are clamped according to an adjacency scheme if
    edge weight schemes are not precalculated.

    Parameters
    ----------
    pre_calculated_weights: bool
    internal_scheme: ndarray
    source_edge_scheme: ndarray
    sink_edge_scheme: ndarray
    n_variables: int

    Returns
    -------
    list[(float, Union[float, None])]

    """
    if pre_calculated_weights:
        bounds = (0, None)
    else:
        ub = np.full((n_variables), None)
        weight_upper_bounds = generate_weight_upper_bounds(internal_scheme,
                                                           sink_edge_scheme,
                                                           source_edge_scheme)
        ub[-n_pitch_class_edges:] = weight_upper_bounds.flatten()
        bounds = list(zip(np.zeros_like(ub, dtype=int), ub))
    return bounds


def generate_weight_upper_bounds(internal_scheme, sink_edge_scheme,
                                 source_edge_scheme):
    """
    Generate the upper bounds allowable for each variable. Pitch-based edge
    weights are clamped according to an adjacency scheme if
    edge weight schemes are not precalculated.

    Parameters
    ----------
    internal_scheme: ndarray
    sink_edge_scheme: ndarray
    source_edge_scheme: ndarray

    Returns
    -------
    ndarray

    """
    pc_scheme = internal_scheme
    pc_scheme_idx = np.arange(n_pitch_classes) * 2
    pc_source_edges = np.zeros(n_pitch_class_internal_nodes, dtype=int)
    pc_source_edges[pc_scheme_idx] = source_edge_scheme
    pc_sink_edges = np.zeros(n_pitch_class_internal_nodes + 1, dtype=int)
    pc_sink_edges[pc_scheme_idx + 1] = sink_edge_scheme
    pc_scheme = add_node(pc_scheme, out_edges=pc_source_edges)
    pc_scheme = add_node(pc_scheme, in_edges=pc_sink_edges)
    weight_upper_bounds = 26 * np.clip(pc_scheme, 0, 1)
    return weight_upper_bounds


def generate_cost_func(accuracy, pre_calculated_weights, n_edges, n_variables):
    """
    Minimize duality gap while maximizing the sum of the edge scheme weights (
    if computed). `accuracy` weights the extent to which duality gap is
    prioritized over the sum over all edge scheme weights. Larger accuracy
    results in smaller duality gap (closer to optimal flow / cut values).

    Parameters
    ----------
    accuracy: float
    pre_calculated_weights: bool
    n_edges: int
    n_variables: int

    Returns
    -------
    ndarray (1D)

    """
    c = np.zeros((n_variables), dtype=int)
    c[2 * n_edges] = accuracy
    if not pre_calculated_weights:
        c[-n_pitch_class_edges:] = -1
    return c


def generate_capacities_def(pre_calculated_weights, big_M, n_edges,
                            n_internal_nodes, n_nodes, weighted_adj,
                            pitched_information):
    """
    Definitions for capacity variables for edges in terms of pitch class
    relations. c(i,j) proportional to w(p(i), p(j)) where w is either fixed or
    variable depending on the value of `pre_calculated_weights`.

    Parameters
    ----------
    pre_calculated_weights: bool
    big_M: ndarray
    n_edges: int
    n_internal_nodes: int
    n_nodes: int
    weighted_adj: ndarray
    pitched_information: ndarray

    Returns
    -------
    ndarray (2D), ndarray (1D)

    """
    if pre_calculated_weights:
        weight_scalers = pitched_information
        capacities_def, capacities_def_rhs = \
            generate_capacities_def_weights_fixed(big_M, n_edges, weighted_adj,
                                                  weight_scalers)
    else:
        pitch_classes = pitched_information
        capacities_def, capacities_def_rhs = \
            generate_capacities_def_weights_variable(
                big_M, n_edges, n_internal_nodes, n_nodes, pitch_classes,
                weighted_adj)
    return capacities_def, capacities_def_rhs


def generate_capacities_def_weights_variable(big_M, n_edges, n_internal_nodes,
                                             n_nodes, pitch_classes,
                                             weighted_adj):
    """
    Generate constraints of the form

        c_i,j - a(i,j) * w_(p(i), p(j)) = 0
\\
    where a(i,j) indicates adjacency between i and j in the graph and p(i)
    denotes the pitch value associated with the node i. Generates constraint
    coefficients and RHS separately.

    Parameters
    ----------
    big_M: ndarray
    n_edges: int
    n_internal_nodes: int
    n_nodes: int
    pitch_classes: ndarray
    weighted_adj: ndarray

    Returns
    -------
    ndarray (2D), ndarray (1d)

    """
    capacities_def = np.eye(n_edges, dtype=float)
    pc_idx = pitch_classes * 2 + np.arange(n_internal_nodes) % 2
    source_pitch_class_index = n_pitch_class_internal_nodes
    sink_pitch_class_index = n_pitch_class_internal_nodes + 1
    pc_idx_with_src_sink = np.append(pc_idx, [source_pitch_class_index,
                                              sink_pitch_class_index])
    pc_edge_2d_idx = pc_idx_with_src_sink[np.indices((n_nodes, n_nodes))]
    pc_edge_1d_idx = (
            pc_edge_2d_idx[0] * n_pitch_class_nodes
            + pc_edge_2d_idx[1]
    ).flatten()
    pitch_based_capacity = np.zeros((n_edges, n_pitch_class_edges),
                                    dtype=float)
    pitch_based_capacity[
        np.arange(n_edges),
        pc_edge_1d_idx
    ] = weighted_adj.flatten()
    capacities_def = np.concatenate([
        capacities_def, -pitch_based_capacity
    ], axis=1)
    # RHS
    capacities_def_rhs = big_M.flatten()
    return capacities_def, capacities_def_rhs


def generate_capacities_def_weights_fixed(big_M, n_edges, weighted_adj,
                                          weight_scalers):
    """
    Generate constraints of the form

        c_{i,j} = a(i,j) * w_{(p(i), p(j))}
\\
    where a(i,j) indicates adjacency between i and j in the graph and p(i)
    denotes the pitch value associated with the node i. Generates constraint
    coefficients and RHS separately.

    Parameters
    ----------
    big_M: ndarray
    n_edges: int
    weighted_adj: ndarray
    weight_scalers: ndarray

    Returns
    -------
    tuple(ndarray (2D), ndarray (1D))

    """
    # c_i,j = a(i,j) * w_(p(i), p(j))
    capacities_def = np.eye(n_edges, dtype=float)
    # RHS
    capacities_def_rhs = (weighted_adj * weight_scalers + big_M).flatten()
    return capacities_def, capacities_def_rhs


def generate_flow_conditions(adj, n_internal_nodes, n_nodes):
    """
    Generate conditions of the form

        sum_(i=1)^n f_(i,k) - sum_(j=1)^n f_(k,j) = 0 for all k != s,t
\\
    specifying that the flow into each internal node equals the flow out of it.

    Parameters
    ----------
    adj: ndarray
    n_internal_nodes: int
    n_nodes: int

    Returns
    -------
    ndarray (2D), ndarray (1D)

    """
    internal_nodes = np.arange(n_internal_nodes)

    row_mask = np.zeros((n_internal_nodes, n_nodes, n_nodes), dtype=int)
    row_mask[internal_nodes, internal_nodes] = 1

    col_mask = np.zeros((n_internal_nodes, n_nodes, n_nodes), dtype=int)
    col_mask[internal_nodes, :, internal_nodes] = 1

    flow_conditions = (row_mask[:] - col_mask[:]) * adj
    flow_conditions = flow_conditions.reshape(n_internal_nodes, -1)
    # RHS
    flow_conditions_rhs = np.zeros((n_internal_nodes), dtype=int)
    return flow_conditions, flow_conditions_rhs


def get_weight_scalers(source_edge_scheme, sink_edge_scheme, internal_scheme,
                       half_internal_nodes, n_internal_nodes, pitch_classes):
    """
    Get weight scalers based on the pitches of the notes in the score.

    Parameters
    ----------
    source_edge_scheme: ndarray
    sink_edge_scheme: ndarray
    internal_scheme: ndarray
    half_internal_nodes: int
    n_internal_nodes: int
    pitch_classes: ndarray

    Returns
    -------
    ndarray

    """
    idx = np.indices((half_internal_nodes,), dtype=int) * 2
    source_edges = source_edge_scheme[pitch_classes]
    source_edges[idx + 1] = 0
    sink_edges = sink_edge_scheme[pitch_classes]
    sink_edges[idx] = 0
    sink_edges = np.append(sink_edges, [0])
    weight_scalers = internal_scheme[
        tuple(
            (pitch_classes * 2 + np.arange(n_internal_nodes) % 2)[
                np.indices((n_internal_nodes, n_internal_nodes))
            ]
        )
    ]
    weight_scalers *= cut_2_by_2_diagonal(n_internal_nodes)
    weight_scalers = add_node(weight_scalers, out_edges=source_edges)
    weight_scalers = add_node(weight_scalers, in_edges=sink_edges)
    return weight_scalers


def cut_2_by_2_diagonal(n):
    """
    Set 2 by 2 block diagonal to 0 in matrix otherwise all 1's.

    Parameters
    ----------
    n: int

    Returns
    -------
    ndarray

    """
    return np.logical_not(
        f_inverse(lambda x: x // 2, (n, n), np.eye(n // 2, dtype=int))
    ).astype(int)


def generate_duality_constraint(cut, n_internal_nodes):
    """
    Generate constraint of the form

        sum_(i=1)^n f_s,i - sum_(e in cut) c_e = delta
\\
    which defines the "duality gap" between flow and cut values. Generates
    constraint coefficients and RHS separately.

    Parameters
    ----------
    cut: ndarray
    n_internal_nodes: int

    Returns
    -------
    ndarray (2D), ndarray (1D)

    """
    duality_constraint = np.concatenate([
        -np.ones(n_internal_nodes, dtype=int), cut.flatten(), [-1]
    ])
    # RHS
    duality_constraint_rhs = [0]
    return duality_constraint, duality_constraint_rhs


def get_big_M_edges(half_internal_nodes):
    """
    Output the "big M" component graph which connects the up node and the
    down node corresponding to a single note in the musical score by a
    directed infinite weight edge.

    Parameters
    ----------
    half_internal_nodes

    Returns
    -------
    ndarray

    """
    adj_within = np.tile(
        [[0, 0], [1, 0]],
        [half_internal_nodes, half_internal_nodes]
    ) * np.repeat(
        np.repeat(
            np.eye(half_internal_nodes, dtype=int),
            2,
            axis=1),
        2,
        axis=0)
    adj_within = add_node(adj_within)
    adj_within = add_node(adj_within)
    big_M = np.array(adj_within, dtype=float)
    big_M[np.isclose(big_M, 1)] = np.inf
    return adj_within, big_M


def extract_cut(adj, y):
    """
    Extract cut from node values and adjacency structure of a network.

    Parameters
    ----------
    adj: ndarray (2D)
    y: ndarray (1D)

    Returns
    -------
    ndarray

    """
    y_plus_source_sink = np.concatenate([y, [0, 1]])
    cut = generate_complete_cut(y_plus_source_sink) * adj
    return cut


def generate_internal_cut_constraints(adj, n_internal_nodes):
    """
    Generate cut constraints of the form

        x_j - x_i - y_(i,j) <= 0 for all i, j != s, t, where (i,j) is an edge

    Parameters
    ----------
    adj: ndarray
    n_internal_nodes: int

    Returns
    -------
    ndarray

    """
    sel = (slice(None, -2), slice(None, -2))
    return generate_cut_constraints(adj, n_internal_nodes, sel, True, True)


def generate_cut_constraints(adj, n_internal_nodes, sel, internal_sources,
                             internal_dests):
    """
    Generate a general cut constraint of e.g. one of the following forms

        x_j - x_i - y_(i,j) <= 0 for all i, j != s, t, where (i,j) is an edge

        x_i - y_(s, i) <=0 for all i: (s,i) is an edge

        \- x_i - y_(i, t) <= -1 for all i: (i,t) is an edge

    Parameters
    ----------
    adj: ndarray
    n_internal_nodes: int
    sel: tuple[slice, slice]
    internal_sources: bool
    internal_dests: bool

    Returns
    -------
    ndarray

    """
    bools = np.zeros_like(adj).astype(bool)
    bools[sel] = True
    bools *= (adj != 0)
    nonzero = np.argwhere(bools).T
    count = nonzero[0].shape[0]
    edge_basis = np.zeros((count,) + adj.shape, dtype=int)
    edge_basis[np.arange(count), nonzero[0], nonzero[1]] = -adj[tuple(nonzero)]
    edge_indicators = edge_basis.reshape(edge_basis.shape[0], -1)
    pairings = np.zeros((count, n_internal_nodes), dtype=int)
    if internal_sources:
        pairings[np.arange(count), nonzero[0]] = -1
    if internal_dests:
        pairings[np.arange(count), nonzero[1]] = 1
    constraints = np.concatenate([pairings, edge_indicators], axis=1)
    rhs = np.zeros(count, dtype=int) if internal_dests \
        else -np.ones(count, dtype=int)
    return constraints, rhs


def generate_source_cut_constraints(adj, n_internal_nodes):
    """
    Generate cut constraints of the form

        x_i - y_(s, i) <=0 for all i: (s,i) is an edge
\\
    for edges from the source node.

    Parameters
    ----------
    adj: ndarray
    n_internal_nodes: int

    Returns
    -------
    ndarray

    """
    sel = (slice(-2, None), slice(None))
    return generate_cut_constraints(adj, n_internal_nodes, sel,
                                    internal_sources=False, internal_dests=True)


def generate_sink_cut_constraints(adj, n_internal_nodes):
    """
    Generate cut constraints of the form

        - x_i - y_(i, t) <= -1 for all i: (i,t) is an edge
\\
    for edges that go towards the sink edge.

    Parameters
    ----------
    adj: ndarray
    n_internal_nodes: int

    Returns
    -------
    ndarray

    """
    sel = (slice(None), slice(-1, None))
    return generate_cut_constraints(adj, n_internal_nodes, sel,
                                    internal_sources=True, internal_dests=False)


def extract_adjacencies(distance_cutoff, distance_rolloff,
                        between_part_scalar, chains, ends, events, timefactor,
                        half_internal_nodes, n_internal_nodes, parts,
                        starts):
    """
    Generate unweighted and weighted adjacency structures from score data.

    Parameters
    ----------
    distance_cutoff: int
    distance_rolloff: float in (0,1]
    between_part_scalar: float
    chains: ndarray (1D) of length `n_internal_nodes`
    ends: ndarray (1D) of length `n_internal_nodes`
    events: ndarray (1D) of length `n_internal_nodes`
    timefactor: ndarray (1D) of length `n_internal_nodes`
    half_internal_nodes: int
    n_internal_nodes: int
    parts: ndarray (1D) of length `n_internal_nodes`
    starts: ndarray (1D) of length `n_internal_nodes`

    Returns
    -------
    ndarray, ndarray

    """
    within_chain_adjs = generate_within_part_adj(chains, distance_cutoff,
                                                 half_internal_nodes,
                                                 events,
                                                 n_internal_nodes,
                                                 parts)
    # Connect concurrent notes in different parts
    between_part_adj = generate_between_parts_adj(ends, parts, starts)
    # Add a scale factor according to position in the score - present in
    # the input matrix 'X'
    endweighting = generate_endweighting(timefactor)
    # Generate adjacency matrix
    adj = generate_adj(between_part_adj, half_internal_nodes, n_internal_nodes,
                       within_chain_adjs)
    # Adjacency with relative weights based on proximity in score
    weighted_adj = generate_weighted_adj(between_part_adj, between_part_scalar,
                                         distance_rolloff, adj, endweighting,
                                         within_chain_adjs)
    return adj, weighted_adj


def generate_weighted_adj(between_parts_adj, between_part_scalar,
                          distance_rolloff, adj, endweighting,
                          within_chain_adjs):
    """
    Generate weighted adjacency incorporating scalefactors applied to
    different subcomponents of adjacency structure

    Parameters
    ----------
    between_parts_adj: ndarray
    between_part_scalar: float
    distance_rolloff: float in (0,1]
    adj: ndarray
    endweighting: ndarray
    within_chain_adjs: ndarray

    Returns
    -------
    ndarray
    """
    weighted_adj = adj.astype(float)
    weighted_adj[:-2, :-2] = sum([
        pow(distance_rolloff, i) * chain for i, chain in
        enumerate(within_chain_adjs)
    ]) + between_part_scalar * between_parts_adj
    weighted_adj = endweighting * weighted_adj
    return weighted_adj


def generate_adj(between_part_adj, half_internal_nodes, n_internal_nodes,
                 within_chain_adjs):
    """
    Generate adjacency structures between chains (for simultaneous events) and
    within chains (each precomputed).

    Parameters
    ----------
    between_part_adj: ndarray
    half_internal_nodes: int
    n_internal_nodes: int
    within_chain_adjs: ndarray

    Returns
    -------
    ndarray

    """
    source_adj = np.zeros((n_internal_nodes,), dtype=int)
    sink_adj = np.zeros((n_internal_nodes,), dtype=int)
    idx = np.indices((half_internal_nodes,), dtype=int) * 2
    source_adj[idx] = 1
    sink_adj[idx + 1] = 1
    sink_adj = np.append(sink_adj, 0)
    adj = sum(within_chain_adjs) + between_part_adj
    adj = add_node(adj, out_edges=source_adj)
    adj = add_node(adj, in_edges=sink_adj)
    return adj


def generate_between_parts_adj(ends, parts, starts):
    """
    Generate adjacency structure that connects notes in separate parts that
    are simultaneous.

    Parameters
    ----------
    ends: ndarray
    parts: ndarray
    starts: ndarray

    Returns
    -------
    ndarray

    """
    part_adj = pullback(parts)
    not_part_adj = np.logical_not(part_adj).astype(int)
    between_parts_adj = concurrencies(
        starts, ends
    ) * not_part_adj
    return between_parts_adj


def generate_within_part_adj(chains, distance_cutoff, half_internal_nodes,
                             events, n_internal_nodes, parts):
    """
    Generate connections between notes within the same chain (sub-sections of
    parts between pre-defined "chainbreakers" like long rests or double
    barlines).

    Parameters
    ----------
    chains: ndarray
    distance_cutoff: int
    half_internal_nodes: int
    n_events: int
    n_internal_nodes: int
    parts: ndarray

    Returns
    -------
    list[ndarray]

    """
    n_events = events.max() + 1
    part_adj = pullback(parts)
    chain_adj = pullback(chains) * part_adj
    within_chain_adjs = list(map(
        lambda arr: pullback(events, arr) * chain_adj,
        [hop_adjacencies(i, n_events) for i in range(distance_cutoff + 1)]
    ))
    # Remove adjacency within the same note (between notes in the same
    # event is fine)
    within_chain_adjs[0] *= np.logical_not(f_inverse(
        lambda x: x // 2, (n_internal_nodes, n_internal_nodes), np.eye(
            half_internal_nodes, dtype=int)
    )).astype(int)
    return within_chain_adjs


def generate_endweighting(timefactor):
    """
    Multiply together the timefactor associated to each node for each pair of
    nodes (each edge in a complete graph).

    Parameters
    ----------
    timefactor: ndarray (1D)

    Returns
    -------
    ndarray (2D)

    """
    endweighting = np.hstack(timefactor) * np.vstack(timefactor)
    endweighting = add_node(endweighting, out_edges=timefactor)
    endweighting = add_node(endweighting, in_edges=np.append(timefactor, 0))
    return endweighting
