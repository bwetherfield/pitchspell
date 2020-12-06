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
    c = np.zeros((n_variables), dtype=int)
    c[2 * n_edges] = accuracy
    if not pre_calculated_weights:
        c[-n_pitch_class_edges:] = -1
    return c


def generate_capacities_def(pre_calculated_weights, big_M, weight_scalers,
                            n_edges,
                            n_internal_nodes, n_nodes, weighted_adj,
                            pitch_classes):
    capacities_def = np.eye(n_edges, dtype=int)
    if pre_calculated_weights:
        # c_i,j = a(i,j) * w_(p(i), p(j))
        # RHS
        capacities_def_rhs = (weighted_adj * weight_scalers + big_M).flatten()
    else:
        # c_i,j - a(i,j) * w_(p(i), p(j)) = 0
        pc_idx = pitch_classes * 2 + np.arange(n_internal_nodes) % 2
        pc_idx_with_src_sink = np.append(pc_idx,
                                         [n_pitch_class_internal_nodes,
                                          n_pitch_class_internal_nodes + 1])
        pc_edge_2d_idx = pc_idx_with_src_sink[np.indices((n_nodes, n_nodes))]
        pc_edge_1d_idx = (
                pc_edge_2d_idx[0] * n_pitch_class_nodes
                + pc_edge_2d_idx[1]
        ).flatten()

        pitch_based_capacity = np.zeros((n_edges, n_pitch_class_edges),
                                        dtype=int)
        pitch_based_capacity[
            np.indices((n_edges,)),
            pc_edge_1d_idx
        ] = weighted_adj.flatten()
        capacities_def = np.concatenate([
            capacities_def, -pitch_based_capacity
        ], axis=1)
        # RHS
        capacities_def_rhs = big_M.flatten()
    return capacities_def, capacities_def_rhs


def generate_flow_conditions(adj, n_internal_nodes, n_nodes):
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
    return np.logical_not(
        f_inverse(lambda x: x // 2, (n, n), np.eye(n // 2, dtype=int))
    ).astype(int)


def generate_duality_constraint(cut, n_internal_nodes):
    duality_constraint = np.concatenate([
        -np.ones(n_internal_nodes, dtype=int), cut.flatten(), [-1]
    ])
    # RHS
    duality_constraint_rhs = [0]
    return duality_constraint, duality_constraint_rhs


def get_big_M_edges(half_internal_nodes):
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


def generate_cut(adj, y):
    y_plus_source_sink = np.concatenate([y, [0, 1]])
    cut = generate_complete_cut(y_plus_source_sink) * adj
    return cut


def generate_internal_cut_constraints(adj, n_internal_nodes):
    sel = (slice(None, -2), slice(None, -2))
    return generate_cut_constraints(adj, n_internal_nodes, sel, True, True)


def generate_cut_constraints(adj, n_internal_nodes, sel, internal_sources,
                             internal_dests):
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
    sel = (slice(-2, None), slice(None))
    return generate_cut_constraints(adj, n_internal_nodes, sel,
                                    internal_sources=False, internal_dests=True)


def generate_sink_cut_constraints(adj, n_internal_nodes):
    sel = (slice(None), slice(-1, None))
    return generate_cut_constraints(adj, n_internal_nodes, sel,
                                    internal_sources=True, internal_dests=False)


def extract_adjacencies(distance_cutoff, distance_rolloff,
                        between_part_scalar, chains, ends, events, timefactor,
                        half_internal_nodes, n_internal_nodes, parts,
                        starts):
    big_M_adj, big_M = get_big_M_edges(half_internal_nodes)
    within_chain_adjs = generate_within_part_adj(chains, distance_cutoff,
                                                 half_internal_nodes,
                                                 events,
                                                 n_internal_nodes,
                                                 parts)
    # Connect concurrent notes in different parts
    between_part_adj = generate_between_parts_adj(ends, parts, starts)
    # Add a scale factor according to position in the score - present in
    # the input matrix 'X'
    endweighting = generate_endweighting(timefactor, n_internal_nodes)
    # Generate adjacency matrix
    adj = generate_adj(between_part_adj, half_internal_nodes, n_internal_nodes,
                       within_chain_adjs)
    # Adjacency with relative weights based on proximity in score
    weighted_adj = generate_weighted_adj(between_part_adj, between_part_scalar,
                                         distance_rolloff, adj, endweighting,
                                         within_chain_adjs)
    return big_M_adj, big_M, adj, weighted_adj


def generate_weighted_adj(between_part_adj, between_part_scalar,
                          distance_rolloff, adj, endweighting,
                          within_chain_adjs):
    weighted_adj = adj
    weighted_adj[:-2, :-2] = sum([
        pow(distance_rolloff, i) * chain for i, chain in
        enumerate(within_chain_adjs)
    ]) + between_part_scalar * between_part_adj
    weighted_adj *= endweighting
    return weighted_adj


def generate_adj(between_part_adj, half_internal_nodes, n_internal_nodes,
                 within_chain_adjs):
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
    part_adj = pullback(parts)
    not_part_adj = np.logical_not(part_adj).astype(int)
    between_parts_adj = concurrencies(
        starts, ends
    ) * not_part_adj
    return between_parts_adj


def generate_within_part_adj(chains, distance_cutoff, half_internal_nodes,
                             events, n_internal_nodes, parts):
    """

    Parameters
    ----------
    chains
    distance_cutoff
    half_internal_nodes
    n_events
    n_internal_nodes
    parts

    Returns
    -------
    list of arrays

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


def generate_endweighting(timefactor, n_internal_nodes):
    idx = np.indices((n_internal_nodes, n_internal_nodes), sparse=True)
    endweighting = timefactor[idx[0]] * timefactor[idx[1]]
    endweighting = add_node(endweighting, out_edges=timefactor)
    endweighting = add_node(endweighting, in_edges=np.append(timefactor, 0))
    return endweighting
