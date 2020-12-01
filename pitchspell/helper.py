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


def generate_capacities_def(pre_calculated_weights, big_M, edge_weights, n_edges,
                            n_internal_nodes, n_nodes, weighted_adj,
                            pitch_classes):
    capacities_def = np.eye(n_edges, dtype=int)
    if pre_calculated_weights:
        # c_i,j = t(i,j) * w_(p(i), p(j))
        # RHS
        capacities_def_rhs = (
                weighted_adj * edge_weights + big_M
        ).flatten()
    else:
        # c_i,j - t(i,j) * w_(p(i), p(j)) = 0
        pitch_class_indexing = pitch_classes * 2 + np.indices(
            (n_internal_nodes,), dtype=int
        ) % 2
        pc_idx_with_src_sink = np.append(pitch_class_indexing,
                                         [n_pitch_class_internal_nodes,
                                          n_pitch_class_internal_nodes + 1])
        pc_edge_2d_idx = pc_idx_with_src_sink[
            np.indices((n_nodes, n_nodes))]
        pc_edge_1d_idx = (
                pc_edge_2d_idx[0] * n_pitch_class_nodes + pc_edge_2d_idx[
            1]).flatten()
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


def generate_flow_conditions(internal_adj, n_edges,
                             n_internal_nodes, n_nodes, square_idx):
    flow_conditions = np.zeros((n_internal_nodes, n_edges), dtype=int)
    flow_conditions[
        square_idx[0], (n_nodes) * square_idx[0] + square_idx[1]
    ] = internal_adj[tuple(square_idx)]
    flow_conditions[
        square_idx[0], (n_nodes) * square_idx[1] + square_idx[0]
    ] = -internal_adj[tuple(square_idx)]
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
    edge_weights = internal_scheme[
        tuple(
            (pitch_classes * 2 + np.indices((n_internal_nodes,), dtype=int)[
                0] % 2)[
                np.indices((n_internal_nodes, n_internal_nodes), dtype=int)
            ]
        )
    ]
    edge_weights = add_node(edge_weights, out_edges=source_edges)
    edge_weights = add_node(edge_weights, in_edges=sink_edges)
    return edge_weights


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


def generate_internal_cut_constraints(adj, n_edges,
                                      n_internal_edges,
                                      n_internal_nodes, n_nodes):
    internal_adj = adj[-2:, -2:]
    sq_idx = np.indices((n_internal_nodes, n_internal_nodes))
    internal_pairings = np.zeros(
        (n_internal_edges, n_internal_nodes), dtype=int)
    internal_pairings[
        np.arange(n_internal_edges), sq_idx[0].flatten()] -= 1
    internal_pairings[
        np.arange(n_internal_edges), sq_idx[1].flatten()] += 1
    edge_indicators = np.zeros(
        (n_internal_edges, n_edges), dtype=int)
    edge_indicators[
        n_internal_nodes * sq_idx[0] + sq_idx[1],
        (n_nodes) * sq_idx[0] + sq_idx[1]
    ] = internal_adj[tuple(sq_idx)]
    internal_constraints = np.concatenate([internal_pairings,
                                           edge_indicators], axis=1)
    internal_constraints *= internal_adj.reshape(
        (n_internal_edges, 1))
    internal_constraints_rhs = np.zeros(n_internal_edges, dtype=int)
    return internal_constraints, internal_constraints_rhs


def generate_source_cut_constraints(adj, half_internal_nodes,
                                    n_edges, n_internal_nodes,
                                    n_nodes):
    source_adj = adj[-2]
    source_pairings = np.zeros((half_internal_nodes, n_internal_nodes),
                               dtype=int)
    source_pairings[np.arange(half_internal_nodes), np.arange(
        half_internal_nodes) * 2] = 1
    source_edge_indicators = np.zeros(
        (half_internal_nodes, n_edges), dtype=int)
    source_edge_indicators[
        np.arange(half_internal_nodes),
        (n_nodes) * n_internal_nodes + np.arange(
            half_internal_nodes) * 2
    ] = source_adj[np.arange(half_internal_nodes) * 2]
    source_constraints = np.concatenate([source_pairings,
                                         source_edge_indicators], axis=1)
    source_constraints *= source_adj[
        np.arange(half_internal_nodes) * 2
        ].reshape((half_internal_nodes, 1))
    source_constraints_rhs = np.zeros(half_internal_nodes, dtype=int)
    return source_constraints, source_constraints_rhs


def generate_sink_cut_constraints(adj, half_internal_nodes,
                                  n_edges, n_internal_nodes, n_nodes):
    sink_adj = adj[:, -1]
    sink_pairings = np.zeros((half_internal_nodes, n_internal_nodes),
                             dtype=int)
    sink_pairings[np.arange(half_internal_nodes), np.arange(
        half_internal_nodes) * 2 + 1] = -1
    sink_edge_indicators = np.zeros(
        (half_internal_nodes, n_edges), dtype=int)
    sink_edge_indicators[
        np.arange(half_internal_nodes),
        (n_nodes) * (
                np.arange(half_internal_nodes) * 2 + 1) + (
                n_internal_nodes + 1)
    ] = sink_adj[np.arange(half_internal_nodes) * 2 + 1]
    sink_constraints = np.concatenate([sink_pairings,
                                       sink_edge_indicators], axis=1)
    sink_constraints *= sink_adj[
        np.arange(half_internal_nodes) * 2 + 1
        ].reshape((half_internal_nodes, 1))
    sink_constraints_rhs = np.full((half_internal_nodes,), -1)
    return sink_constraints, sink_constraints_rhs


def extract_adjacencies(distance_cutoff, distance_rolloff,
                        between_part_scalar, chains, ends, events,
                        half_internal_nodes, n_internal_nodes, parts,
                        starts):
    n_events = events.max() + 1
    part_adj = pullback(parts)
    chain_adj = pullback(chains) * part_adj
    not_part_adj = -part_adj
    within_chain_adjs = list(map(
        lambda arr: pullback(events) * chain_adj,
        [hop_adjacencies(i, n_events) for i in range(distance_cutoff)]
    ))
    # Remove adjacency within the same note (between notes in the same
    # event is fine)
    within_chain_adjs[0] *= -f_inverse(
        lambda x: x // 2, (n_internal_nodes, n_internal_nodes), np.eye(
            half_internal_nodes, dtype=int)
    )
    # Connect concurrent notes in different parts
    between_part_adj = concurrencies(
        starts, ends
    ) * not_part_adj
    # Add a scale factor according to position in the score - present in
    # the input matrix 'X'
    idx = np.indices((n_internal_nodes, n_internal_nodes), sparse=True)
    timefactor = ends[idx[0]] * ends[idx[1]]
    timefactor = add_node(timefactor, out_edges=ends)
    timefactor = add_node(timefactor, in_edges=ends)
    # Generate adjacency matrix
    source_adj = np.zeros((n_internal_nodes,), dtype=int)
    sink_adj = np.zeros((n_internal_nodes,), dtype=int)
    idx = np.indices((half_internal_nodes,), dtype=int) * 2
    source_adj[idx] = 1
    sink_adj[idx + 1] = 1
    adj = sum(within_chain_adjs) + between_part_adj
    adj = add_node(adj, out_edges=source_adj)
    adj = add_node(adj, in_edges=sink_adj)
    # Adjacency with relative weights based on proximity in score
    weighted_adj = sum([
        pow(distance_rolloff, i) * adj for i, adj in
        enumerate(within_chain_adjs)
    ]) + between_part_scalar * between_part_adj
    weighted_adj = add_node(weighted_adj, out_edges=source_adj)
    weighted_adj = add_node(weighted_adj, in_edges=sink_adj)
    weighted_adj *= timefactor
    return adj, weighted_adj
