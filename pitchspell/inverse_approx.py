from sklearn.base import BaseEstimator
import numpy as np

from pitchspell.minimum_cut import generate_complete_cut
from pitchspell.prepare_edges import hop_adjacencies, concurrencies, add_node
from pitchspell.pullback import pullback, f_inverse, stretch, pad


class ApproximateInverter(BaseEstimator):
    int_cols = ['eventnum', 'chain', 'partnum', 'Pitch Class']
    float_cols = ['Offset', 'Duration', 'timefactor']

    source_edge_scheme = np.array((13, 26, 3, 1, 13, 13, 26, 3, 0, 3, 1, 13))
    sink_edge_scheme = np.array((13, 1, 3, 26, 13, 13, 1, 3, 0, 3, 26, 13))

    down_to_down = np.array([
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    ])

    up_to_up = np.array([
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    ])

    down_to_up = np.array([
        [1, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 1, 2, 0, 3, 0, 2, 0, 0, 2],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 1, 1, 2, 0, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0],
        [2, 0, 0, 2, 0, 2, 0, 2, 1, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 3, 0, 0, 0, 0, 3, 0, 3, 0, 1, 2],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    ])

    up_to_down = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
        [2, 1, 0, 3, 0, 2, 0, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 2, 0, 0, 1],
        [1, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0],
        [0, 0, 0, 3, 0, 2, 1, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0],
        [2, 0, 0, 2, 0, 2, 0, 2, 1, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 1, 1],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1]
    ])

    internal_scheme = np.array([
        [0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        [2, 0, 1, 0, 0, 1, 3, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 3, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 3, 1, 0, 0, 1, 0, 2, 1, 0, 0, 3, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2],
        [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 1, 0, 0, 2, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 3, 0, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 0, 1, 0, 1],
        [0, 2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 0, 1, 1, 0, 0, 3, 1, 0],
        [2, 0, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 0, 0, 1, 3, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 3, 1, 0, 1, 0, 0, 0, 1, 0, 0, 3, 1, 0, 0, 3, 1, 0, 0, 1, 0, 2],
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 3, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0]
    ])

    def __init__(self, accuracy, distance_cutoff, distance_rolloff,
                 same_event_scalar, between_part_scalar):
        self.accuracy = accuracy
        self.distance_cutoff = distance_cutoff
        self.distance_rolloff = distance_rolloff
        self.same_event_scalar = same_event_scalar
        self.between_part_scalar = between_part_scalar

    # TODO: finish implementation
    def fit(self, X, y):
        """
        Constructs matrix inputs for scipy.optimize.linprog from `X` and `y`
        matrix data. Produces edge weight matrix in terms of pitch classes (
        and additional binary index) for use with test data sets (and
        populating linear program matrices accordingly).

        Parameters
        ----------
        X: numpy.ndarray
        y: numpy.ndarray (1D)

        """
        N = len(y)
        n_events = X[:, 0].max() + 1

        part_adj = pullback(X[:, 2])
        chain_adj = pullback(X[:, 1]) * part_adj
        not_part_adj = ~part_adj

        within_chain_adjs = list(map(
            lambda arr: pullback(X[:, 0]) * chain_adj,
            [hop_adjacencies(i, n_events) for i in range(self.distance_cutoff)]
        ))
        # Remove adjacency within the same note (between notes in the same
        # event is fine)
        within_chain_adjs[0] *= ~f_inverse(
            lambda x: x // 2, (N, N), np.eye(
                N // 2, dtype='int')
        )

        between_part_adj = concurrencies(
            X[:, 4], X[:, 5]
        ) * not_part_adj

        internal_adj = np.tile([[0, 0], [1, 0]], [N // 2, N // 2]) * np.repeat(
            np.repeat(
                np.eye(N // 2, N // 2, dtype='int'),
                [2],
                axis=1),
            [2],
            axis=0)
        big_M = np.inf * internal_adj
        big_M[np.isnan(big_M)] = 0
        big_M = add_node(big_M)
        big_M = add_node(big_M)

        idx = np.indices((N, N), sparse=True)
        timefactor = X[:, -1][idx[0]] * X[:, -1][idx[1]]
        timefactor = add_node(timefactor, out_edges=X[:, -1])
        timefactor = add_node(timefactor, in_edges=X[:, -1])

        idx = np.indices((N // 2,), dtype='int') * 2
        source_edges = self.source_edge_scheme[X[:, 3]]
        source_edges[idx + 1] = 0
        sink_edges = self.sink_edge_scheme[X[:, 3]]
        sink_edges[idx] = 0

        edge_weights = self.internal_scheme[
            (X[:, 3] * 2 + np.indices((N,), dtype='int') % 2)[
                np.indices((N, N), dtype='int')
            ]
        ]
        edge_weights = add_node(edge_weights, out_edges=source_edges)
        edge_weights = add_node(edge_weights, in_edges=sink_edges)

        full_adj = sum(within_chain_adjs) + between_part_adj + internal_adj
        full_adj = add_node(full_adj, out_edges=np.clip(source_edges, 0, 1))
        full_adj = add_node(full_adj, in_edges=np.clip(sink_edges, 0, 1))
        cut = generate_complete_cut(y) * full_adj

        # sum_(i=1)^n f_(i,k) - sum_(j=1)^n f_(k,j) = 0 for all k != s,t
        flow_conditions = full_adj - full_adj.T
        # Remove flow conditions from source and sink edges
        flow_conditions[-2:] = 0
        flow_conditions[:, -2:] = 0
        # add space for delta variable and pitch class weight matrix
        flow_conditions_spaced = pad(
            (N, 2 * pow(N + 2, 2) + 1 + pow(26, 2)),
            flow_conditions,
            np.indices(flow_conditions.shape)
        )

        # f_(i,j) <= c_(i,j)
        capacity_conditions = np.concatenate(
            [
                np.eye(pow(N + 2, 2), dtype='int'),
                ~np.eye(pow(N + 2, 2), dtype='int')
            ],
            axis=1
        )
        # add space for delta variable and pitch class weight matrix
        capacity_conditions_spaced = pad(
            capacity_conditions.shape[0],
            (N, 2 * pow(N + 2, 2) + 1 + pow(26, 2)),
            np.indices(capacity_conditions.shape)
        )

        # sum_(i=1)^n f_s,i - sum_(e in cut) c_e = delta
        duality_constraint = np.concatenate([
            np.ones(N, dtype='int'), ~cut.flatten(), [1]
        ])
        duality_constraint_spaced = pad(
            (2 * pow(N + 2, 2) + 1 + pow(26, 2),),
            duality_constraint,
            np.concatenate([
                np.indices((N + 2,)) + N * (N + 2),
                np.indices((pow(N + 2, 2),)),
                [-1]
            ])
        )
        scaled_adj = sum([
            pow(self.distance_rolloff, i) * adj for i, adj in
            enumerate(within_chain_adjs)
        ]) + self.between_part_scalar * between_part_adj
        # add edges for source and sink edges
        scaled_adj = add_node(scaled_adj, out_edges=np.clip(source_edges, 0, 1))
        scaled_adj = add_node(scaled_adj, in_edges=np.clip(sink_edges, 0, 1))

        # condition for if pitch class weight matrix is not given
        pitch_class_indexing = X[:, 3] * 2 + np.indices((N,), dtype='int') % 2
        pc_idx_with_src_sink = np.append(pitch_class_indexing, [24, 25])
        pc_edge_2d_idx = pc_idx_with_src_sink[np.indices((N + 2, N + 2))]
        pc_edge_1d_idx = (pc_edge_2d_idx[0] * 26 + pc_edge_2d_idx[1]).flatten()
        pitch_based_capacity = np.zeros(pow(N + 2, 2), pow(26, 2))
        pitch_based_capacity[
            np.indices((pow(N + 2, 2),)),
            pc_edge_1d_idx
        ] = 1
        capacity_condition_lhs = pitch_based_capacity * scaled_adj.flatten()
        column_shift = np.indices(capacity_condition_lhs.shape)
        column_shift[1] += 2 * pow(N + 2, 2) + 1
        capacity_condition_lhs_spaced = pad(
            (
                capacity_condition_lhs.shape[0],
                2 * pow(N + 2, 2) + 1 + pow(26, 2)
            ),
            capacity_condition_lhs,
            column_shift
        )

        # capacities definition (RHS)
        capacities_values = (
                timefactor * (scaled_adj * edge_weights + big_M)
        ).flatten()
        # capacities definition (LHS)
        capacities_def = np.eye(pow(N + 2, 2), dtype='int')

        abs_idx = np.indices((pow(N + 2, 2), pow(N + 2, 2)))
        abs_idx[1] += pow(N + 2, 2)

        capacities_def_spaced = pad(
            (pow(N + 2, 2), 2 * pow(N + 2, 2) + 1 + pow(24, 2)),
            capacities_def,
            abs_idx
        )
