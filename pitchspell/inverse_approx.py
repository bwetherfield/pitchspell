from sklearn.base import BaseEstimator
import numpy as np
from scipy.optimize import linprog

from pitchspell.minimum_cut import generate_complete_cut
from pitchspell.prepare_edges import hop_adjacencies, concurrencies, add_node
from pitchspell.pullback import pullback, f_inverse, stretch, pad


class ApproximateInverter(BaseEstimator):
    """
        Learns edge weight scale factors from the pitches and spellings of a
        score translated to a data frame. Provides a
    """
    # int_cols = ['eventnum', 'chain', 'partnum', 'Pitch Class']
    # float_cols = ['Offset', 'Duration', 'timefactor']

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
                 same_event_scalar, between_part_scalar,
                 pre_calculated_weights=True):
        self.accuracy = accuracy
        self.distance_cutoff = distance_cutoff
        self.distance_rolloff = distance_rolloff
        self.same_event_scalar = same_event_scalar
        self.between_part_scalar = between_part_scalar
        self.pre_calculated_weights = pre_calculated_weights

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

        # Construct 'big M' edges between second and first subindex of each
        # note in the score.
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

        # Add a scale factor according to position in the score - present in
        # the input matrix 'X'
        idx = np.indices((N, N), sparse=True)
        timefactor = X[:, -1][idx[0]] * X[:, -1][idx[1]]
        timefactor = add_node(timefactor, out_edges=X[:, -1])
        timefactor = add_node(timefactor, in_edges=X[:, -1])

        # ----------------------------------------
        # AUXILIARY MATRICES
        # Generate full adjacency matrix, cut edges
        source_adj = np.zeros((N,), dtype='int')
        sink_adj = np.zeros((N,), dtype='int')
        idx = np.indices((N // 2,), dtype='int') * 2
        source_adj[idx] = 1
        sink_adj[idx + 1] = 1
        full_adj = sum(within_chain_adjs) + between_part_adj + internal_adj
        full_adj = add_node(full_adj, out_edges=source_adj)
        full_adj = add_node(full_adj, in_edges=sink_adj)
        y_plus_source_sink = np.concatenate([y, [0, 1]])
        cut = generate_complete_cut(y_plus_source_sink) * full_adj

        # Adjacency with relative weights based on proximity in score
        weighted_adj = sum([
            pow(self.distance_rolloff, i) * adj for i, adj in
            enumerate(within_chain_adjs)
        ]) + self.between_part_scalar * between_part_adj
        # add edges for source and sink edges
        weighted_adj = add_node(weighted_adj, out_edges=source_adj)
        weighted_adj = add_node(weighted_adj, in_edges=sink_adj)
        combined_adj = weighted_adj * timefactor

        # ----------------------------------------
        # EQUALITY CONSTRAINTS
        # sum_(i=1)^n f_(i,k) - sum_(j=1)^n f_(k,j) = 0 for all k != s,t
        flow_conditions = full_adj - full_adj.T
        # Remove flow conditions from source and sink edges
        flow_conditions[-2:] = 0
        flow_conditions[:, -2:] = 0

        # sum_(i=1)^n f_s,i - sum_(e in cut) c_e = delta
        duality_constraint = np.concatenate([
            np.ones(N, dtype='int'), ~cut.flatten(), [1]
        ])

        if self.pre_calculated_weights:
            # c_i,j = t(i,j) * w_(p(i), p(j))
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
            capacities_def_with_weights_given = np.eye(pow(N + 2, 2),
                                                       dtype='int')
        else:
            # c_i,j - t(i,j) * w_(p(i), p(j)) = 0
            pitch_class_indexing = X[:, 3] * 2 + np.indices((N,),
                                                            dtype='int') % 2
            pc_idx_with_src_sink = np.append(pitch_class_indexing, [24, 25])
            pc_edge_2d_idx = pc_idx_with_src_sink[np.indices((N + 2, N + 2))]
            pc_edge_1d_idx = (
                    pc_edge_2d_idx[0] * 26 + pc_edge_2d_idx[1]).flatten()
            pitch_based_capacity = np.zeros(pow(N + 2, 2), pow(26, 2))
            pitch_based_capacity[
                np.indices((pow(N + 2, 2),)),
                pc_edge_1d_idx
            ] = combined_adj.flatten()
            capacity_def_with_weights_variable = np.concatenate([
                np.eye(pow(N + 2, 2)), ~pitch_based_capacity
            ], axis=1)

        # ----------------------------------------
        # INEQUALITY CONSTRAINTS
        # f_(i,j) <= c_(i,j)
        capacity_conditions = np.concatenate(
            [
                np.eye(pow(N + 2, 2), dtype='int'),
                ~np.eye(pow(N + 2, 2), dtype='int')
            ],
            axis=1
        )

        # ----------------------------------------
        # SPACED EQUALITY CONSTRAINTS
        # add space for delta variable and pitch class weight matrix
        flow_conditions_spaced = pad(
            (N, 2 * pow(N + 2, 2) + 1 + pow(26, 2)),
            flow_conditions,
            np.indices(flow_conditions.shape)
        )
        flow_conditions_rhs = np.zeros((N), dtype=int)

        duality_constraint_spaced = pad(
            (2 * pow(N + 2, 2) + 1 + pow(26, 2),),
            duality_constraint,
            np.concatenate([
                np.indices((N + 2,)) + N * (N + 2),
                np.indices((pow(N + 2, 2),)),
                [-1]
            ])
        )
        duality_constraint_rhs = [0]

        if self.pre_calculated_weights:
            abs_idx = np.indices((pow(N + 2, 2), pow(N + 2, 2)))
            abs_idx[1] += pow(N + 2, 2)
            capacities_def_spaced = pad(
                (pow(N + 2, 2), 2 * pow(N + 2, 2) + 1 + pow(24, 2)),
                capacities_def_with_weights_given,
                abs_idx
            )
            capacities_def_rhs = (
                    combined_adj * edge_weights + big_M
            ).flatten()
        else:
            pitch_based_capacity_abs_idx = np.indices(
                capacity_def_with_weights_variable.shape
            )
            pitch_based_capacity_abs_idx[1] += 2 * pow(N + 2, 2) + 1
            capacity_idx = np.indices((pow(N + 2, 2), pow(N + 2, 2)))
            capacity_idx[1] += pow(N + 2, 2)
            capacity_definition_spaced_idx = np.concatenate([
                capacity_idx,
                pitch_based_capacity_abs_idx
            ], axis=2)
            capacities_def_spaced = pad(
                (
                    capacity_def_with_weights_variable.shape[0],
                    2 * pow(N + 2, 2) + 1 + pow(26, 2)
                ),
                capacity_def_with_weights_variable,
                capacity_definition_spaced_idx
            )
            capacities_def_rhs = np.zeros((pow(N + 2, 2)), dtype=int)

        # ----------------------------------------
        # SPACED INEQUALITY CONSTRAINTS
        # add space for delta variable and pitch class weight matrix
        capacity_conditions_spaced = pad(
            (N + 2, 2 * pow(N + 2, 2) + 1 + pow(26, 2)),
            capacity_conditions,
            np.indices(capacity_conditions.shape)
        )
        capacity_conditions_rhs = np.zeros((pow(N + 2, 2)), dtype=int)

        # ----------------------------------------
        # SET UP LINEAR PROGRAM
        if self.pre_calculated_weights:
            c = np.zeros((2 * pow(N + 2, 2) + 1))
            c[2 * pow(N + 2, 2)] = self.accuracy
        else:
            c = np.zeros((2 * pow(N + 2, 2) + 1 + pow(26, 2)))
            c[2 * pow(N + 2, 2)] = self.accuracy
            c[-pow(26, 2):] = -1
        A_eq = np.concatenate([
            flow_conditions_spaced,
            duality_constraint_spaced,
            capacities_def_spaced
        ], axis=0)
        b_eq = np.concatenate([
            flow_conditions_rhs,
            duality_constraint_rhs,
            capacities_def_rhs
        ])
        A_ub = capacity_conditions_spaced
        b_ub = capacity_conditions_rhs
        if self.pre_calculated_weights:
            bounds = (0, 100)
        else:
            ub = np.full((2 * pow(N + 2, 2) + 1 + pow(26, 2)), 100)
            ub[-pow(26, 2):] = edge_weights
            pc_scheme = self.internal_scheme
            pc_scheme_idx = np.indices(12) * 2
            pc_source_edges = np.zeros(24)
            pc_source_edges[pc_scheme_idx] = self.source_edge_scheme
            pc_sink_edges = np.zeros(25)
            pc_sink_edges[pc_scheme_idx + 1] = self.sink_edge_scheme
            pc_scheme = add_node(pc_scheme, out_edges=pc_source_edges)
            pc_scheme = add_node(pc_scheme, in_edges=pc_sink_edges)
            ub[-pow(26, 2):] = 26 * np.clip(pc_scheme, 0, 1)
            bounds = (0, ub)

        output_ = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds)
        # Accuracy score given by size of duality gap of linear program
        self._score = output_.x[2 * pow(N + 2, 2)]
        if not self.pre_calculated_weights:
            weights_unfiltered = output_.x[-pow(26, 2):]
            self.source_edge_scheme = weights_unfiltered[
                (np.indices((12,), dtype=int) * 2) * 26 + 24
                ]
            self.sink_edge_scheme = weights_unfiltered[
                (np.indices((12,), dtype=int) * 2 + 1) + 25 * 26
                ]
            self.internal_scheme = weights_unfiltered[
                np.indices((pow(26,2),)).reshape((26,26))[
                    tuple(np.indices(24, 24))
                ].flatten()
            ]

    def score(self, X, y=None):
        return self._score