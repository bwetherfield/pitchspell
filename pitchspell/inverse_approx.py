from sklearn.base import BaseEstimator
import numpy as np
from scipy.optimize import linprog

from pitchspell.minimum_cut import generate_complete_cut
from pitchspell.prepare_edges import hop_adjacencies, concurrencies, add_node
from pitchspell.pullback import pullback, f_inverse, pad

n_pitch_classes = 12
n_pitch_class_internal_nodes = 24  # n_pitch_classes * 2
n_pitch_class_nodes = 26  # n_pitch_class_internal_nodes + 2
n_pitch_class_edges = 676  # pow(pitch_class_nodes, 2)


class ApproximateInverter(BaseEstimator):
    """
        Learns edge weight scale factors from the pitches and spellings of a
        score translated to a data frame. Provides a
        `sklearn.base.BaseEstimator` interface for use with parameter searches.
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
        n_internal_nodes = X.shape[0]
        half_internal_nodes = n_internal_nodes // 2
        n_nodes = n_internal_nodes + 2
        n_edges = pow(n_nodes, 2)
        n_variables = 2 * n_edges + 1 if self.pre_calculated_weights else \
            2 * n_edges + 1 + n_pitch_class_edges

        adj, weighted_adj = self.extract_adjacencies(
            half_internal_nodes, n_internal_nodes, X)
        internal_adj = np.zeros((n_nodes, n_nodes))
        internal_adj[-2:, -2:] = adj[-2:, -2:]

        # ----------------------------------------
        # EQUALITY CONSTRAINTS
        # sum_(i=1)^n f_(i,k) - sum_(j=1)^n f_(k,j) = 0 for all k != s,t
        square_idx = np.indices((n_internal_nodes, n_internal_nodes))
        flow_conditions, flow_conditions_rhs = self.generate_flow_conditions(
            internal_adj, n_edges, n_internal_nodes, n_nodes, square_idx)

        # sum_(i=1)^n f_s,i - sum_(e in cut) c_e = delta
        cut = self.generate_cut(adj, y)
        duality_constraint, duality_constraint_rhs = \
            self.generate_duality_constraint(
                cut, n_internal_nodes)

        # Capacity variable definitions
        big_M_adj, big_M = self.get_big_M_edges(half_internal_nodes)
        edge_weights = self.get_weight_scalers(
            half_internal_nodes, n_internal_nodes, X)
        capacities_def, capacities_def_rhs = \
            self.generate_capacities_def(
                X,
                big_M,
                edge_weights,
                n_edges,
                n_internal_nodes,
                n_nodes,
                weighted_adj
            )

        # ----------------------------------------
        # INEQUALITY CONSTRAINTS
        # f_(i,j) <= c_(i,j)
        capacity_conditions = np.concatenate(
            [
                np.eye(n_edges, dtype='int'),
                -np.eye(n_edges, dtype='int')
            ],
            axis=1
        )

        # ----------------------------------------
        # SPACED EQUALITY CONSTRAINTS
        # add space for delta variable and pitch class weight matrix
        flow_conditions_spaced = pad(
            (flow_conditions.shape[0], n_variables),
            flow_conditions,
            np.indices(flow_conditions.shape)
        )

        duality_constraint_spaced = pad(
            (duality_constraint.shape[0], n_variables),
            duality_constraint,
            np.indices(duality_constraint.shape)
        )

        capacities_def_spaced = self.space_capacities_def(
            capacities_def,
            n_edges,
            n_variables
        )

        # ----------------------------------------
        # SPACED INEQUALITY CONSTRAINTS
        # add space for delta variable and pitch class weight matrix
        capacity_conditions_spaced = pad(
            (capacity_conditions.shape[0], n_variables),
            capacity_conditions,
            np.indices(capacity_conditions.shape)
        )
        capacity_conditions_rhs = np.zeros((n_edges), dtype=int)

        # ----------------------------------------
        # SET UP LINEAR PROGRAM
        c = self.generate_cost_func(n_edges, n_variables)
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
        bounds = self.generate_bounds(edge_weights, n_variables)

        output_ = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds)
        # Accuracy score given by size of duality gap of linear program
        self._score = output_.x[2 * n_edges]

        # Record edge weight schemes
        if not self.pre_calculated_weights:
            weights_unfiltered = output_.x[-n_pitch_class_edges:]
            self.source_edge_scheme = weights_unfiltered[
                (np.indices((n_pitch_classes,),
                            dtype=int) * 2) * n_pitch_class_nodes +
                n_pitch_class_internal_nodes
                ]
            self.sink_edge_scheme = weights_unfiltered[
                (np.indices((n_pitch_classes,), dtype=int) * 2 + 1) + (
                        n_pitch_class_internal_nodes + 1) * n_pitch_class_nodes
                ]
            self.internal_scheme = weights_unfiltered[
                np.indices((n_pitch_class_edges,)).reshape(
                    (n_pitch_class_nodes, n_pitch_class_nodes))[
                    tuple(np.indices(n_pitch_class_internal_nodes,
                                     n_pitch_class_internal_nodes))
                ].flatten()
            ]

    def generate_cost_func(self, n_edges, n_variables):
        c = np.zeros((n_variables), dtype=int)
        c[2 * n_edges] = self.accuracy
        if not self.pre_calculated_weights:
            c[:-n_pitch_class_edges] = -1
        return c

    def generate_bounds(self, edge_weights, n_variables):
        if self.pre_calculated_weights:
            bounds = (0, None)
        else:
            ub = np.full((n_variables), None)
            ub[-n_pitch_class_edges:] = edge_weights
            pc_scheme = self.internal_scheme
            pc_scheme_idx = np.indices(n_pitch_classes) * 2
            pc_source_edges = np.zeros(n_pitch_class_internal_nodes, dtype=int)
            pc_source_edges[pc_scheme_idx] = self.source_edge_scheme
            pc_sink_edges = np.zeros(n_pitch_class_internal_nodes + 1, dtype=int)
            pc_sink_edges[pc_scheme_idx + 1] = self.sink_edge_scheme
            pc_scheme = add_node(pc_scheme, out_edges=pc_source_edges)
            pc_scheme = add_node(pc_scheme, in_edges=pc_sink_edges)
            ub[-n_pitch_class_edges:] = n_pitch_class_nodes * np.clip(pc_scheme,
                                                                      0, 1)
            bounds = list(zip(np.zeros_like(ub, dtype=int), ub))
        return bounds

    def space_capacities_def(self, capacities_def, n_edges, n_variables):
        if self.pre_calculated_weights:
            abs_idx = np.indices(
                (n_edges, n_edges))
            abs_idx[1] += n_edges
            capacities_def_spaced = pad(
                (capacities_def.shape[0],
                 n_variables),
                capacities_def,
                abs_idx
            )
        else:
            pitch_based_capacity_abs_idx = np.indices(
                capacities_def.shape
            )
            pitch_based_capacity_abs_idx[1] += n_variables
            capacity_idx = np.indices(
                (n_edges, n_edges))
            capacity_idx[1] += n_edges
            capacity_definition_spaced_idx = np.concatenate([
                capacity_idx,
                pitch_based_capacity_abs_idx
            ], axis=2)
            capacities_def_spaced = pad(
                (
                    capacities_def.shape[0],
                    n_variables
                ),
                capacities_def,
                capacity_definition_spaced_idx
            )
        return capacities_def_spaced

    def generate_capacities_def(self, X, big_M, edge_weights, n_edges,
                                n_internal_nodes, n_nodes, weighted_adj):
        capacities_def = np.eye(n_edges, dtype=int)
        if self.pre_calculated_weights:
            # c_i,j = t(i,j) * w_(p(i), p(j))
            # RHS
            capacities_def_rhs = (
                    weighted_adj * edge_weights + big_M
            ).flatten()
        else:
            # c_i,j - t(i,j) * w_(p(i), p(j)) = 0
            pitch_class_indexing = X[:, 3] * 2 + np.indices((n_internal_nodes,),
                                                            dtype='int') % 2
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

    def generate_duality_constraint(self, cut, n_internal_nodes):
        duality_constraint = np.concatenate([
            np.ones(n_internal_nodes, dtype='int'), -cut.flatten(), [-1]
        ])
        # RHS
        duality_constraint_rhs = [0]
        return duality_constraint, duality_constraint_rhs

    def generate_flow_conditions(self, internal_adj, n_edges, n_internal_nodes,
                                 n_nodes, square_idx):
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

    def get_weight_scalers(self, half_internal_nodes, n_internal_nodes, X):
        """
        Get weight scalers based on the pitches of the notes in the dataset X.

        Parameters
        ----------
        n_internal_nodes: int
        X: numpy.ndarray

        Returns
        -------
        numpy.ndarray

        """
        idx = np.indices((half_internal_nodes,), dtype='int') * 2
        source_edges = self.source_edge_scheme[X[:, 3]]
        source_edges[idx + 1] = 0
        sink_edges = self.sink_edge_scheme[X[:, 3]]
        sink_edges[idx] = 0
        edge_weights = self.internal_scheme[
            tuple(
                (X[:, 3] * 2 + np.indices((n_internal_nodes,), dtype='int')[
                    0] % 2)[
                    np.indices((n_internal_nodes, n_internal_nodes), dtype='int')
                ]
            )
        ]
        edge_weights = add_node(edge_weights, out_edges=source_edges)
        edge_weights = add_node(edge_weights, in_edges=sink_edges)

        return edge_weights

    def get_big_M_edges(self, half_internal_nodes):
        """
        Output the "big M" component graph which connects the up node and the
        down node corresponding to a single note in the musical score by a
        directed infinite weight edge.

        Parameters
        ----------
        n_internal_nodes: int

        Returns
        -------
        numpy.ndarray, numpy.ndarray

        """

        adj_within = np.tile(
            [[0, 0], [1, 0]],
            [half_internal_nodes, half_internal_nodes]
        ) * np.repeat(
            np.repeat(
                np.eye(half_internal_nodes, dtype='int'),
                2,
                axis=1),
            2,
            axis=0)
        big_M = adj_within
        big_M[big_M == 1] = np.inf
        big_M = add_node(big_M)
        big_M = add_node(big_M)

        return adj_within, big_M

    def extract_adjacencies(self, half_internal_nodes, n_internal_nodes, X):
        """
        Generate unweighted and weighted adjacency structures from node data X.

        Parameters
        ----------
        n_internal_nodes: int
        X: numpy.ndarray

        Returns
        -------
        numpy.ndarray, numpy.ndarray

        """
        half_internal_nodes = n_internal_nodes // 2
        n_events = X[:, 0].max() + 1
        part_adj = pullback(X[:, 2])
        chain_adj = pullback(X[:, 1]) * part_adj
        not_part_adj = -part_adj
        within_chain_adjs = list(map(
            lambda arr: pullback(X[:, 0]) * chain_adj,
            [hop_adjacencies(i, n_events) for i in range(self.distance_cutoff)]
        ))

        # Remove adjacency within the same note (between notes in the same
        # event is fine)
        within_chain_adjs[0] *= -f_inverse(
            lambda x: x // 2, (n_internal_nodes, n_internal_nodes), np.eye(
                half_internal_nodes, dtype='int')
        )

        # Connect concurrent notes in different parts
        between_part_adj = concurrencies(
            X[:, 4], X[:, 5]
        ) * not_part_adj

        # Add a scale factor according to position in the score - present in
        # the input matrix 'X'
        idx = np.indices((n_internal_nodes, n_internal_nodes), sparse=True)
        timefactor = X[:, -1][idx[0]] * X[:, -1][idx[1]]
        timefactor = add_node(timefactor, out_edges=X[:, -1])
        timefactor = add_node(timefactor, in_edges=X[:, -1])

        # Generate adjacency matrix
        source_adj = np.zeros((n_internal_nodes,), dtype='int')
        sink_adj = np.zeros((n_internal_nodes,), dtype='int')
        idx = np.indices((half_internal_nodes,), dtype='int') * 2
        source_adj[idx] = 1
        sink_adj[idx + 1] = 1
        adj = sum(within_chain_adjs) + between_part_adj
        adj = add_node(adj, out_edges=source_adj)
        adj = add_node(adj, in_edges=sink_adj)

        # Adjacency with relative weights based on proximity in score
        weighted_adj = sum([
            pow(self.distance_rolloff, i) * adj for i, adj in
            enumerate(within_chain_adjs)
        ]) + self.between_part_scalar * between_part_adj
        weighted_adj = add_node(weighted_adj, out_edges=source_adj)
        weighted_adj = add_node(weighted_adj, in_edges=sink_adj)
        weighted_adj *= timefactor
        return adj, weighted_adj

    def generate_cut(self, adj, y):
        y_plus_source_sink = np.concatenate([y, [0, 1]])
        cut = generate_complete_cut(y_plus_source_sink) * adj
        return cut

    def score(self, X, y=None):
        return self._score

    def predict(self, X):
        """
        Forward solver for `self`.

        Parameters
        ----------
        X: numpy.ndarray

        Returns
        -------
        numpy.ndarray

        Variable order: x_i (N), y_ij ((N+2)^2) with final two columns of
        implicit square of variables s, and t
        """
        n_internal_nodes = X.shape[0]
        n_internal_edges = pow(n_internal_nodes, 2)
        n_nodes = n_internal_nodes + 2
        n_edges = pow(n_nodes, 2)
        half_internal_nodes = n_internal_nodes // 2
        adj, weighted_adj = self.extract_adjacencies(
            half_internal_nodes, n_internal_nodes, X
        )
        edge_weights = self.get_weight_scalers(
            half_internal_nodes, n_internal_nodes, X)
        big_M_adj, big_M = self.get_big_M_edges(half_internal_nodes)

        # x_j - x_i - y_(i,j) <= 0 for all i, j != s, t, where (i,j) is an edge
        internal_constraints, internal_constraints_rhs = \
            self.generate_internal_cut_constraints(
            adj, n_edges, n_internal_edges, n_internal_nodes, n_nodes)

        # x_i - y_(s, i) <=0 for all i: (s,i) is an edge
        source_constraints, source_constraints_rhs = \
            self.generate_source_cut_constraints(
            adj, half_internal_nodes, n_edges, n_internal_nodes, n_nodes)

        # - x_i - y_(i, t) <= -1 for all i: (i,t) is an edge
        sink_constraints, sink_constraints_rhs = \
            self.generate_sink_cut_constraints(
            adj, half_internal_nodes, n_edges, n_internal_nodes, n_nodes)

        # x_i <= 1 bounds
        # x_i, y_ij >= 0 bounds (default)
        bounds = [(0, 1)] * n_internal_nodes + [(0, n_internal_nodes)] * n_edges

        capacities = (weighted_adj * edge_weights + big_M).flatten()
        c = np.append([np.zeros(n_internal_nodes, dtype=int), capacities])
        A_ub = np.concatenate([
            internal_constraints,
            source_constraints,
            sink_constraints
        ], axis=0)
        b_ub = np.append([
            internal_constraints_rhs,
            source_constraints_rhs,
            sink_constraints_rhs
        ])
        return linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds).x

    def generate_sink_cut_constraints(self, adj, half_internal_nodes, n_edges,
                                      n_internal_nodes, n_nodes):
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

    def generate_source_cut_constraints(self, adj, half_internal_nodes, n_edges,
                                        n_internal_nodes, n_nodes):
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

    def generate_internal_cut_constraints(self, adj, n_edges, n_internal_edges,
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
