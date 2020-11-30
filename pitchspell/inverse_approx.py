from sklearn.base import BaseEstimator
import numpy as np
from scipy.optimize import linprog

from pitchspell.helper import generate_bounds, generate_cost_func, \
    generate_capacities_def, generate_flow_conditions, \
    get_weight_scalers, generate_duality_constraint, \
    get_big_M_edges, generate_cut, \
    generate_internal_cut_constraints, \
    generate_source_cut_constraints, generate_sink_cut_constraints
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

    def generate_cost_func(self, n_edges, n_variables):
        return generate_cost_func(self.accuracy,
                                         self.pre_calculated_weights, n_edges,
                                         n_variables)

    def generate_bounds(self, edge_weights, n_variables):
        return generate_bounds(self.pre_calculated_weights,
                               self.internal_scheme,
                               self.source_edge_scheme,
                               self.sink_edge_scheme, edge_weights,
                               n_variables)

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
        return generate_capacities_def(self.pre_calculated_weights, X,
                                              big_M, edge_weights,
                                              n_edges, n_internal_nodes,
                                              n_nodes, weighted_adj)

    def generate_duality_constraint(self, cut, n_internal_nodes):
        duality_constraint, duality_constraint_rhs = \
            generate_duality_constraint(
                cut, n_internal_nodes)
        return duality_constraint, duality_constraint_rhs

    def generate_flow_conditions(self, internal_adj, n_edges, n_internal_nodes,
                                 n_nodes, square_idx):
        return generate_flow_conditions(internal_adj, n_edges,
                                               n_internal_nodes, n_nodes,
                                               square_idx)

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
        return get_weight_scalers(self.source_edge_scheme,
                                         self.sink_edge_scheme,
                                         self.internal_scheme, X,
                                         half_internal_nodes,
                                         n_internal_nodes)

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

        return get_big_M_edges(half_internal_nodes)

    def generate_cut(self, adj, y):
        cut = generate_cut(adj, y)
        return cut

    def generate_sink_cut_constraints(self, adj, half_internal_nodes, n_edges,
                                      n_internal_nodes, n_nodes):
        sink_constraints, sink_constraints_rhs = \
            generate_sink_cut_constraints(
                adj, half_internal_nodes, n_edges, n_internal_nodes, n_nodes)
        return sink_constraints, sink_constraints_rhs

    def generate_source_cut_constraints(self, adj, half_internal_nodes, n_edges,
                                        n_internal_nodes, n_nodes):
        source_constraints, source_constraints_rhs = \
            generate_source_cut_constraints(
                adj, half_internal_nodes, n_edges, n_internal_nodes, n_nodes)
        return source_constraints, source_constraints_rhs

    def generate_internal_cut_constraints(self, adj, n_edges, n_internal_edges,
                                          n_internal_nodes, n_nodes):
        return generate_internal_cut_constraints(adj, n_edges,
                                                        n_internal_edges,
                                                        n_internal_nodes,
                                                        n_nodes)
