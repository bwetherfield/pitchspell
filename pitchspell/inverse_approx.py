from sklearn.base import BaseEstimator
import numpy as np
from scipy.optimize import linprog

from helper import space_capacities_def
from pitchspell.helper import generate_bounds, generate_cost_func, \
    generate_flow_conditions, \
    get_weight_scalers, generate_duality_constraint, \
    get_big_M_edges, extract_cut, \
    generate_internal_cut_constraints, \
    generate_source_cut_constraints, generate_sink_cut_constraints, \
    extract_adjacencies, n_pitch_classes, n_pitch_class_internal_nodes, \
    n_pitch_class_nodes, n_pitch_class_edges
from pitchspell import helper
from pitchspell.pullback import pad


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
        and additional binary index) for use with tests data sets (and
        populating linear program matrices accordingly).

        Parameters
        ----------
        X: ndarray
            2D array
        y: ndarray
            1D array

        """
        n_internal_nodes = X.shape[0]
        half_internal_nodes = n_internal_nodes // 2
        n_nodes = n_internal_nodes + 2
        n_edges = pow(n_nodes, 2)
        n_variables = 2 * n_edges + 1 if self.pre_calculated_weights else \
            2 * n_edges + 1 + n_pitch_class_edges

        A_eq, A_ub, b_eq, b_ub, bounds, c = self.prepare_fit_input(
            X,
            half_internal_nodes,
            n_edges,
            n_internal_nodes,
            n_nodes,
            n_variables,
            y
        )

        _output = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds)
        # Accuracy score given by size of duality gap of linear program
        self._score = _output.x[2 * n_edges]

        # Record edge weight schemes
        if not self.pre_calculated_weights:
            weights_unfiltered = _output.x[-n_pitch_class_edges:]
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

    def prepare_fit_input(self, X, half_internal_nodes, n_edges,
                          n_internal_nodes, n_nodes, n_variables, y):
        events = X[:, 0]
        chains = X[:, 1]
        parts = X[:, 2]
        starts = X[:, 4]
        ends = X[:, 4] + X[:, 5]
        time_factor = X[:, 6]
        adj, weighted_adj = extract_adjacencies(self.distance_cutoff,
                                         self.distance_rolloff,
                                         self.between_part_scalar,
                                         chains, ends, events,
                                         time_factor,
                                         half_internal_nodes,
                                         n_internal_nodes,
                                         parts, starts)
        internal_adj = np.zeros((n_nodes, n_nodes))
        internal_adj[-2:, -2:] = adj[-2:, -2:]
        # ----------------------------------------
        # EQUALITY CONSTRAINTS
        # sum_(i=1)^n f_(i,k) - sum_(j=1)^n f_(k,j) = 0 for all k != s,t
        flow_conditions, flow_conditions_rhs = generate_flow_conditions(
            adj,
            n_internal_nodes,
            n_nodes)
        # sum_(i=1)^n f_s,i - sum_(e in cut) c_e = delta
        cut = extract_cut(adj, y)
        constraint, rhs = \
            generate_duality_constraint(
                cut, n_internal_nodes)
        result = constraint, rhs
        duality_constraint, duality_constraint_rhs = \
            result
        # Capacity variable definitions
        big_M_adj, big_M = get_big_M_edges(half_internal_nodes)
        pitch_classes = X[:, 3]
        capacities_def: np.ndarray
        capacities_def_rhs: np.ndarray
        if self.pre_calculated_weights:
            weight_scalers = get_weight_scalers(self.source_edge_scheme,
                                                self.sink_edge_scheme,
                                                self.internal_scheme,
                                                half_internal_nodes,
                                                n_internal_nodes, pitch_classes)
            capacities_def, capacities_def_rhs = \
                helper.generate_capacities_def_weights_fixed(big_M, n_edges,
                                                             weighted_adj,
                                                             weight_scalers)
        else:
            capacities_def, capacities_def_rhs = \
                helper.generate_capacities_def_weights_variable(
                    big_M,
                    n_edges,
                    n_internal_nodes, n_nodes, pitch_classes,
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
        capacities_def_spaced = space_capacities_def(self.pre_calculated_weights,
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
        c = generate_cost_func(self.accuracy,
                               self.pre_calculated_weights, n_edges,
                               n_variables)
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
        bounds = generate_bounds(self.pre_calculated_weights,
                                 self.internal_scheme,
                                 self.source_edge_scheme, self.sink_edge_scheme,
                                 n_variables)
        return A_eq, A_ub, b_eq, b_ub, bounds, c

    def score(self, X, y=None):
        return self._score

    def predict(self, X):
        """
        Forward solver for `self`.

        Parameters
        ----------
        X: ndarray
            2D array

        Returns
        -------
        numpy.ndarray
            1D array \\
            Variable order: x_i (N), y_ij ((N+2)^2) with final two columns of
            implicit square of variables s, and t
        """
        A_ub, b_ub, bounds, c = self.prepare_predict_input(X)
        return linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds).x

    def prepare_predict_input(self, X):
        n_internal_nodes = X.shape[0]
        n_nodes = n_internal_nodes + 2
        n_edges = pow(n_nodes, 2)
        half_internal_nodes = n_internal_nodes // 2
        events = X[:, 0]
        chains = X[:, 1]
        parts = X[:, 2]
        starts = X[:, 4]
        ends = X[:, 4] + X[:, 5]
        time_factor = X[:, 6]
        adj, weighted_adj = extract_adjacencies(self.distance_cutoff,
                                         self.distance_rolloff,
                                         self.between_part_scalar,
                                         chains, ends, events,
                                         time_factor,
                                         half_internal_nodes,
                                         n_internal_nodes,
                                         parts, starts)
        pitch_classes = X[:, 3]
        weight_scalers = get_weight_scalers(self.source_edge_scheme,
                                            self.sink_edge_scheme,
                                            self.internal_scheme,
                                            half_internal_nodes,
                                            n_internal_nodes, pitch_classes)
        big_M_adj, big_M = get_big_M_edges(half_internal_nodes)
        # x_j - x_i - y_(i,j) <= 0 for all i, j != s, t, where (i,j) is an edge
        internal_constraints, internal_constraints_rhs = \
            generate_internal_cut_constraints(adj, n_internal_nodes)
        # x_i - y_(s, i) <=0 for all i: (s,i) is an edge
        source_constraints, source_constraints_rhs = \
            generate_source_cut_constraints(adj, n_internal_nodes)
        # - x_i - y_(i, t) <= -1 for all i: (i,t) is an edge
        sink_constraints, sink_constraints_rhs = \
            generate_sink_cut_constraints(adj, n_internal_nodes)
        # x_i <= 1 bounds
        # x_i, y_ij >= 0 bounds (default)
        bounds = [(0, 1)] * n_internal_nodes + [(0, n_internal_nodes)] * n_edges
        capacities = (weighted_adj * weight_scalers + big_M).flatten()
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
        return A_ub, b_ub, bounds, c
