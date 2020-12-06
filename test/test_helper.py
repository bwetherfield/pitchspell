import numpy as np
import pytest

from pitchspell.helper import generate_bounds, generate_weight_upper_bounds, \
    generate_cost_func
from pitchspell import helper


@pytest.fixture
def within_part_adj():
    return helper.generate_within_part_adj(
        chains=np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2]),
        distance_cutoff=1,
        half_internal_nodes=5,
        events=np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3]),
        n_internal_nodes=10,
        parts=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    )


@pytest.fixture
def between_parts_adj():
    return helper.generate_between_parts_adj(
        starts=np.array([0, 1, 2, 4, 0, 3]),
        ends=np.array([1, 3, 3, 5, 1, 5]),
        parts=np.array([0, 0, 0, 0, 1, 1])
    )


class TestHelperFunctions:
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

    source_edge_scheme = np.array((13, 26, 3, 1, 13, 13, 26, 3, 0, 3, 1, 13))
    sink_edge_scheme = np.array((13, 1, 3, 26, 13, 13, 1, 3, 0, 3, 26, 13))
    target = np.array(
        [[0, 26, 0, 26, 26, 0, 26, 0, 0, 26, 26, 0, 0, 0, 26, 0,
          0, 26, 26, 0, 26, 0, 26, 0, 0, 0],
         [26, 0, 26, 0, 0, 26, 0, 26, 26, 0, 0, 26, 0, 0, 0, 26,
          26, 0, 0, 26, 0, 26, 0, 26, 0, 26],
         [0, 26, 0, 26, 26, 0, 0, 26, 26, 0, 0, 26, 26, 0, 0, 0,
          26, 0, 26, 0, 0, 26, 26, 0, 0, 0],
         [26, 0, 26, 0, 0, 26, 26, 0, 0, 26, 26, 0, 0, 26, 0, 0,
          0, 26, 0, 26, 26, 0, 0, 26, 0, 26],
         [26, 0, 26, 0, 0, 26, 26, 0, 26, 0, 26, 0, 26, 0, 26, 0,
          0, 0, 26, 0, 26, 0, 26, 0, 0, 0],
         [0, 26, 0, 26, 26, 0, 0, 26, 0, 26, 0, 26, 0, 26, 0, 26,
          0, 0, 0, 26, 0, 26, 0, 26, 0, 26],
         [26, 0, 0, 26, 26, 0, 0, 26, 0, 26, 26, 0, 0, 26, 26, 0,
          0, 26, 0, 0, 26, 0, 0, 26, 0, 0],
         [0, 26, 26, 0, 0, 26, 26, 0, 26, 0, 0, 26, 26, 0, 0, 26,
          26, 0, 0, 0, 0, 26, 26, 0, 0, 26],
         [0, 26, 26, 0, 26, 0, 0, 26, 0, 26, 0, 26, 26, 0, 26, 0,
          26, 0, 26, 0, 0, 0, 26, 0, 0, 0],
         [26, 0, 0, 26, 0, 26, 26, 0, 26, 0, 26, 0, 0, 26, 0, 26,
          0, 26, 0, 26, 0, 0, 0, 26, 0, 26],
         [26, 0, 0, 26, 26, 0, 26, 0, 0, 26, 0, 26, 0, 26, 26, 0,
          0, 26, 26, 0, 26, 0, 0, 0, 0, 0],
         [0, 26, 26, 0, 0, 26, 0, 26, 26, 0, 26, 0, 26, 0, 0, 26,
          26, 0, 0, 26, 0, 26, 0, 0, 0, 26],
         [0, 0, 26, 0, 26, 0, 0, 26, 26, 0, 0, 26, 0, 26, 26, 0,
          26, 0, 26, 0, 0, 26, 26, 0, 0, 0],
         [0, 0, 0, 26, 0, 26, 26, 0, 0, 26, 26, 0, 26, 0, 0, 26,
          0, 26, 0, 26, 26, 0, 0, 26, 0, 26],
         [26, 0, 0, 0, 26, 0, 26, 0, 26, 0, 26, 0, 26, 0, 0, 26,
          0, 26, 26, 0, 26, 0, 26, 0, 0, 0],
         [0, 26, 0, 0, 0, 26, 0, 26, 0, 26, 0, 26, 0, 26, 26, 0,
          26, 0, 0, 26, 0, 26, 0, 26, 0, 26],
         [0, 26, 26, 0, 0, 0, 0, 26, 26, 0, 0, 26, 26, 0, 0, 26,
          0, 26, 26, 0, 0, 26, 26, 0, 0, 0],
         [26, 0, 0, 26, 0, 0, 26, 0, 0, 26, 26, 0, 0, 26, 26, 0,
          26, 0, 0, 26, 26, 0, 0, 26, 0, 0],
         [26, 0, 26, 0, 26, 0, 0, 0, 26, 0, 26, 0, 26, 0, 26, 0,
          26, 0, 0, 26, 26, 0, 26, 0, 0, 0],
         [0, 26, 0, 26, 0, 26, 0, 0, 0, 26, 0, 26, 0, 26, 0, 26,
          0, 26, 26, 0, 0, 26, 0, 26, 0, 26],
         [26, 0, 0, 26, 26, 0, 26, 0, 0, 0, 26, 0, 0, 26, 26, 0,
          0, 26, 26, 0, 0, 26, 0, 26, 0, 0],
         [0, 26, 26, 0, 0, 26, 0, 26, 0, 0, 0, 26, 26, 0, 0, 26,
          26, 0, 0, 26, 26, 0, 26, 0, 0, 26],
         [26, 0, 26, 0, 26, 0, 0, 26, 26, 0, 0, 0, 26, 0, 26, 0,
          26, 0, 26, 0, 0, 26, 0, 26, 0, 0],
         [0, 26, 0, 26, 0, 26, 26, 0, 0, 26, 0, 0, 0, 26, 0, 26,
          0, 26, 0, 26, 26, 0, 26, 0, 0, 26],
         [26, 0, 26, 0, 26, 0, 26, 0, 26, 0, 26, 0, 26, 0, 26, 0,
          0, 0, 26, 0, 26, 0, 26, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    def test_generate_bounds_weights_fixed(self):
        bounds = generate_bounds(True, self.internal_scheme,
                                 self.source_edge_scheme,
                                 self.sink_edge_scheme, 676)
        assert bounds == (0, None)

    def test_generate_weight_upper_bounds(self):
        upper_bounds = generate_weight_upper_bounds(self.internal_scheme,
                                                    self.sink_edge_scheme,
                                                    self.source_edge_scheme)

        np.testing.assert_array_equal(upper_bounds, self.target)

    def test_generate_bounds_weights_variable(self):
        bounds = generate_bounds(False, self.internal_scheme,
                                 self.source_edge_scheme,
                                 self.sink_edge_scheme, 677)
        bound_target = [(0, None)] + list(
            zip(np.zeros_like(self.target).flatten(), self.target.flatten())
        )
        np.testing.assert_array_equal(bounds, bound_target)

    def test_generate_cost_function_with_weights_fixed(self):
        c = generate_cost_func(4, True, 9, 19)
        target = 18 * [0] + [4]
        np.testing.assert_array_equal(c, target)

    def test_generate_cost_function_with_weights_variable(self):
        c = generate_cost_func(4, False, 9, 19 + 676)
        target = 18 * [0] + [4] + 676 * [-1]
        np.testing.assert_array_equal(c, target)

    def test_get_big_m_edges(self):
        adj, big_M = helper.get_big_M_edges(
            half_internal_nodes=4,
        )
        target_array = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )
        target_bigM_array = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, np.inf, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, np.inf, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, np.inf, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )
        np.testing.assert_array_equal(adj, target_array)
        np.testing.assert_array_equal(big_M, target_bigM_array)

    def test_generate_duality_constraint(self):
        duality, duality_rhs = helper.generate_duality_constraint(
            cut=np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            ),
            n_internal_nodes=4
        )
        duality_target = np.array(
            [-1] * 4 + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,
                        0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [-1]
        )
        np.testing.assert_array_equal(duality, duality_target)
        np.testing.assert_array_equal(duality_rhs, [0])

    def test_generate_cut(self):
        cut = helper.generate_cut(
            adj=np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            ),
            y=[0, 0, 0, 1]
        )
        target_cut = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        )
        np.testing.assert_array_equal(cut, target_cut)

    def test_generate_flow_conditions(self):
        adj = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        )
        flow_conditions, flow_conditions_rhs = helper.generate_flow_conditions(
            adj=adj,
            n_internal_nodes=4,
            n_nodes=6,
        )
        target_flow_conditions = np.concatenate(
            [np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]).reshape(1, -1),
             np.array([
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1],
                 [0, -1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]
             ]).reshape(1, -1),
             np.array([
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, -1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]
             ]).reshape(1, -1),
             np.array([
                 [0, 0, 0, -1, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]
             ]).reshape(1, -1)
             ], axis=0)
        np.testing.assert_array_equal(flow_conditions,
                                      target_flow_conditions)

    def test_get_weight_scalers(self):
        weight_scalers = helper.get_weight_scalers(
            source_edge_scheme=self.source_edge_scheme,
            sink_edge_scheme=self.sink_edge_scheme,
            internal_scheme=self.internal_scheme,
            half_internal_nodes=2,
            n_internal_nodes=4,
            pitch_classes=np.array([0, 0, 1, 1]),
        )
        target_weight_scalers = np.array([
            [0, 0, 0, 2, 0, 0],
            [0, 0, 1, 0, 0, 13],
            [0, 1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 1],
            [13, 0, 26, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(weight_scalers, target_weight_scalers)

    def test_generate_internal_cut_constraints(self):
        constraints, rhs = helper.generate_internal_cut_constraints(
            adj=np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            ),
            n_internal_nodes=4
        )

        target_constraints = np.concatenate(
            [
                np.append([-1, 0, 0, 1],
                          [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
                np.append([1, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
                np.append([0, 1, -1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
                np.append([0, 0, 1, -1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
            ],
            axis=0)
        np.testing.assert_array_equal(constraints, target_constraints)
        np.testing.assert_array_equal(rhs, np.zeros((4,), dtype=int))

    def test_generate_sink_cut_constraints(self):
        constraints, rhs = helper.generate_sink_cut_constraints(
            adj=np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            ),
            n_internal_nodes=4,
        )
        target_constraints = np.concatenate(
            [
                np.append([0, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
                np.append([0, 0, 0, -1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
            ],
            axis=0)
        target_rhs = np.array([-1, -1])
        np.testing.assert_array_equal(constraints, target_constraints)
        np.testing.assert_array_equal(rhs, target_rhs)

    def test_generate_source_cut_constraints(self):
        constraints, rhs = helper.generate_source_cut_constraints(
            adj=np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            ),
            n_internal_nodes=4
        )
        target_constraints = np.concatenate(
            [
                np.append([1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
                np.append([0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]).reshape(1, -1),
            ],
            axis=0)
        target_rhs = np.array([0, 0])
        np.testing.assert_array_equal(constraints, target_constraints)
        np.testing.assert_array_equal(rhs, target_rhs)

    @pytest.mark.skip(reason="not yet testing")
    def test_generate_capacities_def(self):
        helper.generate_capacities_def(
            pre_calculated_weights=True,
            big_M=[],
            weight_scalers=[],
            n_edges=36,
            n_internal_nodes=4,
            n_nodes=6,
            weighted_adj=[],
            pitch_classes=[]
        )
        assert False

    @pytest.mark.xfail(reason="not yet fixed")
    def test_extract_adjacencies(self):
        big_M_adj, big_M, adj, weighted_adj = helper.extract_adjacencies(
            distance_cutoff=4,
            distance_rolloff=0.4,
            between_part_scalar=0.5,
            chains=np.array([0, 0, 0, 0, 1, 1]),
            ends=np.array([1, 1, 3, 4, 5, 6]),
            events=np.array([0, 0, 1, 2, 3, 4]),
            half_internal_nodes=3,
            n_internal_nodes=6,
            parts=np.array([0, 0, 0, 0, 1, 1]),
            starts=np.array([0, 0, 2, 3, 4, 5])
        )
        print(weighted_adj)
        assert False

    def test_generate_between_parts_adj(self, between_parts_adj):
        target_adj = np.array(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )
        np.testing.assert_array_equal(between_parts_adj, target_adj)

    def test_generate_endweighting(self):
        endweighting = helper.generate_endweighting(
            timefactor=np.array([1.0, 1.5, 2.0, 2.5]),
            n_internal_nodes=4
        )

        target_endweighting = np.array([
            [1., 1.5, 2., 2.5, 0., 1.],
            [1.5, 2.25, 3., 3.75, 0., 1.5],
            [2., 3., 4., 5., 0., 2.],
            [2.5, 3.75, 5., 6.25, 0., 2.5],
            [1., 1.5, 2., 2.5, 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        np.testing.assert_array_almost_equal(endweighting, target_endweighting)

    def test_generate_within_parts_adj(self, within_part_adj):
        target = [
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
            ])
        ]

        np.testing.assert_array_equal(within_part_adj[0], target[0])
        np.testing.assert_array_equal(within_part_adj[1], target[1])

    def test_get_big_m_edges(self):
        big_M_adj, big_M = helper.get_big_M_edges(4)
        target_big_M_adj = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        target_big_M = np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [np.inf, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., np.inf, 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., np.inf, 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., np.inf, 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])
        np.testing.assert_array_equal(big_M_adj, target_big_M_adj)
        np.testing.assert_array_equal(big_M, target_big_M)

    @pytest.mark.skip('incomplete')
    def test_generate_adj(self):
        adj = helper.generate_adj(
            between_part_adj=[],
            half_internal_nodes=4,
            n_internal_nodes=8,
            within_chain_adjs=[]
        )
        assert False


def test_cut_2_by_2_diagonal():
    no_diag = helper.cut_2_by_2_diagonal(4)
    target_no_diag = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    np.testing.assert_array_equal(no_diag, target_no_diag)
