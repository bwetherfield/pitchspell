import numpy as np
import pytest

from pitchspell.helper import generate_bounds, generate_weight_upper_bounds, \
    generate_cost_func
from pitchspell import helper


@pytest.fixture
def internal_scheme():
    return np.array([
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


@pytest.fixture
def source_edge_scheme():
    return np.array((13, 26, 3, 1, 13, 13, 26, 3, 0, 3, 1, 13))


@pytest.fixture
def sink_edge_scheme():
    return np.array((13, 1, 3, 26, 13, 13, 1, 3, 0, 3, 26, 13))


@pytest.fixture
def chains():
    return np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])


@pytest.fixture
def events():
    return np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])


@pytest.fixture
def pitches():
    return np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])


@pytest.fixture
def weight_scalers(internal_scheme, source_edge_scheme, sink_edge_scheme,
                   pitches):
    return helper.get_weight_scalers(
        source_edge_scheme=source_edge_scheme,
        sink_edge_scheme=sink_edge_scheme,
        internal_scheme=internal_scheme,
        half_internal_nodes=5,
        n_internal_nodes=10,
        pitch_classes=pitches
    )


@pytest.fixture
def parts():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


@pytest.fixture
def starts():
    return np.array([0, 0, 1, 1, 0, 0, 0, 0, 3, 3])


@pytest.fixture
def ends():
    return np.array([1, 1, 4, 4, 1, 1, 1, 1, 4, 4])


@pytest.fixture
def timefactor():
    return np.array([1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2])


@pytest.fixture
def endweighting(timefactor):
    return helper.generate_endweighting(timefactor)


@pytest.fixture
def within_part_adj(chains, events, parts):
    return helper.generate_within_part_adj(
        chains=chains,
        distance_cutoff=1,
        half_internal_nodes=5,
        events=events,
        n_internal_nodes=10,
        parts=parts
    )


@pytest.fixture
def between_parts_adj(starts, ends, parts):
    return helper.generate_between_parts_adj(
        starts=starts,
        ends=ends,
        parts=parts
    )


@pytest.fixture
def adj(within_part_adj, between_parts_adj):
    return helper.generate_adj(
        between_part_adj=between_parts_adj,
        half_internal_nodes=5,
        n_internal_nodes=10,
        within_chain_adjs=within_part_adj
    )


@pytest.fixture
def weighted_adj(between_parts_adj, adj, endweighting, within_part_adj):
    return helper.generate_weighted_adj(
        between_parts_adj=between_parts_adj,
        between_part_scalar=0.5,
        distance_rolloff=0.9,
        adj=adj,
        endweighting=endweighting,
        within_chain_adjs=within_part_adj
    )


@pytest.fixture
def big_M_adjs():
    return helper.get_big_M_edges(5)


class TestHelperFunctions:
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

    def test_generate_bounds_weights_fixed(self, internal_scheme,
                                           source_edge_scheme, sink_edge_scheme):
        bounds = generate_bounds(True, internal_scheme,
                                 source_edge_scheme,
                                 sink_edge_scheme, 676)
        assert bounds == (0, None)

    def test_generate_weight_upper_bounds(self, internal_scheme,
                                          source_edge_scheme, sink_edge_scheme):
        upper_bounds = generate_weight_upper_bounds(internal_scheme,
                                                    sink_edge_scheme,
                                                    source_edge_scheme)

        np.testing.assert_array_equal(upper_bounds, self.target)

    def test_generate_bounds_weights_variable(self, internal_scheme,
                                              source_edge_scheme,
                                              sink_edge_scheme):
        bounds = generate_bounds(False, internal_scheme,
                                 source_edge_scheme,
                                 sink_edge_scheme, 677)
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

    def test_get_weight_scalers(self, internal_scheme, source_edge_scheme,
                                sink_edge_scheme):
        weight_scalers = helper.get_weight_scalers(
            source_edge_scheme=source_edge_scheme,
            sink_edge_scheme=sink_edge_scheme,
            internal_scheme=internal_scheme,
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

    def test_get_weight_scalers_larger(self, pitches, weight_scalers):
        target_weight_scalers = np.array([
            [0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 13],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            [2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1],
            [0, 1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 13],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 1],
            [0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 13],
            [13, 0, 26, 0, 13, 0, 26, 0, 13, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

    def test_generate_capacities_def_weights_variable(
            self, weighted_adj, big_M_adjs, pitches
    ):
        capacities, rhs = helper.generate_capacities_def(
            pre_calculated_weights=False,
            big_M=big_M_adjs[1],
            n_edges=144, n_internal_nodes=10,
            n_nodes=12, weighted_adj=weighted_adj,
            pitched_information=pitches)
        N = len(pitches)
        pitches_extended = np.append(pitches, [12, 12])
        pitches_plus_parity = 2 * pitches_extended + (np.arange(12) % 2)
        np.testing.assert_array_equal(
            rhs.nonzero()[0], np.arange(len(pitches_extended),
                                        144 - helper.n_pitch_class_nodes,
                                        helper.n_pitch_class_nodes)
        )
        locations = np.where(capacities[:, 144:])
        weighted_nonzero = weighted_adj.nonzero()
        np.testing.assert_allclose(
            weighted_nonzero[0] * len(pitches_extended) +
            weighted_nonzero[1], locations[0])
        np.testing.assert_allclose(
            pitches_plus_parity[weighted_nonzero[0]] * 26 +
            pitches_plus_parity[weighted_nonzero[1]],
            locations[1]
        )

    def test_generate_capacities_def_weights_variable(
            self, weighted_adj, big_M_adjs, pitches, weight_scalers
    ):
        capacities, rhs = helper.generate_capacities_def(
            pre_calculated_weights=True,
            big_M=big_M_adjs[1],
            n_edges=144, n_internal_nodes=10,
            n_nodes=12, weighted_adj=weighted_adj,
            pitched_information=weight_scalers)
        np.testing.assert_allclose(capacities, np.eye(144,
                                                                dtype=float))
        np.testing.assert_allclose(
            rhs, ((weighted_adj * weight_scalers) + big_M_adjs[1]).flatten()
        )
        assert rhs[12] == np.inf

    def test_extract_adjacencies(self, chains, ends, events, starts, parts,
                                 timefactor, weighted_adj, adj, big_M_adjs):
        test_big_M_adj, test_big_M, test_adj, test_weighted_adj = \
            helper.extract_adjacencies(
            distance_cutoff=1,
            distance_rolloff=0.9,
            between_part_scalar=0.5,
            chains=chains,
            ends=ends,
            events=events,
            timefactor=timefactor,
            half_internal_nodes=5,
            n_internal_nodes=10,
            parts=parts,
            starts=starts
        )
        np.testing.assert_allclose(test_big_M, big_M_adjs[1])
        np.testing.assert_allclose(test_big_M_adj, big_M_adjs[0])
        np.testing.assert_allclose(test_adj, adj)
        np.testing.assert_allclose(test_weighted_adj, weighted_adj)

    def test_generate_between_parts_adj(self, between_parts_adj):
        target_adj = np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(between_parts_adj, target_adj)

    def test_generate_endweighting_larger(self, timefactor):
        endweighting = helper.generate_endweighting(timefactor)
        target = np.array([
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 1.],
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 1.],
            [1.1, 1.1, 1.21, 1.21, 1.1, 1.1, 1.1, 1.1, 1.32, 1.32, 0., 1.1],
            [1.1, 1.1, 1.21, 1.21, 1.1, 1.1, 1.1, 1.1, 1.32, 1.32, 0., 1.1],
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 1.],
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 1.],
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 1.],
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 1.],
            [1.2, 1.2, 1.32, 1.32, 1.2, 1.2, 1.2, 1.2, 1.44, 1.44, 0., 1.2],
            [1.2, 1.2, 1.32, 1.32, 1.2, 1.2, 1.2, 1.2, 1.44, 1.44, 0., 1.2],
            [1., 1., 1.1, 1.1, 1., 1., 1., 1., 1.2, 1.2, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])
        np.testing.assert_allclose(endweighting, target)

    def test_generate_endweighting(self):
        endweighting = helper.generate_endweighting(
            timefactor=np.array([1.0, 1.5, 2.0, 2.5])
        )

        target_endweighting = np.array([
            [1., 1.5, 2., 2.5, 0., 1.],
            [1.5, 2.25, 3., 3.75, 0., 1.5],
            [2., 3., 4., 5., 0., 2.],
            [2.5, 3.75, 5., 6.25, 0., 2.5],
            [1., 1.5, 2., 2.5, 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        np.testing.assert_allclose(endweighting, target_endweighting)

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

    def test_generate_adj(self, between_parts_adj, within_part_adj, adj):
        target = np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )
        np.testing.assert_array_equal(adj, target)

    def test_generate_weighted_adj(self, between_parts_adj, within_part_adj,
                                   adj, endweighting, weighted_adj):
        target = np.array(
            [
                [0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.66, 0.66, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.66, 0.66, 0., 1.1],
                [0.5, 0.5, 0., 0., 0., 0., 1., 1., 1.08, 1.08, 0., 0.],
                [0.5, 0.5, 0., 0., 0., 0., 1., 1., 1.08, 1.08, 0., 1.],
                [0.5, 0.5, 0., 0., 1., 1., 0., 0., 1.08, 1.08, 0., 0.],
                [0.5, 0.5, 0., 0., 1., 1., 0., 0., 1.08, 1.08, 0., 1.],
                [0., 0., 0.66, 0.66, 1.08, 1.08, 1.08, 1.08, 0., 0., 0., 0.],
                [0., 0., 0.66, 0.66, 1.08, 1.08, 1.08, 1.08, 0., 0., 0., 1.2],
                [1., 0., 1.1, 0., 1., 0., 1., 0., 1.2, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        )
        np.testing.assert_allclose(target, weighted_adj)


def test_cut_2_by_2_diagonal():
    no_diag = helper.cut_2_by_2_diagonal(4)
    target_no_diag = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    np.testing.assert_array_equal(no_diag, target_no_diag)
