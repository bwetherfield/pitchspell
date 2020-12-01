import numpy as np
import pytest

from pitchspell.helper import generate_bounds, generate_weight_upper_bounds, \
    generate_cost_func
from pitchspell import helper

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

    @pytest.mark.xfail(reason='need to test function below')
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
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        )
        np.testing.assert_array_equal(cut, target_cut)

    @pytest.mark.skip(reason="not yet testing")
    def test_generate_flow_conditions(self):
        helper.generate_flow_conditions(
            internal_adj=[],
            n_edges=36,
            n_internal_nodes=4,
            n_nodes=6,
            square_idx=[]
        )
        assert False

    @pytest.mark.skip(reason="not yet testing")
    def test_get_weight_scalers(self):
        helper.get_weight_scalers(
            sink_edge_scheme=[],
            internal_scheme=[],
            half_internal_nodes=[],
            n_internal_nodes=6,
            pitch_classes=[],
        )
        assert False

    @pytest.mark.skip(reason="not yet testing")
    def test_generate_internal_cut_constraints(self):
        helper.generate_internal_cut_constraints(
            adj=[],
            n_edges=64,
            internal_edges=36,
            n_internal_nodes=6,
            n_nodes=8
        )
        assert False

    @pytest.mark.skip(reason="not yet testing")
    def test_generate_sink_cut_constraints(self):
        helper.generate_sink_cut_constraints(
            adj=[],
            half_internal_nodes=[],
            n_edges=36,
            n_internal_nodes=4,
            n_nodes=6
        )
        assert False

    @pytest.mark.skip(reason="not yet testing")
    def test_generate_source_cut_constraints(self):
        helper.generate_source_cut_constraints(
            adj=[],
            half_internal_nodes=2,
            n_edges=36,
            n_internal_nodes=4,
            n_nodes=6
        )
        assert False

    @pytest.mark.skip(reason="not yet testing")
    def test_generate_capacities_def(self):
        helper.generate_capacities_def(
            pre_calculated_weights=True,
            big_M=[],
            edge_weights=[],
            n_edges=36,
            n_internal_nodes=4,
            n_nodes=6,
            weighted_adj=[],
            pitch_classes=[]
        )
        assert False

    @pytest.mark.skip(reason="not yet testing")
    def test_extract_adjacencies(self):
        helper.extract_adjacencies(
            distance_cutoff=4,
            distance_rolloff=0.4,
            between_part_scalar=0.5,
            chains=[],
            ends=[],
            events=[],
            half_internal_nodes=2,
            n_internal_nodes=4,
            parts=[],
            starts=[]
        )
        assert False
