import numpy as np

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

    def test_generate_capacities_def(self):
        assert False

    def test_generate_flow_conditions(self):
        assert False

    def test_get_weight_scalers(self):
        assert False

    def test_generate_duality_constraint(self):
        assert False

    def test_get_big_m_edges(self):
        assert False

    def test_generate_cut(self):
        assert False

    def test_generate_internal_cut_constraints(self):
        assert False

    def test_generate_source_cut_constraints(self):
        assert False

    def test_generate_sink_cut_constraints(self):
        assert False

    def test_extract_adjacencies(self):
        assert False
