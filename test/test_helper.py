from unittest import TestCase
import numpy as np

from pitchspell.helper import generate_bounds, generate_weight_upper_bounds


class Test(TestCase):
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

    def test_generate_bounds_weights_fixed(self):
        bounds = generate_bounds(True, self.internal_scheme,
                                 self.source_edge_scheme,
                                 self.sink_edge_scheme, 676)
        self.assertEqual(bounds, (0, None))

    def test_generate_weight_upper_bounds(self):
        upper_bounds = generate_weight_upper_bounds(self.internal_scheme,
                                                    self.sink_edge_scheme,
                                                    self.source_edge_scheme)
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
        self.assertTrue(np.array_equal(upper_bounds, target))