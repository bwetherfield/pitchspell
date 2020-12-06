import numpy as np

from pitchspell import minimum_cut

class TestCut:
    def test_generate_complete_cut(self):
        cut = minimum_cut.generate_complete_cut(np.array(
            [0, 1, 1, 0, 1, 1, 1, 0, 0]
        ))
        target_cut = np.array(
            [
                [0, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 0, 0]
            ]
        )
        np.testing.assert_array_equal(cut, target_cut)
