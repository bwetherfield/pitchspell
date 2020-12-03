import numpy as np

from pitchspell import prepare_edges


def test_concurrencies():
    concurrencies = prepare_edges.concurrencies(starts=np.array([0, 0, 3, 5]),
                                                ends=np.array([2, 2, 6, 7]))
    target_concurrencies = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
    )
    np.testing.assert_array_equal(concurrencies, target_concurrencies)
