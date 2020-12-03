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


def test_hop_adjacencies():
    hop_adjacencies = prepare_edges.hop_adjacencies(2, 6)

    target_hop_adjacencies = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ]
    )
    np.testing.assert_array_equal(hop_adjacencies, target_hop_adjacencies)


class TestAddNode:
    def test_add_node_in_and_out(self):
        arr = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]
        )
        output = prepare_edges.add_node(arr, in_edges=arr[0], out_edges=arr[0])

        target_arr = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0]
            ]
        )
        np.testing.assert_array_equal(output, target_arr)

    def test_add_node_in_edges(self):
        arr = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]
        )
        output = prepare_edges.add_node(arr, in_edges=arr[0])

        target_arr = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(output, target_arr)

    def test_add_node_out_edges(self):
        arr = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]
        )
        output = prepare_edges.add_node(arr, out_edges=arr[0])

        target_arr = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        np.testing.assert_array_equal(output, target_arr)

    def test_add_node_out_then_in(self):
        arr = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]
        )
        output = prepare_edges.add_node(arr, out_edges=arr[0])
        output = prepare_edges.add_node(output, in_edges=np.append(arr[0],0))

        target_arr = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(output, target_arr)

    def test_add_node_in_then_out(self):
        arr = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]
        )
        output = prepare_edges.add_node(arr, in_edges=arr[0])
        output = prepare_edges.add_node(output, out_edges=np.append(arr[0],0))

        target_arr = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(output, target_arr)
