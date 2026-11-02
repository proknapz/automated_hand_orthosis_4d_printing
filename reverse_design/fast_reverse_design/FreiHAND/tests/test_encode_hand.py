import unittest

import numpy as np

from FreiHAND.encode_hand import (
    DEFAULT_RESOLUTION,
    FREIHAND_KEYPOINT_NAMES,
    decode_hand_geometry,
    decode_named_keypoints,
    encode_hand_geometry,
)


class EncodeHandTests(unittest.TestCase):
    def test_shape_dtype_range_and_finite_output(self):
        vertices = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        keypoints = np.array([[0.0, 0.25, -0.25]])

        result = encode_hand_geometry(vertices, keypoints=keypoints, resolution=16)

        self.assertEqual(result.shape, (16, 16, 16, 2))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(set(np.unique(result)).issubset({0, 1}))
        self.assertEqual(int(result[..., 0].sum()), 2)
        self.assertEqual(int(result[..., 1].sum()), 1)

    def test_deterministic_and_face_samples_fill_triangle_cells(self):
        vertices = np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        first = encode_hand_geometry(vertices, faces=faces, resolution=16)
        second = encode_hand_geometry(vertices, faces=faces, resolution=16)

        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(first[..., 0].sum()), len(vertices))

    def test_multiple_geometry_samples_encode_independently(self):
        first = np.array([[-0.8, -0.8, -0.8], [-0.6, -0.6, -0.6]])
        second = np.array([[0.6, 0.6, 0.6], [0.8, 0.8, 0.8]])
        first_grid = encode_hand_geometry(first, resolution=8)
        second_grid = encode_hand_geometry(second, resolution=8)

        self.assertEqual(int(first_grid[..., 0].sum()), 2)
        self.assertEqual(int(second_grid[..., 0].sum()), 2)
        self.assertNotEqual(np.argwhere(first_grid[..., 0]).tolist(),
                            np.argwhere(second_grid[..., 0]).tolist())

    def test_coordinate_sign_maps_to_increasing_axis_indices(self):
        negative = encode_hand_geometry(np.array([[-0.5, 0.0, 0.0]]), resolution=8)
        positive = encode_hand_geometry(np.array([[0.5, 0.0, 0.0]]), resolution=8)

        negative_index = np.argwhere(negative[..., 0] == 1)[0]
        positive_index = np.argwhere(positive[..., 0] == 1)[0]
        self.assertGreater(positive_index[0], negative_index[0])
        self.assertEqual(positive_index[1], negative_index[1])
        self.assertEqual(positive_index[2], negative_index[2])

    def test_decoder_returns_cell_centers(self):
        vertex = np.array([[0.0, 0.0, 0.0]])
        grid = encode_hand_geometry(vertex, resolution=8)
        decoded, keypoints = decode_hand_geometry(grid)

        self.assertEqual(decoded.shape, (1, 3))
        self.assertEqual(keypoints.shape, (0, 3))
        np.testing.assert_allclose(decoded[0], [0.125, 0.125, 0.125])
        self.assertLessEqual(
            np.linalg.norm(decoded[0] - vertex[0]),
            np.sqrt(3) * (2.0 / 8) / 2.0,
        )

    def test_named_landmark_channels_preserve_identity_at_shared_voxel(self):
        vertices = np.array([[0.0, 0.0, 0.0]])
        keypoints = np.array([[0.1, 0.1, 0.1], [0.11, 0.11, 0.11]])
        names = ("index_joint_1", "middle_joint_1")

        grid = encode_hand_geometry(
            vertices,
            keypoints=keypoints,
            keypoint_names=names,
            resolution=8,
        )
        decoded = decode_named_keypoints(grid, names)

        self.assertEqual(grid.shape, (8, 8, 8, 3))
        self.assertEqual(int(grid[..., 1].sum()), 1)
        self.assertEqual(int(grid[..., 2].sum()), 1)
        np.testing.assert_array_equal(decoded[names[0]], decoded[names[1]])
        _, union_points = decode_hand_geometry(grid)
        self.assertEqual(union_points.shape, (1, 3))

    def test_named_landmark_encoding_rejects_invalid_channel_names(self):
        vertices = np.array([[0.0, 0.0, 0.0]])
        keypoints = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        invalid_names = [
            ("same", "same"),
            ("one",),
            ("", "two"),
        ]
        for names in invalid_names:
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    encode_hand_geometry(
                        vertices, keypoints=keypoints, keypoint_names=names
                    )

    def test_freihand_landmark_mapping_contains_all_21_ordered_labels(self):
        self.assertEqual(len(FREIHAND_KEYPOINT_NAMES), 21)
        self.assertEqual(len(set(FREIHAND_KEYPOINT_NAMES)), 21)
        self.assertEqual(FREIHAND_KEYPOINT_NAMES[0], "wrist")
        self.assertEqual(FREIHAND_KEYPOINT_NAMES[4], "thumb_tip")
        self.assertEqual(FREIHAND_KEYPOINT_NAMES[8], "index_tip")
        self.assertEqual(FREIHAND_KEYPOINT_NAMES[12], "middle_tip")
        self.assertEqual(FREIHAND_KEYPOINT_NAMES[16], "ring_tip")
        self.assertEqual(FREIHAND_KEYPOINT_NAMES[20], "pinky_tip")

    def test_default_resolution_matches_selected_prototype_setting(self):
        self.assertEqual(DEFAULT_RESOLUTION, 96)
        result = encode_hand_geometry(np.array([[0.0, 0.0, 0.0]]))
        self.assertEqual(result.shape, (96, 96, 96, 2))

    def test_named_landmark_quantization_respects_prototype_error_bounds(self):
        rng = np.random.RandomState(2030)
        resolution = 96
        bounds = (-1.0, 1.0)
        cell_width = (bounds[1] - bounds[0]) / resolution
        keypoints = rng.uniform(bounds[0], bounds[1], size=(21, 3))

        grid = encode_hand_geometry(
            np.array([[0.0, 0.0, 0.0]]),
            keypoints=keypoints,
            keypoint_names=FREIHAND_KEYPOINT_NAMES,
            resolution=resolution,
            bounds=bounds,
        )
        decoded = decode_named_keypoints(grid, FREIHAND_KEYPOINT_NAMES, bounds)
        centers = np.stack(
            [decoded[name][0] for name in FREIHAND_KEYPOINT_NAMES], axis=0
        )
        point_errors = np.linalg.norm(centers - keypoints, axis=1)
        depth_errors = np.abs(centers[:, 2] - keypoints[:, 2])
        source_gaps = np.linalg.norm(np.diff(keypoints, axis=0), axis=1)
        decoded_gaps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        gap_errors = np.abs(decoded_gaps - source_gaps)

        self.assertTrue(np.all(point_errors <= np.sqrt(3.0) * cell_width / 2.0))
        self.assertTrue(np.all(depth_errors <= cell_width / 2.0))
        self.assertTrue(np.all(gap_errors <= np.sqrt(3.0) * cell_width))

    def test_inclusive_bounds_map_to_first_and_last_cells(self):
        vertices = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        grid = encode_hand_geometry(vertices, resolution=8)

        occupied = np.argwhere(grid[..., 0] == 1)
        self.assertIn((0, 0, 0), map(tuple, occupied))
        self.assertIn((7, 7, 7), map(tuple, occupied))

    def test_invalid_vertices_and_nonfinite_values_raise(self):
        invalid_inputs = [
            np.empty((0, 3)),
            np.zeros((3, 2)),
            np.array([[np.nan, 0.0, 0.0]]),
            np.array([[np.inf, 0.0, 0.0]]),
            np.array([[1.01, 0.0, 0.0]]),
        ]
        for vertices in invalid_inputs:
            with self.subTest(vertices=vertices):
                with self.assertRaises(ValueError):
                    encode_hand_geometry(vertices)

    def test_invalid_faces_keypoints_resolution_and_bounds_raise(self):
        vertices = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        invalid_cases = [
            {"faces": np.array([[0, 1]])},
            {"faces": np.array([[0, 1, 5]])},
            {"faces": np.array([[0.0, 1.0, 2.0]])},
            {"keypoints": np.array([[0.0, np.inf, 0.0]])},
            {"keypoints": np.array([[2.0, 0.0, 0.0]])},
            {"resolution": 1},
            {"resolution": 4.5},
            {"bounds": (1.0, -1.0)},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    encode_hand_geometry(vertices, **kwargs)

    def test_out_of_bounds_input_is_rejected_not_clipped(self):
        vertices = np.array([[-0.5, 0.0, 0.0], [1.1, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "rather than clipping"):
            encode_hand_geometry(vertices, resolution=8)


if __name__ == "__main__":
    unittest.main()
