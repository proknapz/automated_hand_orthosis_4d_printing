import unittest

import numpy as np

try:
    import tensorflow as tf

    from FreiHAND.train_reconstruction_baseline import build_reconstruction_model
except ModuleNotFoundError as error:
    if error.name != "tensorflow":
        raise
    tf = None
    build_reconstruction_model = None


@unittest.skipIf(tf is None, "TensorFlow is optional and is not installed")
class ReconstructionModelTests(unittest.TestCase):
    def test_model_maps_single_occupancy_channel_to_ordered_vertices(self):
        model = build_reconstruction_model(
            input_shape=(8, 8, 8, 1), vertex_count=10
        )

        prediction = model.predict(
            np.zeros((2, 8, 8, 8, 1), dtype=np.float32), verbose=0
        )

        self.assertEqual(prediction.shape, (2, 10, 3))
        self.assertTrue(np.isfinite(prediction).all())

    def test_rejects_multiple_input_channels(self):
        with self.assertRaisesRegex(ValueError, "one channel"):
            build_reconstruction_model(input_shape=(8, 8, 8, 22))


if __name__ == "__main__":
    unittest.main()
