import unittest
from pathlib import Path
from unittest.mock import patch

from FreiHAND.data_loader import HandGeometry
from FreiHAND.run_real_validation import run_real_validation
from FreiHAND.inspect_real_dataset import sample_indices


class RunRealValidationTests(unittest.TestCase):
    def test_real_inspection_indices_span_training_range(self):
        indices = sample_indices(32560, 50)

        self.assertEqual(len(indices), 50)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 32559)
        self.assertEqual(indices, sorted(set(indices)))

    @patch("FreiHAND.run_real_validation.validate_indices", return_value=[])
    @patch("FreiHAND.run_real_validation.visualize")
    @patch("FreiHAND.run_real_validation.FreiHANDDataset")
    @patch("FreiHAND.run_real_validation.find_dataset_root")
    def test_finds_root_visualizes_sample_and_runs_requested_checks(
        self, find_root, dataset_class, visualize, validate
    ):
        data_dir = Path("dataset-folder")
        dataset_root = Path("dataset-folder") / "nested"
        sample = HandGeometry(
            sample_id="training:00000005",
            keypoints=[[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]],
        )
        find_root.return_value = dataset_root
        dataset_class.return_value.load_sample.return_value = sample

        failures = run_real_validation(
            data_dir=data_dir,
            indices=[5, 10],
            sample_index=5,
            image_version="hom",
        )

        self.assertEqual(failures, [])
        find_root.assert_called_once_with(data_dir)
        dataset_class.assert_called_once_with(
            dataset_root, split="training", image_version="hom"
        )
        visualize.assert_called_once()
        self.assertEqual(visualize.call_args.args[0], sample)
        self.assertEqual(
            visualize.call_args.args[1],
            Path(__file__).parents[1]
            / "visualizations"
            / "01_training_00000005.png",
        )
        validate.assert_called_once_with(
            dataset_root, [5, 10], split="training", image_version="hom"
        )

    @patch("FreiHAND.run_real_validation.validate_indices", return_value=["FAIL 7"])
    @patch("FreiHAND.run_real_validation.visualize")
    @patch("FreiHAND.run_real_validation.FreiHANDDataset")
    @patch("FreiHAND.run_real_validation.find_dataset_root")
    def test_returns_validation_failures(
        self, find_root, dataset_class, visualize, validate
    ):
        find_root.return_value = Path("dataset-root")
        dataset_class.return_value.load_sample.return_value = HandGeometry(
            sample_id="training:00000007",
            keypoints=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )

        failures = run_real_validation(
            data_dir=Path("dataset-folder"),
            indices=[7],
            sample_index=7,
            output_path=Path("figure.png"),
        )

        self.assertEqual(failures, ["FAIL 7"])
        self.assertEqual(visualize.call_args.args[1], Path("figure.png"))


if __name__ == "__main__":
    unittest.main()
