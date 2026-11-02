"""Tests for MANO topology integration with FreiHAND geometry.

These tests verify MANO model loading, face topology validation,
and integration with the FreiHAND dataset loader.

Note: These tests require a valid MANO_RIGHT.pkl or MANO_LEFT.pkl model
available at the path specified by MANO_TEST_MODEL_PATH environment variable
or a default location. If the model is not available, tests are skipped.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from FreiHAND import mano_loader
from FreiHAND.data_loader import FreiHANDDataset, HandGeometry


class MANOLoaderTests(unittest.TestCase):
    """Test MANO model loading and validation."""

    def test_load_mano_faces_validates_file_exists(self):
        """Test that load_mano_faces raises FileNotFoundError for missing file."""
        nonexistent_path = Path("/nonexistent/MANO_RIGHT.pkl")
        with self.assertRaises(FileNotFoundError):
            mano_loader.load_mano_faces(nonexistent_path)

    def test_load_mano_faces_validates_pickle_format(self):
        """Test that load_mano_faces raises ValueError for invalid pickle."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"not a valid pickle file")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                mano_loader.load_mano_faces(temp_path)
        finally:
            temp_path.unlink()

    def test_load_mano_faces_validates_dictionary_format(self):
        """Test that load_mano_faces expects a dictionary with 'f' key."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Write a non-dict pickle
            pickle.dump([1, 2, 3], f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(TypeError):
                mano_loader.load_mano_faces(temp_path)
        finally:
            temp_path.unlink()

    def test_load_mano_faces_validates_f_key_exists(self):
        """Test that load_mano_faces raises ValueError if 'f' key missing."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Write a dict without 'f' key
            pickle.dump({"other_key": "value"}, f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                mano_loader.load_mano_faces(temp_path)
        finally:
            temp_path.unlink()

    def test_load_mano_faces_converts_to_int64(self):
        """Test that load_mano_faces returns int64 array."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Create valid face topology (1538 triangles, 778 vertices max)
            faces_array = np.random.randint(0, 778, size=(1538, 3), dtype=np.int32)
            pickle.dump({"f": faces_array}, f)
            temp_path = Path(f.name)

        try:
            faces = mano_loader.load_mano_faces(temp_path)
            self.assertEqual(faces.dtype, np.int64)
            self.assertEqual(faces.shape, (1538, 3))
        finally:
            temp_path.unlink()

    def test_load_mano_faces_validates_shape(self):
        """Test that load_mano_faces validates (F, 3) shape."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Write invalid face shape (should be F, 3)
            pickle.dump({"f": np.array([[1, 2], [3, 4]])}, f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                mano_loader.load_mano_faces(temp_path)
            self.assertIn("must have shape (F, 3)", str(ctx.exception))
        finally:
            temp_path.unlink()

    def test_load_mano_faces_validates_face_count(self):
        """Test that load_mano_faces validates expected face count (1538)."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Write wrong number of faces
            faces_array = np.random.randint(0, 778, size=(1000, 3), dtype=np.int32)
            pickle.dump({"f": faces_array}, f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                mano_loader.load_mano_faces(temp_path)
            self.assertIn("1000 faces", str(ctx.exception))
            self.assertIn("expected 1538", str(ctx.exception))
        finally:
            temp_path.unlink()

    def test_load_mano_faces_validates_vertex_index_range(self):
        """Test that load_mano_faces validates vertex indices [0, 777]."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Create faces with invalid vertex index (too high)
            faces_array = np.random.randint(0, 778, size=(1538, 3), dtype=np.int32)
            faces_array[0, 0] = 800  # Out of range
            pickle.dump({"f": faces_array}, f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                mano_loader.load_mano_faces(temp_path)
            self.assertIn("references vertex", str(ctx.exception))
        finally:
            temp_path.unlink()

    def test_load_mano_faces_rejects_negative_indices(self):
        """Test that load_mano_faces rejects negative vertex indices."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            faces_array = np.random.randint(0, 778, size=(1538, 3), dtype=np.int32)
            faces_array[0, 0] = -1
            pickle.dump({"f": faces_array}, f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                mano_loader.load_mano_faces(temp_path)
            self.assertIn("negative values", str(ctx.exception))
        finally:
            temp_path.unlink()


class ValidateFacesWithVerticesTests(unittest.TestCase):
    """Test face-to-vertices validation."""

    def test_validate_faces_with_vertices_accepts_valid_geometry(self):
        """Test that valid faces and vertices pass validation."""
        vertices = np.random.randn(778, 3)
        faces = np.array([[0, 1, 2], [1, 2, 3], [100, 101, 102]], dtype=np.int64)

        is_valid, msg = mano_loader.validate_faces_with_vertices(faces, vertices)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_validate_faces_with_vertices_rejects_invalid_face_shape(self):
        """Test that invalid face shape is rejected."""
        vertices = np.random.randn(778, 3)
        faces = np.array([[0, 1], [1, 2]])  # Wrong shape: should be (F, 3)

        is_valid, msg = mano_loader.validate_faces_with_vertices(faces, vertices)
        self.assertFalse(is_valid)
        self.assertIn("must have shape (F, 3)", msg)

    def test_validate_faces_with_vertices_rejects_out_of_range_indices(self):
        """Test that face indices outside vertex range are rejected."""
        vertices = np.random.randn(778, 3)
        faces = np.array([[0, 1, 2], [1, 2, 800]], dtype=np.int64)  # 800 is out of range

        is_valid, msg = mano_loader.validate_faces_with_vertices(faces, vertices)
        self.assertFalse(is_valid)
        self.assertIn("references vertex 800", msg)

    def test_validate_faces_with_vertices_rejects_negative_indices(self):
        """Test that negative indices are rejected."""
        vertices = np.random.randn(778, 3)
        faces = np.array([[0, 1, 2], [-1, 2, 3]], dtype=np.int64)

        is_valid, msg = mano_loader.validate_faces_with_vertices(faces, vertices)
        self.assertFalse(is_valid)
        self.assertIn("negative values", msg)

    def test_validate_faces_with_vertices_includes_sample_id_in_message(self):
        """Test that sample_id is included in error messages."""
        vertices = np.random.randn(100, 3)
        faces = np.array([[0, 1, 200]], dtype=np.int64)  # Out of range

        is_valid, msg = mano_loader.validate_faces_with_vertices(
            faces, vertices, sample_id="test_sample_123"
        )
        self.assertFalse(is_valid)
        self.assertIn("test_sample_123", msg)


class MANOStatusTests(unittest.TestCase):
    """Test status reporting."""

    def test_get_mano_status_returns_string(self):
        """Test that get_mano_status returns an informative string."""
        status = mano_loader.get_mano_status()
        self.assertIsInstance(status, str)
        self.assertIn("MANO", status)
        self.assertIn("https://mano.is.tue.mpg.de", status)


class MANOIntegrationMockTests(unittest.TestCase):
    """Integration tests using mocks (no real MANO model required)."""

    def create_mock_mano_pkl(self) -> Path:
        """Create a temporary mock MANO pkl file for testing."""
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Create valid face topology
            faces_array = np.random.randint(0, 778, size=(1538, 3), dtype=np.int32)
            pickle.dump({"f": faces_array}, f)
            return Path(f.name)

    def test_freihand_dataset_accepts_mano_path(self):
        """Test that FreiHANDDataset accepts mano_pkl_path parameter."""
        import tempfile
        import json

        mano_path = self.create_mock_mano_pkl()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                # Create minimal FreiHAND annotation files
                camera = [np.eye(3).tolist()]
                mano = [(np.arange(61, dtype=float).reshape(1, 61)).tolist()]
                keypoints = [(np.arange(21 * 3, dtype=float).reshape(21, 3)).tolist()]

                (root / "training_K.json").write_text(json.dumps(camera))
                (root / "training_mano.json").write_text(json.dumps(mano))
                (root / "training_xyz.json").write_text(json.dumps(keypoints))

                # Create dataset with MANO path
                dataset = FreiHANDDataset(
                    root=root,
                    split="training",
                    mano_pkl_path=mano_path,
                )

                # Verify dataset has MANO faces cached
                self.assertIsNotNone(dataset._mano_faces)
                self.assertEqual(dataset._mano_faces.shape, (1538, 3))
        finally:
            mano_path.unlink()

    def test_loaded_sample_includes_mano_faces(self):
        """Test that samples loaded with MANO path include faces."""
        import tempfile
        import json

        mano_path = self.create_mock_mano_pkl()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                # Create minimal FreiHAND annotation files with vertices
                camera = [np.eye(3).tolist()]
                mano = [(np.arange(61, dtype=float).reshape(1, 61)).tolist()]
                keypoints = [(np.arange(21 * 3, dtype=float).reshape(21, 3)).tolist()]
                vertices = [(np.random.randn(778, 3)).tolist()]

                (root / "training_K.json").write_text(json.dumps(camera))
                (root / "training_mano.json").write_text(json.dumps(mano))
                (root / "training_xyz.json").write_text(json.dumps(keypoints))
                (root / "training_verts.json").write_text(
                    json.dumps(vertices), encoding="utf-8"
                )

                dataset = FreiHANDDataset(
                    root=root,
                    split="training",
                    mano_pkl_path=mano_path,
                )

                sample = dataset.load_sample(0)

                # Verify sample includes faces
                self.assertIsNotNone(sample.faces)
                self.assertEqual(sample.faces.shape, (1538, 3))
                self.assertEqual(sample.faces.dtype, np.int64)
        finally:
            mano_path.unlink()


if __name__ == "__main__":
    unittest.main()
