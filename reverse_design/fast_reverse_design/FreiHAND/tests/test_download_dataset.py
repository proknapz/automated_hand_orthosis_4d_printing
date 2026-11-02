import tempfile
import unittest
import zipfile
from pathlib import Path

from FreiHAND.download_dataset import extract_archive, find_dataset_root


class DownloadDatasetTests(unittest.TestCase):
    def test_extracts_archive_and_finds_annotation_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "tiny.zip"
            output = root / "unpacked"
            with zipfile.ZipFile(str(archive), "w") as zipped:
                for filename in (
                    "dataset/training_K.json",
                    "dataset/training_mano.json",
                    "dataset/training_xyz.json",
                ):
                    zipped.writestr(filename, "[]")

            extract_archive(archive, output)

            self.assertEqual(find_dataset_root(output), output / "dataset")

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(str(archive), "w") as zipped:
                zipped.writestr("../outside.txt", "not allowed")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                extract_archive(archive, root / "unpacked")

    def test_requires_all_training_annotation_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training_K.json").write_text("[]", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                find_dataset_root(root)


if __name__ == "__main__":
    unittest.main()
