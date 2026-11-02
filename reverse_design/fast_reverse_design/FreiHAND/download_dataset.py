"""Download and safely extract the official FreiHAND v2 dataset archives.

The archive is several gigabytes. This script uses only Python's standard
library and does not run automatically during import or tests.
"""

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence


TRAINING_URL = (
    "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
)
EVALUATION_URL = (
    "https://lmb.informatik.uni-freiburg.de/data/freihand/"
    "FreiHAND_pub_v2_eval.zip"
)
ANNOTATION_FILES = (
    "training_K.json",
    "training_mano.json",
    "training_xyz.json",
)
CHUNK_SIZE = 1024 * 1024


def download_archive(url: str, destination: Path, force: bool = False) -> Path:
    """Stream an archive to disk, retaining a complete file only on success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        print("Using existing archive: {}".format(destination))
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "FreiHAND-research-data-loader/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else None
            downloaded = 0
            with partial.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = min(100.0, downloaded * 100.0 / total)
                        print(
                            "\rDownloaded {:.1f}/{:.1f} MiB ({:.1f}%)".format(
                                downloaded / (1024.0 * 1024.0),
                                total / (1024.0 * 1024.0),
                                percent,
                            ),
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            "\rDownloaded {:.1f} MiB".format(
                                downloaded / (1024.0 * 1024.0)
                            ),
                            end="",
                            flush=True,
                        )
        if downloaded == 0:
            raise OSError("server returned an empty archive")
        partial.replace(destination)
        print("\nSaved archive: {}".format(destination))
        return destination
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def _safe_destination(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        relative.is_absolute()
        or any(part in ("..", "") for part in relative.parts)
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError("unsafe path in ZIP archive: {!r}".format(member_name))
    destination = (root / Path(*relative.parts)).resolve()
    root_resolved = root.resolve()
    try:
        contained = Path(os.path.commonpath((str(root_resolved), str(destination)))) == root_resolved
    except ValueError:
        contained = False
    if not contained:
        raise ValueError("ZIP member escapes extraction directory: {!r}".format(member_name))
    return destination


def extract_archive(archive: Path, destination: Path) -> Path:
    """Extract ZIP/ZIP64 entries after validating all output paths and sizes."""
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(str(destination)).free
    try:
        with zipfile.ZipFile(str(archive), "r") as zipped:
            entries = zipped.infolist()
            total_uncompressed = sum(item.file_size for item in entries)
            if total_uncompressed > free_bytes:
                raise OSError(
                    "not enough free space to extract archive: need at least "
                    "{:.2f} GiB, have {:.2f} GiB".format(
                        total_uncompressed / (1024.0 ** 3),
                        free_bytes / (1024.0 ** 3),
                    )
                )
            destinations = []
            for item in entries:
                target = _safe_destination(destination, item.filename)
                if item.file_size > 0 and item.compress_size == 0:
                    raise ValueError(
                        "invalid compressed size in ZIP entry {!r}".format(item.filename)
                    )
                destinations.append((item, target))

            for item, target in destinations:
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(item, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                print("Extracted {}".format(item.filename))
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("Could not extract {}: {}".format(archive, error)) from error
    return destination


def find_dataset_root(search_root: Path) -> Path:
    """Find the directory containing all three training annotation files."""
    candidates = [path.parent for path in search_root.rglob(ANNOTATION_FILES[0])]
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in ANNOTATION_FILES):
            return candidate
    raise FileNotFoundError(
        "Could not find all training annotation files under {}: {}".format(
            search_root, ", ".join(ANNOTATION_FILES)
        )
    )


def prepare_dataset(
    data_dir: Path,
    include_evaluation: bool = False,
    force_download: bool = False,
) -> Path:
    """Download/extract required training archive and optionally evaluation archive."""
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_root = find_dataset_root(data_dir)
        print("Training annotations already extracted: {}".format(dataset_root))
    except FileNotFoundError:
        training_archive = download_archive(
            TRAINING_URL,
            data_dir / "FreiHAND_pub_v2.zip",
            force=force_download,
        )
        extract_archive(training_archive, data_dir)
        dataset_root = find_dataset_root(data_dir)

    if include_evaluation:
        evaluation_archive = download_archive(
            EVALUATION_URL,
            data_dir / "FreiHAND_pub_v2_eval.zip",
            force=force_download,
        )
        extract_archive(evaluation_archive, data_dir)

    print("Dataset root: {}".format(dataset_root))
    print("Training annotations verified: {}".format(", ".join(ANNOTATION_FILES)))
    return dataset_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Datasets" / "FreiHAND",
        help="Download/extraction directory (default: ~/Datasets/FreiHAND)",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Also download and extract the optional evaluation-annotations archive",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download archive files again even if they already exist",
    )
    args = parser.parse_args(argv)

    try:
        prepare_dataset(
            args.data_dir,
            include_evaluation=args.include_eval,
            force_download=args.force_download,
        )
    except (OSError, ValueError, urllib.error.URLError) as error:
        print("Dataset preparation failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
