"""FreiHAND annotation and generic hand-geometry loading.

The official dataset files are not bundled with this repository. The
FreiHAND-specific reader follows the annotation layout in the official
``lmb-freiburg/freihand`` reference code and has been exercised against the
official training annotations. Generic ``.npz`` loading provides the stable
vertices/keypoints/faces handoff format for downstream geometry processing.

MANO topology integration: optionally loads face connectivity from MANO models
to provide complete surface topology for downstream processing.
"""

import json
import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import mano_loader


FLOAT_DTYPE = np.float64
FACE_DTYPE = np.int64
FREIHAND_KEYPOINT_COUNT = 21
TRAINING_SAMPLE_COUNT = 32560
TRAINING_IMAGE_VERSIONS = ("gs", "hom", "sample", "auto")
FREIHAND_UNITS = "m"
_JSON_SAMPLE_SEPARATOR = re.compile(rb"\]\]\s*,\s*\[")


@dataclass
class HandGeometry:
    """Generic hand geometry in one source coordinate frame.

    ``vertices`` and ``faces`` are optional. Training annotations include
    3D keypoints and a vertex array, but this reader does not provide face
    connectivity. A caller may supply faces from a version-compatible MANO
    model or another verified geometry source.
    """

    sample_id: str
    vertices: Optional[NDArray[np.float64]] = None
    keypoints: Optional[NDArray[np.float64]] = None
    faces: Optional[NDArray[np.int64]] = None
    units: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FreiHANDDataset:
    """Reader for aligned official-format FreiHAND annotation JSON files.

    The official evaluation code computes errors in the stored XYZ values and
    converts them to centimeters by multiplying by 100, so the coordinates are
    in meters. The caller may override this label for a nonstandard conversion.

    Optional MANO model integration provides face topology for complete surface
    representation. See mano_loader module for setup instructions.
    """

    root: Path
    split: str = "training"
    image_version: str = "gs"
    units: str = FREIHAND_UNITS
    mano_pkl_path: Optional[Path] = None
    _annotations: Optional[Tuple[List[Any], List[Any], List[Any]]] = field(
        default=None, init=False, repr=False
    )
    _vertex_offsets: Optional[List[Tuple[int, int]]] = field(
        default=None, init=False, repr=False
    )
    _scales: Optional[List[Any]] = field(default=None, init=False, repr=False)
    _mano_faces: Optional[NDArray[np.int64]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.split not in ("training", "evaluation"):
            raise ValueError("split must be 'training' or 'evaluation'")
        if self.image_version not in TRAINING_IMAGE_VERSIONS:
            raise ValueError(
                "image_version must be one of {}".format(TRAINING_IMAGE_VERSIONS)
            )
        if self.split == "evaluation" and self.image_version != "gs":
            raise ValueError("evaluation supports only the 'gs' image version")
        if self.mano_pkl_path is not None:
            self.mano_pkl_path = Path(self.mano_pkl_path)
            self._mano_faces = mano_loader.load_mano_faces(self.mano_pkl_path)

    def _load_annotations(self) -> Tuple[List[Any], List[Any], List[Any]]:
        if self._annotations is not None:
            return self._annotations

        containers = []
        for suffix in ("K", "mano", "xyz"):
            path = self.root / "{}_{}.json".format(self.split, suffix)
            if not path.is_file():
                raise FileNotFoundError(
                    "Missing FreiHAND annotation file: {}. "
                    "Download/unpack the official dataset or choose its root "
                    "directory.".format(path)
                )
            try:
                with path.open("r", encoding="utf-8") as annotation_file:
                    content = json.load(annotation_file)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Could not read FreiHAND annotation file {}: {}".format(path, error)
                ) from error
            if not isinstance(content, list):
                raise ValueError(
                    "FreiHAND annotation {} must contain a JSON list".format(path)
                )
            containers.append(content)

        lengths = tuple(len(container) for container in containers)
        if len(set(lengths)) != 1:
            raise ValueError(
                "FreiHAND annotation files have mismatched lengths "
                "(K, mano, xyz): {}".format(lengths)
            )
        self._annotations = (containers[0], containers[1], containers[2])
        return self._annotations

    @property
    def sample_count(self) -> int:
        return len(self._load_annotations()[0])

    def _load_scales(self, expected_count: int) -> Optional[List[Any]]:
        if self._scales is not None:
            return self._scales
        path = self.root / "{}_scale.json".format(self.split)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as scale_file:
                scales = json.load(scale_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Could not read FreiHAND scale file {}: {}".format(path, error)
            ) from error
        if not isinstance(scales, list) or len(scales) != expected_count:
            raise ValueError(
                "FreiHAND scale file {} must contain {} aligned values".format(
                    path, expected_count
                )
            )
        self._scales = scales
        return scales

    def _load_vertex_offsets(self, expected_count: int) -> List[Tuple[int, int]]:
        if self._vertex_offsets is not None:
            return self._vertex_offsets
        path = self.root / "{}_verts.json".format(self.split)
        if not path.is_file():
            raise FileNotFoundError(
                "Vertex annotations are unavailable: {}".format(path)
            )

        offsets: List[Tuple[int, int]] = []
        with path.open("rb") as vertex_file:
            if vertex_file.seek(0, 2) == 0:
                raise ValueError("Vertex annotation file is empty: {}".format(path))
            vertex_file.seek(0)
            with mmap.mmap(vertex_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                position = 0
                while position < len(mapped) and mapped[position] in b" \t\r\n":
                    position += 1
                if position >= len(mapped) or mapped[position] != ord("["):
                    raise ValueError("Vertex annotations must be a JSON array")
                item_start = position + 1
                for separator in _JSON_SAMPLE_SEPARATOR.finditer(mapped, item_start):
                    offsets.append((item_start, separator.start() + 2))
                    item_start = separator.end() - 1

                final_position = len(mapped) - 1
                while final_position >= 0 and mapped[final_position] in b" \t\r\n":
                    final_position -= 1
                if final_position < 0 or mapped[final_position] != ord("]"):
                    raise ValueError("Vertex annotations have no closing JSON array")
                offsets.append((item_start, final_position))

        if len(offsets) != expected_count:
            raise ValueError(
                "Vertex annotation file {} contains {} samples; expected {}".format(
                    path, len(offsets), expected_count
                )
            )
        self._vertex_offsets = offsets
        return offsets

    def _load_vertices(
        self, index: int, expected_count: int
    ) -> Optional[NDArray[np.float64]]:
        path = self.root / "{}_verts.json".format(self.split)
        if not path.is_file():
            return None
        offsets = self._load_vertex_offsets(expected_count)
        start, end = offsets[index]
        with path.open("rb") as vertex_file:
            with mmap.mmap(vertex_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                raw_sample = mapped[start:end]
        try:
            vertices = _finite_array(json.loads(raw_sample), "3D vertices")
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                "Could not parse vertex annotation at index {}: {}".format(index, error)
            ) from error
        if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
            raise ValueError(
                "3D vertices at index {} must have shape (V, 3), got {}".format(
                    index, vertices.shape
                )
            )
        return vertices

    def load_sample(self, index: int) -> HandGeometry:
        """Load the annotations for one unique sample index.

        This returns the 21 3D keypoints, optional training mesh vertices,
        optional reference-bone scale, camera intrinsic matrix, and MANO
        parameter row when they conform to the reference-code contract. Face
        topology is not included in the annotation JSON and is not fabricated.
        """
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("index must be an integer")
        annotations = self._load_annotations()
        if index < 0 or index >= len(annotations[0]):
            raise IndexError(
                "sample index {} is outside [0, {})".format(index, len(annotations[0]))
            )

        camera_list, mano_list, xyz_list = annotations
        camera = _finite_array(camera_list[index], "camera intrinsics")
        if camera.shape != (3, 3):
            raise ValueError(
                "camera intrinsics at index {} must have shape (3, 3), got {}".format(
                    index, camera.shape
                )
            )
        mano = _finite_array(mano_list[index], "MANO parameters")
        if mano.ndim != 2 or mano.shape[0] != 1 or mano.shape[1] < 61:
            raise ValueError(
                "MANO parameters at index {} must have shape (1, P), P >= 61, "
                "as required by the official split_theta helper; got {}".format(
                    index, mano.shape
                )
            )
        keypoints = _finite_array(xyz_list[index], "3D keypoints")
        if keypoints.shape != (FREIHAND_KEYPOINT_COUNT, 3):
            raise ValueError(
                "3D keypoints at index {} must have shape (21, 3) according "
                "to the official reference plotting code, got {}".format(
                    index, keypoints.shape
                )
            )

        if self.split == "training":
            version_index = TRAINING_IMAGE_VERSIONS.index(self.image_version)
            image_index = index + TRAINING_SAMPLE_COUNT * version_index
            image_path = PurePosixPath(self.split) / "rgb" / "{:08d}.jpg".format(image_index)
        else:
            image_index = index
            image_path = PurePosixPath(self.split) / "rgb" / "{:08d}.jpg".format(image_index)

        vertices = self._load_vertices(index, len(camera_list))
        scales = self._load_scales(len(camera_list))
        hand_scale = None if scales is None else _finite_array(scales[index], "hand scale")
        if hand_scale is not None and (
            hand_scale.size != 1 or float(hand_scale) <= 0.0
        ):
            raise ValueError("hand scale at index {} must be a positive scalar".format(index))

        faces = None
        geometry_note = "Vertices and keypoints are loaded from aligned annotations; face topology is not supplied by this reader."
        if self._mano_faces is not None:
            faces = self._mano_faces.copy()
            if vertices is not None:
                is_valid, validation_msg = mano_loader.validate_faces_with_vertices(
                    faces, vertices, sample_id=f"{self.split}:{index:08d}"
                )
                if not is_valid:
                    raise ValueError(validation_msg)
            geometry_note = (
                "Vertices and keypoints are loaded from aligned annotations; "
                "face topology is supplied from MANO model."
            )

        return HandGeometry(
            sample_id="{}:{:08d}".format(self.split, index),
            vertices=vertices,
            keypoints=keypoints,
            faces=faces,
            units=self.units,
            metadata={
                "annotation_index": int(index),
                "split": self.split,
                "image_version": self.image_version,
                "image_index": int(image_index),
                "image_relative_path": str(image_path),
                "camera_intrinsics": camera,
                "mano_parameters": mano,
                "hand_scale": None if hand_scale is None else float(hand_scale),
                "source_coordinate_frame_note": (
                    "Raw XYZ is inferred to use a camera-relative projection "
                    "frame from the official K projection convention; formal "
                    "axis and origin definitions are not stated in the cited "
                    "documentation. The current normalizer translates and "
                    "scales while preserving axis directions."
                ),
                "source_coordinate_units_note": (
                    "Official evaluation computes raw XYZ errors in stored "
                    "coordinates and reports centimeters by multiplying by 100; "
                    "the dataset coordinate unit is therefore meters. Normalized "
                    "XYZ is dimensionless; retain its transform."
                ),
                "geometry_note": geometry_note,
            },
        )


def _finite_array(value: Any, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=FLOAT_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be a numeric array".format(name)) from error
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def validate_geometry(geometry: HandGeometry) -> HandGeometry:
    """Validate and canonicalize a generic hand-geometry object."""
    if not isinstance(geometry.sample_id, str) or not geometry.sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(geometry.units, str) or not geometry.units.strip():
        raise ValueError("units must be a non-empty string; use 'unknown' if needed")

    vertices = (
        None
        if geometry.vertices is None
        else _finite_array(geometry.vertices, "vertices")
    )
    keypoints = (
        None
        if geometry.keypoints is None
        else _finite_array(geometry.keypoints, "keypoints")
    )
    if vertices is None and keypoints is None:
        raise ValueError("geometry must provide vertices, keypoints, or both")
    for name, points in (("vertices", vertices), ("keypoints", keypoints)):
        if points is not None and (
            points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3
        ):
            raise ValueError("{} must have non-empty shape (N, 3)".format(name))

    faces = None
    if geometry.faces is not None:
        if vertices is None:
            raise ValueError("faces cannot be validated without vertices")
        raw_faces = np.asarray(geometry.faces)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("faces must contain integer vertex indices")
        if np.any(raw_faces < 0) or np.any(raw_faces >= vertices.shape[0]):
            raise ValueError("faces contain an index outside the vertices array")
        faces = raw_faces.astype(FACE_DTYPE, copy=True)

    return HandGeometry(
        sample_id=geometry.sample_id,
        vertices=None if vertices is None else vertices.copy(),
        keypoints=None if keypoints is None else keypoints.copy(),
        faces=faces,
        units=geometry.units,
        metadata=dict(geometry.metadata),
    )


def load_geometry_npz(
    path: Path,
    sample_id: Optional[str] = None,
    units: str = "unknown",
    mano_pkl_path: Optional[Path] = None,
) -> HandGeometry:
    """Load the documented generic geometry interchange format from ``.npz``.

    Required/optional arrays are ``vertices`` (optional if keypoints exist),
    ``keypoints`` (optional if vertices exist), and ``faces`` (optional).
    Metadata is supplied by the caller; pickle-based object arrays are disabled.

    Optionally loads face topology from a MANO model if mano_pkl_path is provided.
    """
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError("Geometry archive does not exist: {}".format(archive_path))
    try:
        with np.load(str(archive_path), allow_pickle=False) as archive:
            allowed = {"vertices", "keypoints", "faces"}
            unexpected = set(archive.files) - allowed
            if unexpected:
                raise ValueError(
                    "Unsupported arrays in geometry archive: {}".format(
                        sorted(unexpected)
                    )
                )
            geometry_faces = archive["faces"] if "faces" in archive else None
            if geometry_faces is None and mano_pkl_path is not None:
                geometry_faces = mano_loader.load_mano_faces(mano_pkl_path)

            geometry = HandGeometry(
                sample_id=sample_id or archive_path.stem,
                vertices=archive["vertices"] if "vertices" in archive else None,
                keypoints=archive["keypoints"] if "keypoints" in archive else None,
                faces=geometry_faces,
                units=units,
                metadata={
                    "source_archive": str(archive_path),
                    "mano_model_source": str(mano_pkl_path) if mano_pkl_path else None,
                },
            )
    except (OSError, ValueError) as error:
        raise ValueError(
            "Could not load geometry archive {}: {}".format(archive_path, error)
        ) from error
    return validate_geometry(geometry)


def load_freihand_sample(
    dataset_root: Path,
    index: int,
    split: str = "training",
    image_version: str = "gs",
    units: str = FREIHAND_UNITS,
) -> HandGeometry:
    """Load one official-format sample, labeling XYZ units as meters by default."""
    return FreiHANDDataset(
        Path(dataset_root), split=split, image_version=image_version, units=units
    ).load_sample(index)
