"""Approximate two-view hand reconstruction and printable orthosis geometry.

The dorsal and palmar photographs are aligned by corresponding MediaPipe
landmarks and used to estimate a planar hand layout. They are not treated as a
calibrated stereo pair; thickness and ring sizes therefore remain estimates.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import trimesh

from pipeline.exceptions import PipelineError

LANDMARK_LABELS = (
    "wrist",
    "thumb CMC",
    "thumb MCP",
    "thumb IP",
    "thumb tip",
    "index MCP",
    "index PIP",
    "index DIP",
    "index tip",
    "middle MCP",
    "middle PIP",
    "middle DIP",
    "middle tip",
    "ring MCP",
    "ring PIP",
    "ring DIP",
    "ring tip",
    "pinky MCP",
    "pinky PIP",
    "pinky DIP",
    "pinky tip",
)

DIGITS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

_DEFAULT_CIRCUMFERENCE_RATIO = {
    "thumb": 0.255,
    "index": 0.285,
    "middle": 0.300,
    "ring": 0.280,
    "pinky": 0.235,
}


def canonicalize_landmarks(
    dorsal_points: Sequence[Sequence[float]],
    palmar_points: Sequence[Sequence[float]],
    hand_length_cm: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align the opposite-view 2D landmarks in a shared hand-local XY frame."""
    if not 8.0 <= hand_length_cm <= 28.0:
        raise PipelineError("Hand length must be between 8 and 28 centimetres.")
    dorsal = _landmark_array(dorsal_points, "dorsal")
    palmar = _landmark_array(palmar_points, "palm")
    target_length = hand_length_cm / 100.0
    dorsal_xy = _canonical_view(dorsal, target_length)
    palmar_xy = _canonical_view(palmar, target_length)
    points_xy = (dorsal_xy + palmar_xy) * 0.5
    if not np.isfinite(points_xy).all():
        raise PipelineError("Could not align the two hand landmark sets.")
    disagreement = np.linalg.norm(dorsal_xy - palmar_xy, axis=1)
    return points_xy, {
        "mean_view_disagreement_mm": float(np.mean(disagreement) * 1000.0),
        "max_view_disagreement_mm": float(np.max(disagreement) * 1000.0),
        "landmark_labels": list(LANDMARK_LABELS),
    }


def _landmark_array(points: Sequence[Sequence[float]], view_name: str) -> np.ndarray:
    try:
        result = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PipelineError("Landmark corrections must be numeric pixel coordinates.") from exc
    if result.shape != (21, 2) or not np.isfinite(result).all():
        raise PipelineError(
            "The %s view must contain 21 finite landmark coordinate pairs." % view_name
        )
    return result


def _canonical_view(points: np.ndarray, hand_length_m: float) -> np.ndarray:
    wrist = points[0]
    distal = points[12] - wrist
    distal_length = float(np.linalg.norm(distal))
    if distal_length < 1.0:
        raise PipelineError("Wrist and middle fingertip landmarks are too close together.")
    y_axis = distal / distal_length
    across = points[5] - points[17]
    x_axis = across - np.dot(across, y_axis) * y_axis
    x_length = float(np.linalg.norm(x_axis))
    if x_length < 1.0:
        raise PipelineError("Could not establish the hand's width direction.")
    x_axis /= x_length
    delta = points - wrist
    pixels = np.column_stack((delta @ x_axis, delta @ y_axis))
    return pixels * (hand_length_m / distal_length)


def generate_two_view_orthosis(
    dorsal_points: Sequence[Sequence[float]],
    palmar_points: Sequence[Sequence[float]],
    output_dir: str,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create hand preview, labeled landmark metadata, and modular STL meshes."""
    hand_length_cm = _number(params, "hand_length_cm", 18.0, 8.0, 28.0)
    points_xy, alignment = canonicalize_landmarks(
        dorsal_points, palmar_points, hand_length_cm
    )
    hand_length = hand_length_cm / 100.0
    measured_palm_breadth = float(np.linalg.norm(points_xy[5] - points_xy[17]))
    palm_breadth = _optional_cm(params, "palm_breadth_cm", measured_palm_breadth)
    if not 0.025 <= palm_breadth <= 0.15:
        raise PipelineError("Palm breadth must be between 2.5 and 15 centimetres.")
    wrist_width = _optional_cm(params, "wrist_width_cm", palm_breadth * 0.62)
    if not 0.02 <= wrist_width <= 0.12:
        raise PipelineError("Wrist width must be between 2 and 12 centimetres.")

    palm_thickness = _number(params, "palm_thickness_mm", 18.0, 6.0, 50.0) / 1000.0
    plate_thickness = _number(params, "plate_thickness_mm", 3.0, 1.0, 10.0) / 1000.0
    clearance = _number(params, "ring_clearance_mm", 2.0, 0.0, 8.0) / 1000.0
    band_width = _number(params, "ring_band_width_mm", 6.0, 2.0, 20.0) / 1000.0
    wall = _number(params, "ring_wall_mm", 2.0, 1.0, 8.0) / 1000.0
    include_thumb = _boolean(params, "include_thumb", True)

    circumferences = _finger_circumferences(params, hand_length)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    hand_mesh = _build_hand_preview(
        points_xy, palm_breadth, palm_thickness, circumferences
    )
    hand_path = output / "hand_preview.ply"
    hand_mesh.export(str(hand_path), file_type="ply")

    palm_mesh = _palm_plate(
        points_xy,
        palm_breadth,
        wrist_width,
        palm_thickness,
        plate_thickness,
        clearance,
    )
    part_meshes: Dict[str, trimesh.Trimesh] = {"palm_plate": palm_mesh}
    digit_items = list(DIGITS.items())
    if not include_thumb:
        digit_items = [(name, ids) for name, ids in digit_items if name != "thumb"]
    ring_locations = {"thumb": (1, 2), "index": (0, 1, 2), "middle": (0, 1, 2), "ring": (0, 1, 2), "pinky": (0, 1, 2)}
    scale_factors = {"thumb": (1.0, 0.88), "index": (1.0, 0.86, 0.72), "middle": (1.0, 0.86, 0.72), "ring": (1.0, 0.86, 0.72), "pinky": (1.0, 0.86, 0.72)}

    for digit_name, ids in digit_items:
        parts: List[trimesh.Trimesh] = []
        digit_points = points_xy[list(ids)]
        radii_z: List[float] = []
        for ordinal, landmark_offset in enumerate(ring_locations[digit_name]):
            landmark_id = ids[landmark_offset]
            before = ids[max(0, landmark_offset - 1)]
            after = ids[min(3, landmark_offset + 1)]
            axis = points_xy[after] - points_xy[before]
            circumference = circumferences[digit_name] * scale_factors[digit_name][ordinal]
            ring, vertical_radius = _closed_ring(
                points_xy[landmark_id],
                axis,
                circumference,
                palm_thickness,
                clearance,
                band_width,
                wall,
            )
            parts.append(ring)
            if digit_name == "thumb":
                joint_name = "mcp" if landmark_offset == 1 else "ip"
            else:
                joint_name = ("mcp", "pip", "dip")[landmark_offset]
            part_meshes["%s_ring_%s" % (digit_name, joint_name)] = ring
            radii_z.append(vertical_radius)

        # Embed the dorsal rail slightly into each collar crown so these
        # separate closed shells overlap in the exported assembly mesh.
        rail_height = max(radii_z) + 0.001
        first_id = ids[0]
        if digit_name == "thumb":
            root = points_xy[1]
        else:
            root = points_xy[0] * 0.20 + points_xy[first_id] * 0.80
        first_ring_point = points_xy[first_id]
        connector = _bar(
                np.array([root[0], root[1] - 0.004, palm_thickness * 0.5 + plate_thickness]),
                np.array([first_ring_point[0], first_ring_point[1], rail_height]),
                0.0015,
            )
        parts.append(connector)
        part_meshes["%s_palm_link" % digit_name] = connector
        if digit_name == "thumb":
            # The thumb CMC sits outside the palm plate outline. Add a diagonal
            # web bridge from the nearer MCP edge of the plate to the thumb rail.
            anchor_id = min((5, 17), key=lambda index: float(np.linalg.norm(points_xy[index] - root)))
            plate_top = palm_thickness * 0.5 + clearance + plate_thickness
            bridge = _bar(
                np.array([points_xy[anchor_id, 0], points_xy[anchor_id, 1], plate_top - 0.0005]),
                np.array([root[0], root[1] - 0.004, rail_height]),
                0.002,
            )
            parts.append(bridge)
            part_meshes["thumb_palm_bridge"] = bridge
        for link_index, (start, end) in enumerate(zip(ids, ids[1:]), start=1):
            link = _bar(
                    np.array([points_xy[start, 0], points_xy[start, 1], rail_height]),
                    np.array([points_xy[end, 0], points_xy[end, 1], rail_height]),
                    0.0015,
                )
            parts.append(link)
            part_meshes["%s_link_%d" % (digit_name, link_index)] = link

    assembled_components = []
    # STL stores triangle soup without component IDs. Tiny per-component
    # offsets prevent exact coincident vertices between overlapping closed
    # shells from being welded into false non-manifold edges by STL readers.
    for component_index, mesh in enumerate(part_meshes.values()):
        component = mesh.copy()
        component.apply_translation([component_index * 1e-6, 0.0, 0.0])
        assembled_components.append(component)
    assembled = trimesh.util.concatenate(assembled_components)
    # Preserve individually closed part shells. Processing the assembled mesh
    # merges coincident vertices across touching parts and can create false
    # non-manifold edges at intentional part intersections.
    _validate_mesh(assembled, require_watertight=False)
    stl_path = output / "orthosis.stl"
    # The model is calculated in metres, but STL has no unit field and slicers
    # conventionally interpret coordinates as millimetres. Export scaled copies
    # so an 18 cm hand opens at approximately 180 mm in a slicer.
    assembled_mm = assembled.copy()
    assembled_mm.apply_scale(1000.0)
    assembled_mm.export(str(stl_path), file_type="stl")
    exported_assembly = trimesh.load(str(stl_path), force="mesh")
    component_files = []
    for name, mesh in part_meshes.items():
        trimesh.repair.fix_normals(mesh)
        _validate_mesh(mesh)
        part_path = output / (name + ".stl")
        part_mm = mesh.copy()
        part_mm.apply_scale(1000.0)
        part_mm.export(str(part_path), file_type="stl")
        exported_part = trimesh.load(str(part_path), force="mesh")
        _validate_mesh(exported_part)
        component_files.append(part_path)
    parts_zip_path = output / "orthosis_parts.zip"
    with zipfile.ZipFile(parts_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for part_path in component_files:
            archive.write(part_path, arcname=part_path.name)

    landmark_3d = np.column_stack((points_xy, np.zeros(21, dtype=np.float64)))
    payload = {
        "landmark_labels": list(LANDMARK_LABELS),
        "landmarks_3d_m": landmark_3d.tolist(),
        "hand_length_cm": hand_length_cm,
        "hand_side": str(params.get("hand_side", "auto")),
        "palm_breadth_cm": palm_breadth * 100.0,
        "wrist_width_cm": wrist_width * 100.0,
        "palm_thickness_mm": palm_thickness * 1000.0,
        "finger_circumferences_mm": circumferences,
        "alignment": alignment,
        "assumptions": [
            "Dorsal and palmar images are aligned by semantic landmarks, not stereo calibration.",
            "Hand depth and finger cross-sections are estimated from optional measurements and defaults.",
            "Landmark depths are shown on the hand mid-plane in the preview.",
        ],
        "mesh": {
            "stl_units": "mm",
            "vertices": int(len(assembled.vertices)),
            "faces": int(len(assembled.faces)),
            "watertight": bool(exported_assembly.is_watertight),
            "component_shells_watertight": all(mesh.is_watertight for mesh in part_meshes.values()),
            "closed_part_count": int(sum(bool(mesh.is_watertight) for mesh in part_meshes.values())),
            "winding_consistent": bool(assembled.is_winding_consistent),
            "components": int(len(assembled.split(only_watertight=False))),
            "bounds_m": assembled.bounds.tolist(),
            "bounds_mm": (assembled.bounds * 1000.0).tolist(),
        },
        "parts": [path.name for path in component_files],
        "parts_zip": parts_zip_path.name,
    }
    (output / "landmarks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "hand_preview_path": str(hand_path),
        "stl_path": str(stl_path),
        "landmarks_path": str(output / "landmarks.json"),
        "meta": payload,
    }


def _number(params: Mapping[str, Any], key: str, default: float, low: float, high: float) -> float:
    raw = params.get(key, default)
    if raw in (None, ""):
        raw = default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise PipelineError(key.replace("_", " ").capitalize() + " must be a number.") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise PipelineError(
            "%s must be between %g and %g." % (key.replace("_", " ").capitalize(), low, high)
        )
    return value


def _optional_cm(params: Mapping[str, Any], key: str, default: float) -> float:
    raw = params.get(key)
    if raw in (None, ""):
        return default
    try:
        value = float(raw) / 100.0
    except (TypeError, ValueError) as exc:
        raise PipelineError(key.replace("_", " ").capitalize() + " must be a number.") from exc
    if not math.isfinite(value):
        raise PipelineError(key.replace("_", " ").capitalize() + " must be finite.")
    return value


def _boolean(params: Mapping[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _finger_circumferences(params: Mapping[str, Any], hand_length: float) -> Dict[str, float]:
    supplied = params.get("finger_circumferences_mm", {})
    if isinstance(supplied, str):
        try:
            supplied = json.loads(supplied)
        except ValueError as exc:
            raise PipelineError("Finger circumferences must be valid JSON values.") from exc
    if not isinstance(supplied, Mapping):
        supplied = {}
    result = {}
    for digit, ratio in _DEFAULT_CIRCUMFERENCE_RATIO.items():
        raw = supplied.get(digit)
        if raw in (None, ""):
            raw = params.get("finger_%s_circumference_mm" % digit)
        value = hand_length * ratio * 1000.0 if raw in (None, "") else float(raw)
        if not math.isfinite(value) or not 15.0 <= value <= 100.0:
            raise PipelineError("%s finger circumference must be between 15 and 100 mm." % digit.capitalize())
        result[digit] = value
    return result


def _palm_plate(
    points: np.ndarray,
    palm_breadth: float,
    wrist_width: float,
    palm_thickness: float,
    plate_thickness: float,
    clearance: float,
) -> trimesh.Trimesh:
    wrist_y = float(points[0, 1])
    mcp_y = float(np.mean(points[[5, 9, 13, 17], 1]))
    palm_length = mcp_y - wrist_y
    if palm_length < 0.025:
        raise PipelineError("The detected palm is too short; check the wrist and MCP landmarks.")
    breadth = palm_breadth * 1.05
    wrist_half = wrist_width * 0.5
    palm_half = breadth * 0.5
    bottom_y = wrist_y - 0.010
    slot_width = min(0.004, wrist_width * 0.12)
    slot_length = min(0.025, palm_length * 0.32)
    slot_centers = (-wrist_half * 0.42, wrist_half * 0.42)
    outline_points = [(-wrist_half, bottom_y)]
    for center_x in slot_centers:
        outline_points.extend(
            (
                (center_x - slot_width * 0.5, bottom_y),
                (center_x - slot_width * 0.5, bottom_y + slot_length),
                (center_x + slot_width * 0.5, bottom_y + slot_length),
                (center_x + slot_width * 0.5, bottom_y),
            )
        )
    outline_points.extend(
        (
            (wrist_half, bottom_y),
            (palm_half * 0.88, wrist_y + palm_length * 0.20),
            (palm_half, mcp_y + 0.002),
            (-palm_half, mcp_y + 0.002),
            (-palm_half * 0.88, wrist_y + palm_length * 0.20),
        )
    )
    outline = np.asarray(outline_points, dtype=np.float64)
    z_bottom = palm_thickness * 0.5 + clearance
    return _extruded_prism(outline, z_bottom, z_bottom + plate_thickness)


def _extruded_prism(outline: np.ndarray, z_bottom: float, z_top: float) -> trimesh.Trimesh:
    """Extrude a simple polygon using ear clipping (supports wrist slot notches)."""
    if _signed_area(outline) < 0:
        outline = outline[::-1]
    count = len(outline)
    cap_triangles = _triangulate_polygon(outline)
    vertices = np.vstack(
        (
            np.column_stack((outline, np.full(count, z_bottom))),
            np.column_stack((outline, np.full(count, z_top))),
        )
    )
    faces: List[List[int]] = []
    for tri in cap_triangles:
        faces.append([tri[2], tri[1], tri[0]])
        faces.append([count + tri[0], count + tri[1], count + tri[2]])
    for i in range(count):
        j = (i + 1) % count
        faces.extend(
            (
                [i, j, count + j],
                [i, count + j, count + i],
            )
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _signed_area(points: np.ndarray) -> float:
    return float(0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))


def _triangulate_polygon(outline: np.ndarray) -> List[Tuple[int, int, int]]:
    """Triangulate a simple counter-clockwise polygon without extra packages."""
    remaining = list(range(len(outline)))
    triangles: List[Tuple[int, int, int]] = []
    while len(remaining) > 3:
        found_ear = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = outline[previous], outline[current], outline[following]
            cross = float(np.cross(b - a, c - b))
            if cross <= 1e-12:
                continue
            contains_vertex = False
            for candidate in remaining:
                if candidate in (previous, current, following):
                    continue
                point = outline[candidate]
                c1 = float(np.cross(b - a, point - a))
                c2 = float(np.cross(c - b, point - b))
                c3 = float(np.cross(a - c, point - c))
                if c1 >= -1e-12 and c2 >= -1e-12 and c3 >= -1e-12:
                    contains_vertex = True
                    break
            if contains_vertex:
                continue
            triangles.append((previous, current, following))
            del remaining[position]
            found_ear = True
            break
        if not found_ear:
            raise PipelineError("Could not triangulate the palm plate outline.")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _closed_ring(
    center_xy: np.ndarray,
    axis_xy: np.ndarray,
    circumference_mm: float,
    palm_thickness: float,
    clearance: float,
    band_width: float,
    wall: float,
) -> Tuple[trimesh.Trimesh, float]:
    """Create a watertight, fully closed annular finger band."""
    axis_length = float(np.linalg.norm(axis_xy))
    if axis_length < 1e-8:
        raise PipelineError("A finger joint has overlapping landmarks; move the points apart.")
    axis = np.array([axis_xy[0] / axis_length, axis_xy[1] / axis_length, 0.0])
    across = np.array([-axis[1], axis[0], 0.0])
    inner_rx = circumference_mm / (2.0 * math.pi) / 1000.0 + clearance
    inner_rz = min(inner_rx, max(0.004, palm_thickness * 0.38))
    outer_rx, outer_rz = inner_rx + wall, inner_rz + wall
    count = 32
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    center = np.array([center_xy[0], center_xy[1], 0.0])
    outer = [center + across * (outer_rx * math.cos(a)) + np.array([0.0, 0.0, outer_rz * math.sin(a)]) for a in angles]
    inner = [center + across * (inner_rx * math.cos(a)) + np.array([0.0, 0.0, inner_rz * math.sin(a)]) for a in angles]
    verts: List[np.ndarray] = []
    for axial in (-band_width * 0.5, band_width * 0.5):
        shift = axis * axial
        verts.extend([point + shift for point in outer])
        verts.extend([point + shift for point in inner])
    layer_size = 2 * count
    faces: List[List[int]] = []
    for i in range(count):
        following = (i + 1) % count
        # Outer and inner curved faces.
        faces.extend(
            (
                [i, following, layer_size + following],
                [i, layer_size + following, layer_size + i],
                [count + i, layer_size + count + following, count + following],
                [count + i, layer_size + count + i, layer_size + count + following],
            )
        )
        # Side walls across the band thickness at both axial ends.
        faces.extend(
            (
                [i, count + i, count + following],
                [i, count + following, following],
                [layer_size + i, layer_size + following, layer_size + count + following],
                [layer_size + i, layer_size + count + following, layer_size + count + i],
            )
        )
    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise PipelineError("Could not create a watertight closed finger ring mesh.")
    return mesh, outer_rz


def _bar(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise PipelineError("Could not create a finger support between adjacent landmarks.")
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction / length)
    transform[:3, 3] = (start + end) * 0.5
    mesh.apply_transform(transform)
    return mesh


def _build_hand_preview(
    points_xy: np.ndarray,
    palm_breadth: float,
    palm_thickness: float,
    circumferences: Mapping[str, float],
) -> trimesh.Trimesh:
    wrist_y = float(points_xy[0, 1])
    mcp_y = float(np.mean(points_xy[[5, 9, 13, 17], 1]))
    palm_length = max(0.03, mcp_y - wrist_y)
    meshes: List[trimesh.Trimesh] = []
    palm = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    palm.apply_scale([palm_breadth * 0.50, palm_length * 0.58, palm_thickness * 0.50])
    palm.apply_translation([0.0, wrist_y + palm_length * 0.48, 0.0])
    meshes.append(palm)
    wrist = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    wrist.apply_scale([palm_breadth * 0.28, palm_length * 0.17, palm_thickness * 0.45])
    wrist.apply_translation([points_xy[0, 0], wrist_y - palm_length * 0.08, 0.0])
    meshes.append(wrist)
    for digit_name, ids in DIGITS.items():
        radius = max(0.003, circumferences[digit_name] / (2.0 * math.pi) / 1000.0 * 0.90)
        for start_id, end_id in zip(ids, ids[1:]):
            start = np.array([points_xy[start_id, 0], points_xy[start_id, 1], 0.0])
            end = np.array([points_xy[end_id, 0], points_xy[end_id, 1], 0.0])
            meshes.append(_bar(start, end, radius))
            joint = trimesh.creation.icosphere(subdivisions=1, radius=radius)
            joint.apply_translation(end)
            meshes.append(joint)
    return trimesh.util.concatenate(meshes)


def _validate_mesh(mesh: trimesh.Trimesh, require_watertight: bool = True) -> None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise PipelineError("Generated mesh is empty.")
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.bounds).all():
        raise PipelineError("Generated mesh contains invalid coordinates.")
    if require_watertight and not mesh.is_watertight:
        raise PipelineError("Generated STL mesh is not watertight.")
    if not mesh.is_winding_consistent:
        raise PipelineError("Generated STL mesh has inconsistent face winding.")

