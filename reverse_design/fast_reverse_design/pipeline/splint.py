"""Parametric dorsal/volar splint shell from landmarks and a hand point cloud."""

from __future__ import annotations

from typing import Literal, Tuple, List

import numpy as np
import open3d as o3d
import trimesh
from scipy.interpolate import griddata

from pipeline.exceptions import PipelineError
from pipeline.landmarks import MCP_INDICES, PIP_INDICES, WRIST_INDEX

Side = Literal["dorsal", "volar"]

OFFSET_M = 0.003  # 3 mm clearance from the skin
THICKNESS_M = 0.002  # 2 mm shell
GRID_U = 28
GRID_V = 40


def _palm_frame(landmarks_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return origin, x (ulnar-radial), y (distal), n (out of dorsum, camera-facing)."""
    origin = landmarks_3d[WRIST_INDEX]
    index_mcp = landmarks_3d[5]
    pinky_mcp = landmarks_3d[17]
    middle_mcp = landmarks_3d[9]
    y = middle_mcp - origin
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-8:
        raise PipelineError("Invalid wrist-to-MCP vector.")
    y = y / y_norm
    across = index_mcp - pinky_mcp
    # x should be roughly perpendicular to distal axis.
    x = across - np.dot(across, y) * y
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        x = np.cross(y, np.array([0.0, 0.0, 1.0]))
        x_norm = np.linalg.norm(x)
    x = x / max(x_norm, 1e-8)
    n = np.cross(x, y)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        raise PipelineError("Could not build a palm frame from landmarks.")
    n = n / n_norm
    # Point the normal toward the camera so it matches the photographed surface.
    if np.dot(n, np.array([0.0, 0.0, -1.0])) < 0:
        n = -n
        x = -x
    return origin, x, y, n


def generate_splint(
    pcd: o3d.geometry.PointCloud,
    landmarks_3d: np.ndarray,
    side: Side = "dorsal",
    finger_extension: bool = False,
    offset_m: float = OFFSET_M,
    thickness_m: float = THICKNESS_M,
) -> trimesh.Trimesh:
    """Build a watertight splint shell covering wrist to finger bases (or PIPs)."""
    points = np.asarray(pcd.points)
    if points.shape[0] < 30:
        raise PipelineError("Not enough points to generate a splint.")

    origin, x_axis, y_axis, n_vis = _palm_frame(landmarks_3d)
    n = n_vis if side == "dorsal" else -n_vis

    hand_len = float(np.linalg.norm(landmarks_3d[12] - landmarks_3d[WRIST_INDEX]))
    distal_ids = PIP_INDICES if finger_extension else MCP_INDICES
    distal_pts = landmarks_3d[list(distal_ids)]
    y_min = -0.18 * hand_len  # slightly onto the forearm
    y_max = float(np.max((distal_pts - origin) @ y_axis)) + 0.02 * hand_len
    x_span = float(np.max(np.abs((landmarks_3d[[5, 17]] - origin) @ x_axis)))
    x_min, x_max = -x_span * 1.25, x_span * 1.25

    rel = points - origin
    u = rel @ x_axis
    v = rel @ y_axis
    h = rel @ n
    in_region = (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
    # Keep the photographed (outer) side of the palm plane.
    on_side = h > np.percentile(h[in_region], 35) if np.any(in_region) else h > 0
    sel = in_region & on_side
    if np.count_nonzero(sel) < 40:
        sel = in_region
    if np.count_nonzero(sel) < 40:
        raise PipelineError(
            "Not enough surface points on the requested splint side. "
            "Photograph the same side of the hand as the splint type."
        )

    uu = np.linspace(x_min, x_max, GRID_U)
    vv = np.linspace(y_min, y_max, GRID_V)
    grid_u, grid_v = np.meshgrid(uu, vv, indexing="xy")
    height = griddata(
        np.column_stack([u[sel], v[sel]]),
        h[sel],
        (grid_u, grid_v),
        method="linear",
    )
    fill = griddata(
        np.column_stack([u[sel], v[sel]]),
        h[sel],
        (grid_u, grid_v),
        method="nearest",
    )
    height = np.where(np.isfinite(height), height, fill)

    valid_vertex = np.isfinite(height)
    # Drop a margin so the shell does not reach the noisy silhouette.
    margin_u = max(1, GRID_U // 16)
    margin_v = max(1, GRID_V // 18)
    valid_vertex[:margin_v, :] = False
    valid_vertex[-margin_v:, :] = False
    valid_vertex[:, :margin_u] = False
    valid_vertex[:, -margin_u:] = False

    slot_mask = _strap_slot_mask(grid_u, grid_v, x_min, x_max, y_min, y_max)
    valid_vertex = valid_vertex & ~slot_mask

    outer = np.zeros((GRID_V, GRID_U, 3), dtype=np.float64)
    inner = np.zeros((GRID_V, GRID_U, 3), dtype=np.float64)
    for j in range(GRID_V):
        for i in range(GRID_U):
            base = origin + grid_u[j, i] * x_axis + grid_v[j, i] * y_axis + height[j, i] * n
            outer[j, i] = base + offset_m * n
            inner[j, i] = base + (offset_m - thickness_m) * n

    mesh = _shell_from_grids(outer, inner, valid_vertex)
    if mesh.faces.shape[0] < 20:
        raise PipelineError("Splint mesh is empty. Try a clearer hand photo.")
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    mesh.process(validate=True)
    return mesh


def _strap_slot_mask(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Two strap slots across the wrist and mid-hand."""
    width = x_max - x_min
    length = y_max - y_min
    slot_half_w = 0.22 * width
    slot_half_h = max(0.002, 0.035 * length)
    centers_v = (y_min + 0.22 * length, y_min + 0.55 * length)
    mask = np.zeros(grid_u.shape, dtype=bool)
    for cv in centers_v:
        mask |= (np.abs(grid_u) < slot_half_w) & (np.abs(grid_v - cv) < slot_half_h)
    return mask


def _shell_from_grids(
    outer: np.ndarray,
    inner: np.ndarray,
    valid: np.ndarray,
) -> trimesh.Trimesh:
    """Watertight shell: outer + inner surfaces and walls around valid quads."""
    nv, nu = valid.shape
    outer_idx = -np.ones((nv, nu), dtype=np.int32)
    inner_idx = -np.ones((nv, nu), dtype=np.int32)
    vertices: List[np.ndarray] = []

    def add(p: np.ndarray) -> int:
        vertices.append(p)
        return len(vertices) - 1

    for j in range(nv):
        for i in range(nu):
            if not valid[j, i]:
                continue
            outer_idx[j, i] = add(outer[j, i])
            inner_idx[j, i] = add(inner[j, i])

    faces: List[List[int]] = []

    def quad_ok(j: int, i: int) -> bool:
        return (
            i + 1 < nu
            and j + 1 < nv
            and valid[j, i]
            and valid[j, i + 1]
            and valid[j + 1, i]
            and valid[j + 1, i + 1]
        )

    for j in range(nv - 1):
        for i in range(nu - 1):
            if not quad_ok(j, i):
                continue
            a, b = outer_idx[j, i], outer_idx[j, i + 1]
            c, d = outer_idx[j + 1, i + 1], outer_idx[j + 1, i]
            faces.append([a, b, c])
            faces.append([a, c, d])
            ia, ib = inner_idx[j, i], inner_idx[j, i + 1]
            ic, id_ = inner_idx[j + 1, i + 1], inner_idx[j + 1, i]
            faces.append([ia, ic, ib])
            faces.append([ia, id_, ic])

    # Side walls on edges where a valid quad meets an invalid neighbor.
    def add_wall(o0: int, o1: int, i0: int, i1: int) -> None:
        faces.append([o0, o1, i1])
        faces.append([o0, i1, i0])

    for j in range(nv - 1):
        for i in range(nu - 1):
            is_q = quad_ok(j, i)
            # Right edge of this quad (between i+1)
            if is_q and not quad_ok(j, i + 1):
                add_wall(
                    outer_idx[j, i + 1],
                    outer_idx[j + 1, i + 1],
                    inner_idx[j, i + 1],
                    inner_idx[j + 1, i + 1],
                )
            # Left edge
            if is_q and (i == 0 or not quad_ok(j, i - 1)):
                add_wall(
                    outer_idx[j + 1, i],
                    outer_idx[j, i],
                    inner_idx[j + 1, i],
                    inner_idx[j, i],
                )
            # Top edge (j+1)
            if is_q and not quad_ok(j + 1, i):
                add_wall(
                    outer_idx[j + 1, i + 1],
                    outer_idx[j + 1, i],
                    inner_idx[j + 1, i + 1],
                    inner_idx[j + 1, i],
                )
            # Bottom edge
            if is_q and (j == 0 or not quad_ok(j - 1, i)):
                add_wall(
                    outer_idx[j, i],
                    outer_idx[j, i + 1],
                    inner_idx[j, i],
                    inner_idx[j, i + 1],
                )

    if not vertices or not faces:
        raise PipelineError("Failed to assemble splint faces.")
    return trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.array(faces), process=False)


def export_stl(mesh: trimesh.Trimesh, path: str) -> None:
    mesh.export(path, file_type="stl")
