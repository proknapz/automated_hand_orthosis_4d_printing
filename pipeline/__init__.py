"""Photo → landmarks → depth → point cloud → parametric splint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Dict

from pipeline.depth import estimate_depth
from pipeline.exceptions import HandNotDetectedError, PipelineError
from pipeline.landmarks import extract_landmarks
from pipeline.pointcloud import build_point_cloud, export_ply
from pipeline.splint import export_stl, generate_splint

Side = Literal["dorsal", "volar"]


def run_pipeline(
    image_path: str,
    output_dir: str,
    hand_length_cm: float = 18.0,
    side: Side = "dorsal",
    finger_extension: bool = False,
) -> Dict[str, Any]:
    """Process one hand photo and write cloud.ply, splint.stl, and landmarks JSON."""
    if hand_length_cm <= 0:
        raise PipelineError("Hand length must be a positive number of centimetres.")
    if side not in ("dorsal", "volar"):
        raise PipelineError("Splint side must be 'dorsal' or 'volar'.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hand = extract_landmarks(image_path)
    depth = estimate_depth(hand.image_rgb)
    cloud = build_point_cloud(hand, depth, hand_length_cm=hand_length_cm)
    mesh = generate_splint(
        cloud["pcd"],
        cloud["landmarks_3d"],
        side=side,
        finger_extension=finger_extension,
    )

    ply_path = out / "cloud.ply"
    stl_path = out / "splint.stl"
    json_path = out / "landmarks.json"

    export_ply(cloud["pcd"], str(ply_path))
    export_stl(mesh, str(stl_path))

    payload = hand.as_json()
    payload["hand_length_cm"] = hand_length_cm
    payload["scale"] = float(cloud["scale"])
    payload["side"] = side
    payload["finger_extension"] = finger_extension
    payload["landmarks_3d"] = cloud["landmarks_3d"].tolist()
    payload["point_count"] = int(len(cloud["pcd"].points))
    payload["splint_faces"] = int(mesh.faces.shape[0])
    payload["splint_watertight"] = bool(mesh.is_watertight)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "ply_path": str(ply_path),
        "stl_path": str(stl_path),
        "landmarks_path": str(json_path),
        "meta": payload,
    }


__all__ = [
    "run_pipeline",
    "HandNotDetectedError",
    "PipelineError",
]
