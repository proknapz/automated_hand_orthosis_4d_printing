from __future__ import annotations

import numpy as np
import pytest
import trimesh

from pipeline.comparison import inner_ring_circumference_mm
from pipeline.exceptions import PipelineError
from pipeline.orthosis import _closed_ring


@pytest.mark.parametrize("band_width_mm", [6.0, 20.0])
def test_inner_ring_circumference_reads_inner_stl_loop(tmp_path, band_width_mm) -> None:
    ring, _ = _closed_ring(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.03]),
        52.0,
        0.018,
        0.002,
        band_width_mm / 1000.0,
        0.002,
    )
    path = tmp_path / "index_ring_mcp.stl"
    ring.apply_scale(1000.0)
    ring.export(path)

    measured = inner_ring_circumference_mm(path)

    assert measured == pytest.approx(54.2, abs=1.0)


def test_inner_ring_circumference_rejects_invalid_stl(tmp_path) -> None:
    path = tmp_path / "invalid.stl"
    path.write_text("not an STL", encoding="utf-8")
    with pytest.raises(PipelineError):
        inner_ring_circumference_mm(path)


def test_inner_ring_circumference_rejects_non_ring_mesh(tmp_path) -> None:
    path = tmp_path / "box.stl"
    trimesh.creation.box().export(path)
    with pytest.raises(PipelineError, match="valid mesh"):
        inner_ring_circumference_mm(path)
