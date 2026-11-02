"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = Path(__file__).resolve().parent / "data" / "sample_hand.png"


@pytest.fixture()
def sample_image() -> Path:
    return SAMPLE_IMAGE


@pytest.fixture()
def client():
    import sys

    sys.path.insert(0, str(ROOT))
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
