#!/usr/bin/env python
"""Quick-start example for MANO topology integration with FreiHAND.

This script demonstrates how to load FreiHAND hand geometry with MANO
face topology for orthosis design workflows.

Before running:
1. Download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/
2. Update MANO_MODEL_PATH below with the correct path
3. Update FREIHAND_ROOT below with your FreiHAND dataset directory
"""

from pathlib import Path
import numpy as np

from FreiHAND.data_loader import FreiHANDDataset, load_geometry_npz
from FreiHAND.mano_loader import get_mano_status, validate_faces_with_vertices


# Configuration - UPDATE THESE PATHS
MANO_MODEL_PATH = Path(r"C:\path\to\MANO_RIGHT.pkl")  # ← Update this path
FREIHAND_ROOT = Path(r"C:\path\to\FreiHAND_pub_v2")   # ← Update this path


def example_1_freihand_with_mano():
    """Example 1: Load FreiHAND dataset with MANO topology."""
    print("=" * 70)
    print("Example 1: Load FreiHAND Dataset with MANO Topology")
    print("=" * 70)

    if not MANO_MODEL_PATH.is_file():
        print(f"❌ MANO model not found at {MANO_MODEL_PATH}")
        print(f"   {get_mano_status()}")
        return

    if not FREIHAND_ROOT.is_dir():
        print(f"❌ FreiHAND dataset not found at {FREIHAND_ROOT}")
        return

    try:
        # Create dataset with MANO topology
        print(f"\n📂 Loading dataset from {FREIHAND_ROOT}")
        print(f"📦 Loading MANO topology from {MANO_MODEL_PATH}")

        dataset = FreiHANDDataset(
            root=FREIHAND_ROOT,
            split="training",
            image_version="gs",
            units="m",
            mano_pkl_path=MANO_MODEL_PATH,  # ← Enable MANO topology
        )

        print(f"✓ Dataset ready\n")

        # Load a sample
        sample = dataset.load_sample(0)

        print(f"Sample ID: {sample.sample_id}")
        print(f"Vertices shape: {sample.vertices.shape}")  # (778, 3)
        print(f"Keypoints shape: {sample.keypoints.shape}")  # (21, 3)
        print(f"Faces shape: {sample.faces.shape}")  # (1538, 3) ← NOW INCLUDED!
        print(f"Units: {sample.units}")

        # Verify geometry
        is_valid, msg = validate_faces_with_vertices(
            sample.faces, sample.vertices, sample_id=sample.sample_id
        )
        if is_valid:
            print(f"✓ Geometry validation: PASSED")
        else:
            print(f"✗ Geometry validation: FAILED - {msg}")

        # Print metadata
        print(f"\nMetadata:")
        for key, value in sample.metadata.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_npz_with_mano():
    """Example 2: Load NPZ geometry file with MANO topology."""
    print("\n" + "=" * 70)
    print("Example 2: Load NPZ Geometry with MANO Topology")
    print("=" * 70)

    if not MANO_MODEL_PATH.is_file():
        print(f"❌ MANO model not found at {MANO_MODEL_PATH}")
        return

    # Create a sample NPZ file for demonstration
    sample_npz_path = Path("/tmp/sample_hand.npz")

    try:
        # Generate synthetic hand geometry
        print(f"\n🔨 Creating synthetic hand geometry for demonstration...")
        vertices = np.random.randn(778, 3)  # Random 778 vertices
        keypoints = np.random.randn(21, 3)  # Random 21 keypoints

        # Save to NPZ
        np.savez(sample_npz_path, vertices=vertices, keypoints=keypoints)
        print(f"   Saved to {sample_npz_path}")

        # Load with MANO topology
        print(f"\n📦 Loading MANO topology...")
        geometry = load_geometry_npz(
            path=sample_npz_path,
            sample_id="synthetic_hand",
            units="m",
            mano_pkl_path=MANO_MODEL_PATH,  # ← Enable MANO topology
        )

        print(f"✓ Geometry loaded")
        print(f"\nGeometry:")
        print(f"  Sample ID: {geometry.sample_id}")
        print(f"  Vertices shape: {geometry.vertices.shape}")
        print(f"  Faces shape: {geometry.faces.shape}")  # ← MANO topology added!
        print(f"  Units: {geometry.units}")

        # Verify
        is_valid, msg = validate_faces_with_vertices(
            geometry.faces, geometry.vertices, sample_id=geometry.sample_id
        )
        print(f"  Validation: {'✓ PASSED' if is_valid else f'✗ FAILED - {msg}'}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if sample_npz_path.is_file():
            sample_npz_path.unlink()


def example_3_mano_status():
    """Example 3: Check MANO integration status."""
    print("\n" + "=" * 70)
    print("Example 3: MANO Integration Status")
    print("=" * 70)

    status = get_mano_status()
    print(f"\n{status}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MANO Topology Integration - Quick Start Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    # Show current configuration
    print("\nCurrent Configuration:")
    print(f"  MANO Model:       {MANO_MODEL_PATH}")
    print(f"  FreiHAND Dataset: {FREIHAND_ROOT}")

    print("\n" + "-" * 70)

    # Run examples
    example_3_mano_status()
    example_1_freihand_with_mano()
    example_2_npz_with_mano()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
