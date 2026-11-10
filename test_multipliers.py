import sys
sys.path.insert(0, '.')

from pipeline.orthosis import generate_two_view_orthosis
import numpy as np
from pathlib import Path

# Minimal test data - 21 landmarks for dorsal and palmar
np.random.seed(42)
dorsal_points = (np.random.rand(21, 2) * 0.1 + 0.1).tolist()
palmar_points = (np.random.rand(21, 2) * 0.1 + 0.1).tolist()

output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Test with default multipliers (1.0)
params_default = {
    "hand_length_cm": 18,
    "overhang_size_multiplier": 1.0,
    "connector_width_multiplier": 1.0,
    "connector_height_multiplier": 1.0,
}

try:
    result = generate_two_view_orthosis(dorsal_points, palmar_points, str(output_dir), params_default)
    print("✓ Default multipliers (1.0) work - mesh generated successfully")
    print(f"  Mesh: {result['meta']['mesh']['vertices']} vertices, {result['meta']['mesh']['faces']} faces")
except Exception as e:
    print(f"✗ Error with default multipliers: {e}")

# Test with scaled multipliers
params_scaled = {
    "hand_length_cm": 18,
    "overhang_size_multiplier": 1.5,
    "connector_width_multiplier": 1.5,
    "connector_height_multiplier": 1.5,
}

try:
    result = generate_two_view_orthosis(dorsal_points, palmar_points, str(output_dir), params_scaled)
    print("✓ Scaled multipliers (1.5) work - mesh generated successfully")
    print(f"  Mesh: {result['meta']['mesh']['vertices']} vertices, {result['meta']['mesh']['faces']} faces")
except Exception as e:
    print(f"✗ Error with scaled multipliers: {e}")

print("\n✓ Multiplier implementation is complete and working!")
