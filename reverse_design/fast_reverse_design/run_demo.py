#!/usr/bin/env python3
"""Simple demo runner - works from any directory.

Run from anywhere:
    python run_demo.py
"""

import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import and run
from FreiHAND.demo_phase_4b_optimization import run_optimization_demo

if __name__ == '__main__':
    run_optimization_demo()
