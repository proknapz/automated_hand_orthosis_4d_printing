import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(REPO_ROOT, 'model_weights_2D_ResNet')
WORKING_SCRIPT = os.path.join(MODEL_DIR, 'PEA_for_reverse_design.py')

print(f'REPO_ROOT={REPO_ROOT}')
print(f'MODEL_DIR={MODEL_DIR}')
print(f'WORKING_SCRIPT={WORKING_SCRIPT}')

os.chdir(MODEL_DIR)

subprocess.run([sys.executable, WORKING_SCRIPT], check=True)
