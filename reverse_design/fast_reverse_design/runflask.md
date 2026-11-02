python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install flask pillow numpy scipy opencv-python-headless mediapipe==0.10.11 transformers open3d trimesh pytest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
set FLASK_APP=app.py
flask run
Open http://127.0.0.1:5000


