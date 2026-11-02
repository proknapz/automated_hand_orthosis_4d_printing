This is a project for the fast reverse design of 4D-printed voxelized composite structures.

# Photo-to-splint web demo

A **demonstration** Flask app: upload dorsal and palmar photos of a hand, review and correct detected landmarks, and generate an approximate hand preview plus modular orthosis STL geometry.

**This is a demonstration only. It is not medical-grade, not a medical device, and not for diagnosis, treatment, or clinical use.** Geometry comes from a single 2D photo plus an assumed hand length, so it is approximate.

## What it does

1. **Upload** matching JPG/PNG dorsal and palmar photos (up to 10 MB each), taken in the same open-hand pose.
2. **MediaPipe Hands** detects 21 landmarks in each view. The website overlays labels and lets users drag points to correct them.
3. The two views are aligned by corresponding landmarks and scaled using wrist-to-middle-fingertip hand length. They are not treated as a calibrated stereo pair; depth, finger cross-sections, and fit remain estimates.
4. The pipeline builds an approximate hand preview and modular orthosis geometry with closed finger bands at MCP/PIP/DIP locations, links, a palm/wrist plate, and thumb support bridged to the palm plate.
5. The viewer displays the estimated hand, orthosis, and 3D landmark labels. It exports an assembled `orthosis.stl`, individually closed part STL files, and a ZIP of the separate parts.

The STL is mesh geometry only and exports dimensions in millimetres for slicers. Material, actuator, sensor, and motion behavior are not modeled. Review the exported mesh and its displayed dimensions in a slicer before any print; this demo does not guarantee a fit.

## Project layout

```
app.py                 Flask routes (/analyze, /generate, legacy /upload)
pipeline/
  landmarks.py         MediaPipe landmarks for both views
  orthosis.py          Landmark alignment + modular STL generation
  depth.py             Depth Anything / MiDaS
  pointcloud.py        Back-project, scale, Open3D cleanup
  splint.py            Parametric shell → STL
static/                CSS + Three.js viewer
templates/index.html
tests/                 pytest + sample PNG
uploads/  outputs/     runtime files
```

## Requirements (Windows, CPU)

Use **64-bit Python 3.8 or newer**. The pinned MediaPipe 0.10.11 release has a Windows CPython 3.8 wheel; newer MediaPipe releases may use Python 3.9+ type annotation syntax. The first run may download `hand_landmarker.task` and the depth weights.

```bat
cd C:\Users\Dell\Documents\GitHub\Fast-reverse-design-of-4D-printing
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first depth run downloads **Depth Anything V2 Small** (Hugging Face). That needs a network connection and some disk space. Everything is intended to run on **CPU** (slow is OK).

If you only want this demo and not the rest of the research stack, install:

```bat
  pip install flask pillow numpy scipy opencv-python-headless mediapipe==0.10.11 transformers open3d trimesh pytest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

(`tensorflow` in `requirements.txt` is for the original reverse-design code, not this demo.)

## Run

```bat
.venv\Scripts\activate
set FLASK_APP=app.py
flask run
```

Open http://127.0.0.1:5000

PowerShell equivalent:

```powershell
$env:FLASK_APP = "app.py"
flask run
```

Or: `python app.py`

## Tests

```bat
pytest tests/test_pipeline.py -q
```

The sample image (`tests/data/sample_hand.png`) has **no hand**, so the test checks the clear **“No hand detected”** error. Splint/point-cloud tests use a synthetic 3D patch so they do not download depth models.

To try the two-view workflow: take well-lit dorsal (back of hand) and palmar (palm) photos of the same open hand, with fingers spread and all fingertips and wrist visible. The two images provide matching 2D landmarks; they do not constitute calibrated stereo, so generated depth and orthosis geometry are estimates. Review and adjust landmark positions before generation.

## API

`POST /analyze` (multipart) accepts `dorsal_photo` and `palmar_photo` (JPG or PNG, up to 10 MB each), plus optional `hand_length_cm` (default `18`) and `hand_side` (`auto`, `left`, or `right`). It returns a job ID and the detected landmarks for both photos. The browser lets the user review and correct the labeled landmarks.

`POST /generate` (JSON) accepts that `job_id`, optional corrected 2D landmark arrays for `dorsal` and `palmar`, and optional geometric parameters. Blank optional measurements are estimated from hand length. It returns the generated hand preview and orthosis files.

Geometric parameter keys include `palm_breadth_cm`, `wrist_width_cm`, `palm_thickness_mm`, `plate_thickness_mm`, `ring_clearance_mm`, `ring_band_width_mm`, `ring_wall_mm`, `include_thumb`, and optional per-finger circumferences in millimetres (`finger_thumb_circumference_mm` through `finger_pinky_circumference_mm`). Finger bands are closed loops, so they need to pass over the fingertips or be assembled from separate parts.

Example generation response:

```json
{
  "hand_preview_url": "/outputs/<id>/hand_preview.ply",
  "orthosis_url": "/outputs/<id>/orthosis.stl",
  "parts_zip_url": "/outputs/<id>/orthosis_parts.zip",
  "landmarks_url": "/outputs/<id>/landmarks.json"
}
```

The legacy single-photo `POST /upload` route remains available for the earlier point-cloud and splint workflow.
