# Automated Hand Orthosis from Photos

A Flask demonstration app that accepts dorsal and palmar hand photos, detects and displays labeled landmarks for review, and generates an estimated hand preview plus a modular orthosis STL.

> Demonstration only. This project is not a medical device and does not provide a clinically fitted orthosis. Two uncalibrated photos do not recover true hand depth; dimensions and fit remain estimates.

## Features

- Upload dorsal and palmar JPG/PNG images of the same open hand.
- Detect 21 MediaPipe landmarks per view and allow landmark correction in the browser.
- Set hand length and optional palm, wrist, hand-thickness, plate, and finger-circumference measurements.
- Preview the estimated hand and orthosis, with labeled landmarks.
- Export the assembled orthosis as STL, individual STL components, a ZIP of parts, and a PLY hand preview.
- Retain the legacy single-photo `/upload` workflow.

The STL is mesh geometry (millimetres for slicers), not editable parametric CAD. The rings are closed, and the thumb support is connected to the palm plate. Inspect dimensions and fit in a slicer before any prototype print.

## Windows setup

Use 64-bit Python 3.8 for the currently pinned MediaPipe 0.10.11 dependency.

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000/>. On the first hand analysis, the app downloads the MediaPipe `hand_landmarker.task` model into `pipeline/models/`; a network connection is needed for that initial download. The legacy depth pipeline may also download its model weights on first use.

For a temporary demo link, use a tunnel such as ngrok or zrok while Flask is running. Keep debug mode off when exposing the app, and stop the tunnel when the demo ends. A tunnel is not a permanent deployment.

## Tests

```powershell
python -m pytest -q
```

## Project structure

```text
app.py                  Flask application and routes
pipeline/               Landmark detection, legacy depth pipeline, orthosis geometry
static/                 Browser JavaScript and CSS
templates/              Flask HTML template
tests/                  Pipeline and two-view geometry tests
```

This app was extracted from [Fast-reverse-design-of-4D-printing](https://github.com/proknapz/Fast-reverse-design-of-4D-printing) into a focused repository.
