"""Flask demo: upload a hand photo, preview the point cloud and splint."""

from __future__ import annotations

import uuid
import traceback
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

import numpy as np
from pipeline import HandNotDetectedError, PipelineError, run_pipeline
from pipeline.comparison import inner_ring_circumference_mm
from pipeline.landmarks import extract_landmarks
from pipeline.orthosis import generate_two_view_orthosis

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
ALLOWED_FORMATS = {"JPEG", "PNG"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 22 * 1024 * 1024
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def _validate_image(file: FileStorage) -> str:
    if file is None or not file.filename:
        raise PipelineError("Please choose a JPG or PNG photo.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise PipelineError("File type not allowed. Upload a JPG or PNG image.")
    data = file.read()
    file.seek(0)
    if len(data) > MAX_BYTES:
        raise PipelineError("File is too large. Maximum size is 10 MB.")
    if len(data) < 32:
        raise PipelineError("The uploaded file is empty or invalid.")
    try:
        with Image.open(file.stream) as img:
            fmt = (img.format or "").upper()
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise PipelineError("Could not read the image. Upload a valid JPG or PNG.") from exc
    file.stream.seek(0)
    if fmt not in ALLOWED_FORMATS:
        raise PipelineError("File type not allowed. Upload a JPG or PNG image.")
    return suffix if suffix != ".jpeg" else ".jpg"


@app.errorhandler(413)
def too_large(_err):
    return jsonify({"error": "File is too large. Maximum size is 10 MB."}), 413


@app.route("/")
def index() -> str:
    return render_template("index.html")


def _job_folder(root: Path, job_id: str) -> Path:
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise PipelineError("Invalid processing job.")
    return root / job_id


def _reference_measurements(form) -> tuple[dict[str, float], str]:
    measurements = {}
    for digit in ("thumb", "index", "middle", "ring", "pinky"):
        raw = form.get("reference_%s_mcp_circumference_mm" % digit, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise PipelineError(
                "%s MCP reference circumference must be a number in millimetres."
                % digit.capitalize()
            ) from exc
        if not 15.0 <= value <= 100.0:
            raise PipelineError(
                "%s MCP reference circumference must be between 15 and 100 mm."
                % digit.capitalize()
            )
        measurements[digit] = value
    source = form.get("reference_source", "").strip()
    if measurements and not source:
        raise PipelineError(
            "Describe the independent reference measurement source before saving measurements."
        )
    return measurements, source


def _load_job_manifest(job_id: str) -> dict:
    manifest_path = _job_folder(OUTPUT_DIR, job_id) / "job.json"
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError("A saved job record is unreadable.") from exc
    if not isinstance(data, dict):
        raise PipelineError("A saved job record has an invalid format.")
    return data


def _comparison_jobs() -> list[dict]:
    jobs = []
    if not OUTPUT_DIR.is_dir():
        return jobs
    for folder in OUTPUT_DIR.iterdir():
        if not folder.is_dir() or not JOB_ID_RE.fullmatch(folder.name):
            continue
        stl_path = folder / "orthosis.stl"
        if not stl_path.is_file():
            continue
        manifest = _load_job_manifest(folder.name)
        references = manifest.get("reference_measurements_mm", {})
        if not isinstance(references, dict):
            references = {}
        upload_folder = _job_folder(UPLOAD_DIR, folder.name)
        photos = []
        if upload_folder.is_dir():
            for name in ("dorsal.jpg", "dorsal.jpeg", "dorsal.png", "palmar.jpg", "palmar.jpeg", "palmar.png"):
                if (upload_folder / name).is_file():
                    photos.append({"label": name, "url": "/uploads/%s/%s" % (folder.name, name)})
        jobs.append(
            {
                "job_id": folder.name,
                "created_at": manifest.get(
                    "created_at",
                    datetime.fromtimestamp(stl_path.stat().st_mtime, timezone.utc).isoformat(),
                ),
                "orthosis_url": "/outputs/%s/orthosis.stl" % folder.name,
                "photos": photos,
                "reference_measurements_mm": references,
                "reference_source": str(manifest.get("reference_source", "")),
                "eligible": bool(references),
            }
        )
    return sorted(jobs, key=lambda job: job["created_at"], reverse=True)


def _delete_saved_job(job_id: str) -> None:
    output_folder = _job_folder(OUTPUT_DIR, job_id)
    if output_folder.is_symlink() or not output_folder.is_dir():
        raise PipelineError("Generated job not found.")
    if not (output_folder / "orthosis.stl").is_file():
        raise PipelineError("Generated job not found.")

    output_root = OUTPUT_DIR.resolve()
    upload_root = UPLOAD_DIR.resolve()
    if output_folder.resolve().parent != output_root:
        raise PipelineError("Generated job path is invalid.")

    upload_folder = _job_folder(UPLOAD_DIR, job_id)
    if upload_folder.is_symlink() or (
        upload_folder.exists() and upload_folder.resolve().parent != upload_root
    ):
        raise PipelineError("Uploaded photo path is invalid.")

    shutil.rmtree(output_folder)
    if upload_folder.is_dir():
        shutil.rmtree(upload_folder)


def _compare_selected_jobs(job_ids: list[str]) -> dict:
    if not job_ids or len(job_ids) > 100:
        raise PipelineError("Select between 1 and 100 generated jobs to compare.")
    if any(not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id) for job_id in job_ids):
        raise PipelineError("Invalid processing job.")
    if len(set(job_ids)) != len(job_ids):
        raise PipelineError("The selected job list contains duplicates.")
    results = []
    samples: dict[str, list[float]] = {}
    for job_id in job_ids:
        output_folder = _job_folder(OUTPUT_DIR, job_id)
        stl_path = output_folder / "orthosis.stl"
        if not stl_path.is_file():
            raise PipelineError("A selected job is missing its assembled STL.")
        manifest = _load_job_manifest(job_id)
        references = manifest.get("reference_measurements_mm", {})
        if not isinstance(references, dict) or not references:
            results.append(
                {
                    "job_id": job_id,
                    "eligible": False,
                    "reason": "No independent reference circumferences were recorded.",
                    "measurements": [],
                }
            )
            continue

        measured = []
        missing = []
        for digit, reference in references.items():
            if digit not in {"thumb", "index", "middle", "ring", "pinky"}:
                continue
            ring_path = output_folder / ("%s_ring_mcp.stl" % digit)
            if not ring_path.is_file():
                missing.append("%s MCP ring STL is missing" % digit)
                continue
            try:
                actual = inner_ring_circumference_mm(ring_path)
            except PipelineError as exc:
                missing.append("%s ring could not be measured: %s" % (digit, exc))
                continue
            try:
                reference_value = float(reference)
            except (TypeError, ValueError):
                missing.append("%s reference circumference is invalid" % digit)
                continue
            if not 15.0 <= reference_value <= 100.0:
                missing.append("%s reference circumference is outside 15–100 mm" % digit)
                continue
            error = actual - reference_value
            sizing_inputs = manifest.get("ring_sizing_inputs_mm", {})
            sizing_value = sizing_inputs.get(digit) if isinstance(sizing_inputs, dict) else None
            try:
                sizing_value = float(sizing_value) if sizing_value is not None else None
            except (TypeError, ValueError):
                sizing_value = None
            measured.append(
                {
                    "digit": digit,
                    "location": "MCP",
                    "stl_inner_circumference_mm": actual,
                    "reference_circumference_mm": reference_value,
                    "signed_error_mm": error,
                    "absolute_error_mm": abs(error),
                    "reference_also_used_for_sizing": (
                        sizing_value is not None
                        and np.isfinite(sizing_value)
                        and abs(float(sizing_value) - reference_value) < 1e-6
                    ),
                }
            )
            samples.setdefault(digit, []).append(error)
        results.append(
            {
                "job_id": job_id,
                "eligible": bool(measured),
                "reference_source": manifest.get("reference_source", ""),
                "measurements": measured,
                "reason": "; ".join(missing) if missing else None,
            }
        )

    statistics = {}
    for digit, errors in samples.items():
        values = np.asarray(errors, dtype=np.float64)
        statistics[digit] = {
            "count": int(values.size),
            "mae_mm": float(np.mean(np.abs(values))),
            "rmse_mm": float(np.sqrt(np.mean(np.square(values)))),
            "bias_mm": float(np.mean(values)),
        }
    return {
        "results": results,
        "statistics": statistics,
        "eligible_job_count": sum(result["eligible"] for result in results),
        "comparison_note": (
            "Geometric comparison only: each STL inner MCP ring circumference is compared "
            "with its independently recorded reference circumference. It does not establish "
            "clinical fit, comfort, safety, or effectiveness."
        ),
    }


@app.route("/analyze", methods=["POST"])
def analyze_pair():
    """Save paired views and return reviewable, editable 2D landmarks."""
    _ensure_dirs()
    try:
        dorsal_file = request.files.get("dorsal_photo")
        palmar_file = request.files.get("palmar_photo")
        dorsal_suffix = _validate_image(dorsal_file)
        palmar_suffix = _validate_image(palmar_file)
        hand_length_cm = float(request.form.get("hand_length_cm", 18))
        if not 8.0 <= hand_length_cm <= 28.0:
            raise PipelineError("Hand length must be between 8 and 28 centimetres.")
        selected_hand_side = request.form.get("hand_side", "auto")
        if selected_hand_side not in {"auto", "left", "right"}:
            raise PipelineError("Hand side must be Auto-detect, Left, or Right.")

        job_id = uuid.uuid4().hex
        upload_folder = _job_folder(UPLOAD_DIR, job_id)
        upload_folder.mkdir(parents=True, exist_ok=False)
        dorsal_path = upload_folder / ("dorsal" + dorsal_suffix)
        palmar_path = upload_folder / ("palmar" + palmar_suffix)
        dorsal_file.save(dorsal_path)
        palmar_file.save(palmar_path)

        dorsal_hand = extract_landmarks(str(dorsal_path))
        palmar_hand = extract_landmarks(str(palmar_path))
        view_mismatch = (
            dorsal_hand.handedness != "Unknown"
            and palmar_hand.handedness != "Unknown"
            and dorsal_hand.handedness != palmar_hand.handedness
        )
        selected_mismatch = (
            selected_hand_side != "auto"
            and dorsal_hand.handedness.lower() != selected_hand_side
        )
        handedness_warning = None
        if view_mismatch:
            handedness_warning = (
                "MediaPipe classified the two views as different hands. Confirm they show "
                "the same hand and pose, and review the landmark alignment."
            )
        elif selected_mismatch:
            handedness_warning = (
                "The dorsal photo's detected hand side differs from your selection. "
                "Check the hand-side selection and landmarks."
            )
        return jsonify(
            {
                "job_id": job_id,
                "hand_length_cm": hand_length_cm,
                "selected_hand_side": selected_hand_side,
                "dorsal_image_url": f"/uploads/{job_id}/{dorsal_path.name}",
                "palmar_image_url": f"/uploads/{job_id}/{palmar_path.name}",
                "dorsal": dorsal_hand.as_json(),
                "palmar": palmar_hand.as_json(),
                "handedness_warning": handedness_warning,
                "message": (
                    "Landmarks detected. Review or drag the points before generating the mesh."
                ),
            }
        )
    except HandNotDetectedError as exc:
        return jsonify({"error": str(exc)}), 400
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError:
        return jsonify({"error": "Hand length must be a number in centimetres."}), 400
    except Exception as exc:
        try:
            app.logger.exception("Unexpected two-view landmark analysis failure")
        except OSError:
            pass
        return jsonify({"error": "Could not analyze the photos: %s" % exc}), 500


@app.route("/generate", methods=["POST"])
def generate_pair():
    """Generate the estimated hand preview and modular orthosis STL files."""
    try:
        job_id = request.form.get("job_id", "")
        upload_folder = _job_folder(UPLOAD_DIR, job_id)
        output_folder = _job_folder(OUTPUT_DIR, job_id)
        if not upload_folder.is_dir():
            raise PipelineError("This photo review has expired. Please upload both photos again.")
        dorsal_path = next(upload_folder.glob("dorsal.*"), None)
        palmar_path = next(upload_folder.glob("palmar.*"), None)
        if dorsal_path is None or palmar_path is None:
            raise PipelineError("Both dorsal and palm photos are required.")
        try:
            corrections = json.loads(request.form.get("corrections", "{}"))
        except ValueError as exc:
            raise PipelineError("Landmark corrections are invalid JSON.") from exc
        if not isinstance(corrections, dict):
            raise PipelineError("Landmark corrections have an invalid format.")
        reference_measurements, reference_source = _reference_measurements(request.form)
        params = {
            "hand_length_cm": request.form.get("hand_length_cm", 18),
            "hand_side": request.form.get("hand_side", "auto"),
            "palm_breadth_cm": request.form.get("palm_breadth_cm"),
            "wrist_width_cm": request.form.get("wrist_width_cm"),
            "palm_thickness_mm": request.form.get("palm_thickness_mm", 18),
            "plate_thickness_mm": request.form.get("plate_thickness_mm", 3),
            "ring_clearance_mm": request.form.get("ring_clearance_mm", 2),
            "ring_band_width_mm": request.form.get("ring_band_width_mm", 6),
            "ring_wall_mm": request.form.get("ring_wall_mm", 2),
            "include_thumb": request.form.get("include_thumb", "true"),
            "overhang_size_multiplier": request.form.get("overhang_size_multiplier", 1.0),
            "connector_width_multiplier": request.form.get("connector_width_multiplier", 1.0),
            "connector_height_multiplier": request.form.get("connector_height_multiplier", 1.0),
            "finger_circumferences_mm": {
                digit: request.form.get("finger_%s_circumference_mm" % digit)
                for digit in ("thumb", "index", "middle", "ring", "pinky")
            },
        }
        hand_length_cm = float(params["hand_length_cm"])
        result = generate_two_view_orthosis(
            corrections.get("dorsal_points", []),
            corrections.get("palmar_points", []),
            str(output_folder),
            params,
        )
        manifest = {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "photos": [dorsal_path.name, palmar_path.name],
            "reference_measurements_mm": reference_measurements,
            "reference_source": reference_source,
            "design_measurements_mm": result["meta"]["finger_circumferences_mm"],
            "ring_sizing_inputs_mm": {
                digit: float(value)
                for digit, value in params["finger_circumferences_mm"].items()
                if value not in (None, "")
            },
            "stl_units": "mm",
        }
        manifest_path = output_folder / "job.json"
        temporary_manifest = output_folder / "job.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
        return jsonify(
            {
                "job_id": job_id,
                "hand_preview_url": f"/outputs/{job_id}/hand_preview.ply",
                "orthosis_url": f"/outputs/{job_id}/orthosis.stl",
                "landmarks_url": f"/outputs/{job_id}/landmarks.json",
                "part_urls": [
                    f"/outputs/{job_id}/{name}" for name in result["meta"]["parts"]
                ],
                "parts_zip_url": f"/outputs/{job_id}/orthosis_parts.zip",
                "meta": result["meta"],
                "hand_length_cm": hand_length_cm,
            }
        )
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError:
        return jsonify({"error": "Measurement parameters must be numeric."}), 400
    except Exception as exc:
        error_trace = traceback.format_exc()
        try:
            app.logger.error("Unexpected orthosis generation failure\n%s", error_trace)
        except OSError:
            pass
        return jsonify(
            {"error": "Could not generate the STL: %s" % exc, "traceback": error_trace}
        ), 500


@app.route("/api/comparison/jobs", methods=["GET"])
def comparison_jobs():
    try:
        return jsonify({"jobs": _comparison_jobs()})
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/comparison/compare", methods=["POST"])
def compare_jobs():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("job_ids"), list):
        return jsonify({"error": "Select generated jobs to compare."}), 400
    try:
        comparison = _compare_selected_jobs(payload["job_ids"])
        return jsonify(comparison)
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/comparison/jobs/<job_id>", methods=["DELETE"])
def delete_comparison_job(job_id: str):
    try:
        _delete_saved_job(job_id)
        return jsonify({"deleted": [job_id]})
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError:
        app.logger.exception("Could not delete generated job %s", job_id)
        return jsonify({"error": "Could not completely delete this job. Check the server logs."}), 500


@app.route("/api/comparison/jobs", methods=["DELETE"])
def delete_all_comparison_jobs():
    deleted = []
    failed = []
    for job in _comparison_jobs():
        job_id = job["job_id"]
        try:
            _delete_saved_job(job_id)
            deleted.append(job_id)
        except (PipelineError, OSError):
            app.logger.exception("Could not delete generated job %s during delete-all", job_id)
            failed.append(job_id)
    if failed:
        return jsonify(
            {
                "deleted": deleted,
                "failed": failed,
                "error": "Some jobs could not be deleted. Check the server logs.",
            }
        ), 500
    return jsonify({"deleted": deleted})


@app.route("/uploads/<job_id>/<filename>")
def uploaded_image(job_id: str, filename: str):
    if filename not in {"dorsal.jpg", "dorsal.jpeg", "dorsal.png", "palmar.jpg", "palmar.jpeg", "palmar.png"}:
        return jsonify({"error": "Unknown uploaded image."}), 404
    try:
        folder = _job_folder(UPLOAD_DIR, job_id)
    except PipelineError:
        return jsonify({"error": "Unknown upload."}), 404
    if not (folder / filename).is_file():
        return jsonify({"error": "Uploaded image not found."}), 404
    return send_from_directory(folder, filename, as_attachment=False)


@app.route("/upload", methods=["POST"])
def upload():
    _ensure_dirs()
    try:
        image_file = request.files.get("photo")
        suffix = _validate_image(image_file)

        hand_length_cm = float(request.form.get("hand_length_cm", 18))
        side = request.form.get("side", "dorsal").strip().lower()
        finger_extension = request.form.get("finger_extension", "false").lower() in {
            "1",
            "true",
            "on",
            "yes",
        }

        job_id = uuid.uuid4().hex
        upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
        image_file.save(upload_path)
        job_dir = OUTPUT_DIR / job_id
        result = run_pipeline(
            str(upload_path),
            str(job_dir),
            hand_length_cm=hand_length_cm,
            side=side,  # type: ignore[arg-type]
            finger_extension=finger_extension,
        )
        return jsonify(
            {
                "job_id": job_id,
                "cloud_url": f"/outputs/{job_id}/cloud.ply",
                "splint_url": f"/outputs/{job_id}/splint.stl",
                "landmarks_url": f"/outputs/{job_id}/landmarks.json",
                "point_count": result["meta"]["point_count"],
                "splint_watertight": result["meta"]["splint_watertight"],
            }
        )
    except HandNotDetectedError as exc:
        return jsonify({"error": str(exc)}), 400
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError:
        return jsonify({"error": "Hand length must be a number (centimetres)."}), 400
    except Exception as exc:
        error_trace = traceback.format_exc()
        # Some embedded Windows Python consoles have invalid stdout/stderr
        # handles. Logging must not replace the original pipeline exception.
        try:
            app.logger.error("Unexpected pipeline failure\n%s", error_trace)
        except OSError:
            pass
        return jsonify(
            {
                "error": f"Processing failed: {str(exc)}",
                "traceback": error_trace,
            }
        ), 500


@app.route("/outputs/<job_id>/<filename>")
def outputs(job_id: str, filename: str):
    allowed = {
        "cloud.ply", "splint.stl", "landmarks.json", "hand_preview.ply", "orthosis.stl",
        "palm_plate.stl", "thumb_assembly.stl", "index_assembly.stl",
        "middle_assembly.stl", "ring_assembly.stl", "pinky_assembly.stl",
    }
    is_component = bool(
        re.fullmatch(r"(?:thumb_palm_bridge|palm_plate|(?:thumb|index|middle|ring|pinky)_(?:ring_(?:mcp|pip|dip|ip)|palm_link|link_[1-3]|sensor_platform))\.stl", filename)
    )
    if filename not in allowed and filename != "orthosis_parts.zip" and not is_component:
        return jsonify({"error": "Unknown output file."}), 404
    try:
        folder = _job_folder(OUTPUT_DIR, job_id)
    except PipelineError:
        return jsonify({"error": "Unknown output."}), 404
    if not (folder / filename).is_file():
        return jsonify({"error": "Output not found."}), 404
    return send_from_directory(folder, filename, as_attachment=False)


if __name__ == "__main__":
    _ensure_dirs()
    app.run(debug=False)
