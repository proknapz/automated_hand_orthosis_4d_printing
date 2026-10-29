import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const form = document.getElementById("upload-form");
const reviewPanel = document.getElementById("review-panel");
const statusEl = document.getElementById("status");
const spinner = document.getElementById("spinner");
const canvas = document.getElementById("viewport");
const partDownloads = document.getElementById("part-downloads");
const meshSummary = document.getElementById("mesh-summary");
const views = {};
let currentJob = null;
let submittedMeasurements = null;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);
const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 10);
camera.position.set(0.12, 0.08, 0.28);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
keyLight.position.set(0.4, 0.6, 0.5);
scene.add(keyLight);
scene.add(new THREE.GridHelper(0.4, 8, 0x334155, 0x1e293b));

const modelGroup = new THREE.Group();
scene.add(modelGroup);
let handMesh = null;
let orthosisMesh = null;
let landmarkGroup = null;
let dragState = null;

function resize() {
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  const height = canvas.clientHeight || window.innerHeight - 88;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

function setBusy(busy, message) {
  spinner.classList.toggle("hidden", !busy);
  spinner.setAttribute("aria-hidden", busy ? "false" : "true");
  statusEl.textContent = message || "";
  statusEl.classList.remove("error");
  form.querySelector("button[type=submit]").disabled = busy;
  document.getElementById("generate-button").disabled = busy;
}

function showError(message) {
  setBusy(false, message);
  statusEl.classList.add("error");
}

async function readResponse(response) {
  const raw = await response.text();
  let payload;
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`Server returned HTTP ${response.status}. Check the Flask console.`);
  }
  if (!response.ok) {
    const details = payload.traceback ? `\n\n${payload.traceback}` : "";
    throw new Error(`${payload.error || "Request failed."}${details}`);
  }
  return payload;
}

function markerLabel(name) {
  const labels = {
    wrist: "Wrist", "thumb CMC": "Th-CMC", "thumb MCP": "Th-MCP",
    "thumb IP": "Th-IP", "thumb tip": "Th-tip", "index MCP": "I-MCP",
    "index PIP": "I-PIP", "index DIP": "I-DIP", "index tip": "I-tip",
    "middle MCP": "M-MCP", "middle PIP": "M-PIP", "middle DIP": "M-DIP",
    "middle tip": "M-tip", "ring MCP": "R-MCP", "ring PIP": "R-PIP",
    "ring DIP": "R-DIP", "ring tip": "R-tip", "pinky MCP": "P-MCP",
    "pinky PIP": "P-PIP", "pinky DIP": "P-DIP", "pinky tip": "P-tip",
  };
  return labels[name] || name;
}

function drawLandmarks(viewName) {
  const view = views[viewName];
  const svg = document.getElementById(`${viewName}-overlay`);
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${view.width} ${view.height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const radius = Math.max(view.width, view.height) * 0.006;
  view.points.forEach((point, index) => {
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", point[0]);
    marker.setAttribute("cy", point[1]);
    marker.setAttribute("r", radius);
    marker.setAttribute("class", "landmark-marker");
    marker.dataset.index = index;
    marker.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      dragState = { viewName, index, pointerId: event.pointerId };
      marker.classList.add("active");
      svg.setPointerCapture(event.pointerId);
    });
    svg.appendChild(marker);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", point[0] + radius * 1.25);
    label.setAttribute("y", point[1] - radius * 0.8);
    label.setAttribute("font-size", Math.max(view.width, view.height) * 0.018);
    label.setAttribute("class", "landmark-label");
    label.textContent = markerLabel(view.labels[index]);
    svg.appendChild(label);
  });
}

function pointerPosition(svg, event) {
  const p = svg.createSVGPoint();
  p.x = event.clientX;
  p.y = event.clientY;
  const transformed = p.matrixTransform(svg.getScreenCTM().inverse());
  return [
    Math.max(0, Math.min(svg.viewBox.baseVal.width, transformed.x)),
    Math.max(0, Math.min(svg.viewBox.baseVal.height, transformed.y)),
  ];
}

document.querySelectorAll(".photo-stage svg").forEach((svg) => {
  svg.addEventListener("pointermove", (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const next = pointerPosition(svg, event);
    views[dragState.viewName].points[dragState.index] = next;
    drawLandmarks(dragState.viewName);
    const newMarker = svg.querySelector(`[data-index="${dragState.index}"]`);
    if (newMarker) newMarker.classList.add("active");
  });
  const finishDrag = (event) => {
    if (dragState && dragState.pointerId === event.pointerId) dragState = null;
  };
  svg.addEventListener("pointerup", finishDrag);
  svg.addEventListener("pointercancel", finishDrag);
});

function showConfidence(viewName, metadata) {
  const target = document.getElementById(`${viewName}-confidence`);
  target.textContent = metadata.confidence == null
    ? `${metadata.handedness} hand detected; per-landmark confidence is unavailable.`
    : `${metadata.handedness} hand detected; handedness score ${(metadata.confidence * 100).toFixed(0)}%.`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submittedMeasurements = new FormData(form);
  const dorsal = submittedMeasurements.get("dorsal_photo");
  const palmar = submittedMeasurements.get("palmar_photo");
  if (!dorsal || !dorsal.size || !palmar || !palmar.size) {
    showError("Choose both the dorsal and palm photos.");
    return;
  }
  currentJob = null;
  reviewPanel.classList.add("hidden");
  setBusy(true, "Detecting hand landmarks in both photos…");
  try {
    const payload = await readResponse(await fetch("/analyze", {
      method: "POST", body: submittedMeasurements,
    }));
    currentJob = payload.job_id;
    for (const viewName of ["dorsal", "palmar"]) {
      const data = payload[viewName];
      const image = document.getElementById(`${viewName}-preview`);
      image.src = payload[`${viewName}_image_url`];
      image.onload = () => {
        views[viewName] = {
          width: data.image_size.width,
          height: data.image_size.height,
          points: data.landmarks.map((item) => [item.pixel.x, item.pixel.y]),
          labels: data.landmarks.map((item) => item.name),
        };
        drawLandmarks(viewName);
      };
      showConfidence(viewName, data);
    }
    const note = document.getElementById("handedness-note");
    note.textContent = payload.handedness_warning
      ? `${payload.handedness_warning} You can still review and correct the landmarks.`
      : "Landmarks are editable. Review both views, especially wrist and finger joints.";
    reviewPanel.classList.remove("hidden");
    setBusy(false, payload.message);
  } catch (error) {
    showError(error.message || String(error));
  }
});

function clearModel() {
  for (const child of [...modelGroup.children]) {
    modelGroup.remove(child);
    child.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => {
          if (material.map) material.map.dispose();
          material.dispose();
        });
      }
    });
  }
  handMesh = null;
  orthosisMesh = null;
  landmarkGroup = null;
  partDownloads.replaceChildren();
  meshSummary.textContent = "";
  for (const id of ["dl-hand", "dl-orthosis"]) {
    const link = document.getElementById(id);
    link.classList.add("disabled");
    link.removeAttribute("href");
  }
}

function addLandmarkLabels(coords, labels, dorsalSurfaceOffset) {
  landmarkGroup = new THREE.Group();
  const dotGeometry = new THREE.SphereGeometry(0.0018, 8, 8);
  const dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffd166 });
  coords.forEach((point, index) => {
    const position = new THREE.Vector3(point[0], point[1], point[2] + dorsalSurfaceOffset);
    const dot = new THREE.Mesh(dotGeometry, dotMaterial);
    dot.position.copy(position);
    landmarkGroup.add(dot);
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 256;
    labelCanvas.height = 64;
    const context = labelCanvas.getContext("2d");
    context.font = "bold 28px Segoe UI, sans-serif";
    context.fillStyle = "#ffffff";
    context.strokeStyle = "#111418";
    context.lineWidth = 6;
    const label = markerLabel(labels[index]);
    context.strokeText(label, 5, 42);
    context.fillText(label, 5, 42);
    const texture = new THREE.CanvasTexture(labelCanvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
    sprite.position.copy(position).add(new THREE.Vector3(0.003, 0.003, 0.001));
    sprite.scale.set(0.018, 0.0045, 1);
    landmarkGroup.add(sprite);
  });
  landmarkGroup.visible = document.getElementById("toggle-landmarks").checked;
  modelGroup.add(landmarkGroup);
}

function fitCamera() {
  const bounds = new THREE.Box3().setFromObject(modelGroup);
  if (bounds.isEmpty()) return;
  const size = bounds.getSize(new THREE.Vector3()).length();
  const center = bounds.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size * 0.8, size * 0.8, size * 1.8));
  camera.near = Math.max(size / 500, 0.0001);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
}

document.getElementById("generate-button").addEventListener("click", async () => {
  if (!currentJob || !views.dorsal || !views.palmar) {
    showError("Analyze both photos and review the landmarks first.");
    return;
  }
  const data = new FormData(form);
  data.set("job_id", currentJob);
  data.set("corrections", JSON.stringify({
    dorsal_points: views.dorsal.points,
    palmar_points: views.palmar.points,
  }));
  const sizes = {};
  for (const digit of ["thumb", "index", "middle", "ring", "pinky"]) {
    const value = data.get(`finger_${digit}_circumference_mm`);
    if (value) sizes[digit] = value;
  }
  data.set("finger_circumferences_mm", JSON.stringify(sizes));
  if (!data.get("include_thumb")) data.set("include_thumb", "false");

  setBusy(true, "Building the estimated hand mesh and open-ring orthosis…");
  clearModel();
  try {
    const payload = await readResponse(await fetch("/generate", { method: "POST", body: data }));
    const plyGeometry = await new PLYLoader().loadAsync(payload.hand_preview_url);
    plyGeometry.computeVertexNormals();
    const handMaterial = new THREE.MeshStandardMaterial({
      color: 0xc98869,
      roughness: 0.85,
      metalness: 0,
      side: THREE.DoubleSide,
    });
    handMesh = new THREE.Mesh(plyGeometry, handMaterial);
    handMesh.visible = document.getElementById("toggle-hand").checked;
    modelGroup.add(handMesh);

    const stlGeometry = await new STLLoader().loadAsync(payload.orthosis_url);
    // STL coordinates are exported in millimetres for slicers; the hand preview
    // and viewer scene use metres, so scale the display copy back to metres.
    stlGeometry.scale(0.001, 0.001, 0.001);
    stlGeometry.computeVertexNormals();
    orthosisMesh = new THREE.Mesh(stlGeometry, new THREE.MeshStandardMaterial({
      color: 0x4d8ce0,
      roughness: 0.55,
      metalness: 0.02,
      side: THREE.DoubleSide,
    }));
    orthosisMesh.visible = document.getElementById("toggle-orthosis").checked;
    modelGroup.add(orthosisMesh);
    addLandmarkLabels(
      payload.meta.landmarks_3d_m,
      payload.meta.landmark_labels,
      payload.meta.palm_thickness_mm / 2000 + 0.004
    );
    fitCamera();

    const handLink = document.getElementById("dl-hand");
    handLink.href = payload.hand_preview_url;
    handLink.classList.remove("disabled");
    const orthosisLink = document.getElementById("dl-orthosis");
    orthosisLink.href = payload.orthosis_url;
    orthosisLink.classList.remove("disabled");
    payload.part_urls.forEach((url) => {
      let list = document.getElementById("component-links");
      if (!list) {
        const details = document.createElement("details");
        details.id = "component-details";
        details.className = "component-details";
        const summary = document.createElement("summary");
        summary.textContent = "Individual STL components";
        list = document.createElement("div");
        list.id = "component-links";
        list.className = "component-links";
        details.append(summary, list);
        partDownloads.appendChild(details);
        const zip = document.createElement("a");
        zip.href = payload.parts_zip_url;
        zip.download = "orthosis_parts.zip";
        zip.className = "btn";
        zip.textContent = "Download separate parts (ZIP)";
        partDownloads.prepend(zip);
      }
      const link = document.createElement("a");
      const partName = url.split("/").pop();
      link.href = url;
      link.download = partName;
      link.className = "btn component-link";
      link.textContent = partName.replace(/\.stl$/, "").replaceAll("_", " ");
      list.appendChild(link);
    });
    const report = payload.meta.mesh;
    meshSummary.textContent = report.component_shells_watertight
      ? `Mesh: ${report.vertices} vertices, ${report.faces} triangles. ${report.closed_part_count} individual part shells are closed; assembled STL watertight: ${report.watertight}. Estimated geometry only.`
      : `Mesh: ${report.vertices} vertices, ${report.faces} triangles. Some parts are not closed; review before printing.`;
    setBusy(false, "Mesh generated. Review the fit visually; the two photos do not provide a calibrated 3D scan.");
  } catch (error) {
    showError(error.message || String(error));
  }
});

document.getElementById("reset-button").addEventListener("click", () => {
  currentJob = null;
  submittedMeasurements = null;
  views.dorsal = null;
  views.palmar = null;
  form.reset();
  reviewPanel.classList.add("hidden");
  for (const name of ["dorsal", "palmar"]) {
    document.getElementById(`${name}-preview`).removeAttribute("src");
    document.getElementById(`${name}-overlay`).replaceChildren();
  }
  clearModel();
  setBusy(false, "Choose both photos to begin.");
});

document.getElementById("toggle-hand").addEventListener("change", (event) => {
  if (handMesh) handMesh.visible = event.target.checked;
});
document.getElementById("toggle-orthosis").addEventListener("change", (event) => {
  if (orthosisMesh) orthosisMesh.visible = event.target.checked;
});
document.getElementById("toggle-landmarks").addEventListener("change", (event) => {
  if (landmarkGroup) landmarkGroup.visible = event.target.checked;
});
