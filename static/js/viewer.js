import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const canvas = document.getElementById("viewport");
const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const spinner = document.getElementById("spinner");
const toggleCloud = document.getElementById("toggle-cloud");
const toggleSplint = document.getElementById("toggle-splint");
const dlPly = document.getElementById("dl-ply");
const dlStl = document.getElementById("dl-stl");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);

const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 10);
camera.position.set(0.12, 0.08, 0.28);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(0.4, 0.6, 0.5);
scene.add(key);
scene.add(new THREE.GridHelper(0.4, 8, 0x334155, 0x1e293b));

let cloudMesh = null;
let splintMesh = null;

function resize() {
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = canvas.clientHeight || window.innerHeight - 88;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
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
  statusEl.classList.toggle("error", false);
  form.querySelector("button[type=submit]").disabled = busy;
}

function showError(message) {
  setBusy(false, message);
  statusEl.classList.add("error");
}

function clearMeshes() {
  if (cloudMesh) {
    scene.remove(cloudMesh);
    cloudMesh.geometry.dispose();
    cloudMesh.material.dispose();
    cloudMesh = null;
  }
  if (splintMesh) {
    scene.remove(splintMesh);
    splintMesh.geometry.dispose();
    splintMesh.material.dispose();
    splintMesh = null;
  }
}

function fitCamera() {
  const box = new THREE.Box3();
  if (cloudMesh) box.expandByObject(cloudMesh);
  if (splintMesh) box.expandByObject(splintMesh);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size * 0.4, size * 0.3, size * 0.9));
  camera.near = Math.max(size / 200, 0.001);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
}

toggleCloud.addEventListener("change", () => {
  if (cloudMesh) cloudMesh.visible = toggleCloud.checked;
});
toggleSplint.addEventListener("change", () => {
  if (splintMesh) splintMesh.visible = toggleSplint.checked;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  if (!data.get("finger_extension")) {
    data.set("finger_extension", "false");
  } else {
    data.set("finger_extension", "true");
  }
  setBusy(true, "Detecting the hand and estimating depth (first run downloads models)…");
  clearMeshes();
  dlPly.classList.add("disabled");
  dlStl.classList.add("disabled");

  try {
    const response = await fetch("/upload", { method: "POST", body: data });
    const responseText = await response.text();
    let payload;
    try {
      payload = responseText ? JSON.parse(responseText) : {};
    } catch {
      // Flask's debug server can return an HTML traceback page for errors
      // raised before the route's JSON error handler runs.
      const detail = responseText.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
      throw new Error(
        detail.slice(0, 500) || `Server returned HTTP ${response.status}. Check the Flask terminal.`
      );
    }
    if (!response.ok) {
      const details = payload.traceback ? `\n\n${payload.traceback}` : "";
      throw new Error(`${payload.error || "Upload failed."}${details}`);
    }

    const plyLoader = new PLYLoader();
    const stlLoader = new STLLoader();
    const [plyGeom, stlGeom] = await Promise.all([
      plyLoader.loadAsync(payload.cloud_url),
      stlLoader.loadAsync(payload.splint_url),
    ]);

    plyGeom.computeVertexNormals();
    cloudMesh = new THREE.Points(
      plyGeom,
      new THREE.PointsMaterial({ size: 0.0016, vertexColors: true })
    );
    cloudMesh.position.x = -0.06;
    cloudMesh.visible = toggleCloud.checked;
    scene.add(cloudMesh);

    stlGeom.computeVertexNormals();
    splintMesh = new THREE.Mesh(
      stlGeom,
      new THREE.MeshStandardMaterial({
        color: 0x5b8def,
        metalness: 0.05,
        roughness: 0.45,
        side: THREE.DoubleSide,
      })
    );
    splintMesh.position.x = 0.08;
    splintMesh.visible = toggleSplint.checked;
    scene.add(splintMesh);

    dlPly.href = payload.cloud_url;
    dlStl.href = payload.splint_url;
    dlPly.classList.remove("disabled");
    dlStl.classList.remove("disabled");
    fitCamera();
    setBusy(
      false,
      `Done. ${payload.point_count} points. Splint watertight: ${payload.splint_watertight}.`
    );
  } catch (err) {
    showError(err.message || String(err));
  }
});
