# Orthosis geometry and representation requirements review

## Purpose

Use this worksheet with a qualified reviewer familiar with the intended hand
orthosis, such as the responsible orthotist, hand therapist, rehabilitation
professional, or biomedical engineer. Its purpose is to obtain use-case-specific
design/engineering limits and the reviewer's approval; it does not prescribe
clinical thresholds or establish clinical validity.

Do not include patient names or other identifying information. Do not label a
threshold "clinical" unless the reviewer confirms that status and provides its
basis. A project-team member should record the reviewer and approval below.

## Suggested request to the reviewer

> We are evaluating a 3D hand-geometry representation for a research orthosis
> workflow. Could you review the attached worksheet and define or approve
> measurable, use-case-specific requirements for hand posture/load, contact and
> keep-clear regions, fit deviation, and minimum finger/device clearance? Please
> state units, measurement method, relevant regions and conditions, and whether
> each value is a clinical requirement or an engineering target, with its
> rationale/source. Please also identify the appropriate reference surface and
> any evidence needed before the requirements can be applied. Please indicate
> whether you can review representation error only, or whether actual orthosis
> fit can be assessed with the supplied hand and device geometry. We will not
> treat voxel quantization measurements as fit or clinical validation.

## What is known about the current geometry

- FreiHAND XYZ and reference-bone values are in meters. The official
  [`eval.py`](https://github.com/lmb-freiburg/freihand/blob/master/eval.py)
  converts raw-coordinate errors to centimeters by multiplying by 100;
  [`eval_util.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py)
  computes Euclidean error directly from the stored values.
- The current candidate encoder uses a per-sample bounding-box normalization
  to `[-1, 1]` and a `96 x 96 x 96` grid. Its normalized cell width is `1/48`.
  Physical cell width varies with each sample's retained transform; use that
  transform to report errors in meters or millimeters.
- FreiHAND provides 21 ordered 3D landmarks and 778 vertices in the inspected
  training version. The dataset archive does not include faces or per-surface
  anatomical labels. The present fingertip measure is landmark-to-landmark
  distance, not skin-to-skin clearance.
- FreiHAND's formal camera-frame origin, axis signs, and handedness are not
  specified in the inspected sources. The encoder preserves incoming XYZ; it
  does not orient the hand anatomically.
- Reported voxel errors measure representation quantization only. They do not
  include acquisition, segmentation, registration, mesh reconstruction,
  manufacturing, soft-tissue, or fit error.
- A reviewer can specify prospective criteria with this worksheet, but actual
  hand-to-device fit cannot be measured from the current inputs: there is no
  patient-specific scan with validated surface regions and no orthosis CAD or
  manufactured device surface in this project data.

## Evidence available for this review

Check what is actually supplied with the request; leave unchecked items marked
as missing. FreiHAND is research data and must not be represented as the
intended patient's hand scan.

| Evidence/input | Supplied? | File/version or reason unavailable |
|---|---|---|
| Intended use, hand side, and use posture/load description | | |
| Patient-specific hand surface in the relevant posture(s) | | |
| Surface topology or validated point-cloud/surface reconstruction method | | |
| Reviewed anatomical contact/keep-clear region annotations | | |
| Candidate orthosis CAD surface or manufactured-device scan | | |
| Hand-to-device registration method and transform | | |
| Calibration/measurement uncertainty and units | | |
| Material, thickness, compliance, and manufacturing process (if relevant to fit) | | |

If the current request includes only FreiHAND landmarks/vertices and a voxel
encoding, the review can address representation specifications and proposed
engineering error budgets. It cannot establish patient-specific fit, pressure,
comfort, safety, or clinical acceptance.

## Review record

| Field | Reviewer response |
|---|---|
| Reviewer name and professional role | |
| Relevant experience/qualification | |
| Organization (optional) | |
| Review date | |
| Intended device and intended use | |
| Is this feedback a clinical requirement, engineering target, or research proposal? | |
| Source/basis for requirements (standard, protocol, experience, measurement, other) | |
| Conditions or populations the requirements do not cover | |

## 1. Intended hand posture and loading

Specify the hand side, posture, and whether the requirement is static or applies
during motion. Do not assume the dataset pose represents the orthosis use pose.

| Condition | Reviewer response |
|---|---|
| Hand side(s) | |
| Required posture(s)/joint positions | |
| Static or dynamic use; motions to consider | |
| Loading, activity, or external-force condition | |
| Wear duration or other relevant condition | |
| Must separate geometries be evaluated for each posture/load? | |

## 2. Contact, support, and keep-clear regions

Please mark regions on an agreed anatomical diagram or scan/mesh. Avoid relying
on informal names alone; attach the annotated image or region-label file and
record its version. This project currently has no verified vertex/face
segmentation for those regions.

| Region ID | Contact/support/keep-clear intent | Required posture/load | Annotation artifact/version |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Reviewer notes on sensitive or excluded regions:

> 

## 3. Quantitative geometry and fit limits

For each row, the reviewer should define the threshold, units, measurement
direction, region, posture/load, and summary statistic. If a requirement is not
applicable, mark N/A. Suggested statistics such as p95 or maximum are prompts,
not defaults.

| Measure | Region and conditions | Limit and units | Direction/statistic (e.g. intrusion, gap, p95, maximum) | Required reference data/method | Requirement basis |
|---|---|---|---|---|---|
| Hand-to-device surface deviation in contact/support regions | | | | Registered hand surface and candidate device CAD/scan; define signed-distance convention | |
| Maximum device intrusion/interference | | | | Registered hand surface and candidate device CAD/scan; specify which surface is tested | |
| Minimum device-to-skin clearance in keep-clear regions | | | | Registered hand surface and candidate device CAD/scan; define sign and sampled region | |
| Minimum adjacent-digit skin-to-skin clearance (if relevant) | | | | Hand surface captured in the specified posture; distinct from device clearance | |
| Critical landmark position error (identify landmarks) | | | | Named landmarks and reference/registration frame | |
| Depth/out-of-plane representation error | | | | Source geometry, axis/frame definition, and whether measured before or after normalization | |
| Other | | | | | |

For every surface-distance metric, specify whether positive distance means a
gap or penetration, how normals/inside-outside are determined, how the surface
is sampled, how missing or invalid regions are handled, and which summary
statistic controls acceptance. Do not combine hand-to-device clearance,
adjacent-digit spacing, and voxel reconstruction error into one threshold.

### Representation-error budget

The voxel quantization error is only one contributor to end-to-end geometry
error. Please indicate whether a separate maximum budget should be allocated
to representation error, and how it relates to the complete fit/clearance
limit. Do not assume the entire fit tolerance is available to the encoder.

| Field | Reviewer response |
|---|---|
| Is a separate representation-error budget needed? | |
| Allowed point/surface representation error and units, if applicable | |
| Allowed landmark/depth representation error and units, if applicable | |
| Required statistic and anatomical regions | |
| How should the budget account for scan/registration/reconstruction/fabrication errors? | |
| Which source surface should the encoded geometry be compared with, and how is it registered to the orthosis frame? | |
| Should the representation error be measured on raw points, a verified continuous mesh, or both? | |
| If a limit cannot be given yet, what evidence or measurement is needed? | |

## 4. Surface-clearance validation data

The current FreiHAND annotation archive has vertices but no faces or surface
region labels. The official FreiHAND reference code can use a separately
obtained MANO model, whose model object exposes a face array. That is a
candidate topology source, not yet validated for this repository's samples.

| Question | Reviewer/project response |
|---|---|
| Is MANO-derived surface topology acceptable for this research measurement? | |
| Must the surface instead come from a patient-specific scan or another source? | |
| Which anatomical regions must be separately labeled? Who will review the labels? | |
| What posture/load must each surface represent? | |
| What distance method and sampling density should be used? | |
| What evidence is required to verify vertex/face correspondence? | |
| Is the goal a representation-only error study or actual device-fit assessment? | |
| If actual device fit is in scope, where are the hand surface and candidate device CAD/scan? | |

MANO topology, if authorized and verified, would provide connectivity only. It
does not itself provide reviewer-approved orthosis contact zones, patient
surface truth, or clinical surface-clearance criteria.

## 5. Coordinate, scale, and operational requirements

| Question | Reviewer/project response |
|---|---|
| Is preserving the published dataset XYZ orientation without anatomical rotation acceptable for this stage? | |
| Is a common anatomical orientation required for comparison/design? If yes, define its reference landmarks and transformation rule. | |
| Is per-sample bounding-box normalization acceptable if the inverse transform and raw meter-valued geometry are retained? | |
| Is physical voxel size expected to be fixed across patients/samples? | |
| Confirm memory cap (32 MiB per sample tensor payload) against target batch/model | |
| Confirm runtime cap (5 seconds per sample) and target hardware | |
| Additional constraints/acceptance checks | |

## 6. Decision and sign-off

| Decision | Reviewer response |
|---|---|
| Requirements approved for engineering evaluation? | |
| Approved requirements and exceptions | |
| Evidence still required before evaluation | |
| Is any statement approved as a clinical acceptance criterion? Identify it and its basis. | |
| Reviewer name/date/signature or documented approval reference | |
| Project owner recording the decision/date | |

## How to use the completed review

1. Preserve the completed form and any annotated region files as versioned
   project inputs; do not silently overwrite earlier approvals.
2. Translate each approved measure into an automated acceptance check only
   after confirming the data source, coordinate frame, units, posture, region,
   sign convention, registration, and statistic are represented by that check.
3. If only end-to-end fit limits are supplied, ask the reviewer/project team to
   allocate an encoder-specific error budget before judging the voxel setting.
4. Update `recommended_representation.md` and
   `representation_contract.json` with the approved values, reviewer/date, and
   evidence source. Keep unapproved rows marked pending.
5. Keep representation verification, orthosis fit evaluation, and clinical
   validation as separate claims.

## Current project blockers

- The official FreiHAND v2 archive has no face array. The official reference
  README points to the separately licensed MANO model; its `MANO_RIGHT.pkl` is
  not present locally. Obtain it only through its authorized distribution,
  verify its version/hash and topology, then test correspondence against
  FreiHAND vertices before applying its faces.
- No reviewed anatomical surface segmentation or patient-specific orthosis
  surface exists in this dataset. A domain reviewer must identify regions and
  acceptable measurement sources; labels cannot be inferred from keypoint
  chain names.
- Actual fit/clearance analysis additionally requires a relevant hand surface,
  candidate orthosis CAD or scan, and a defined registration. Reviewer-approved
  limits alone do not make those measurements possible without the geometry.
- The official coordinate frame's precise origin, axis signs, and handedness
  remain undocumented in inspected references. Preserve source XYZ as-is
  unless a reviewer approves a defined transform for the intended use.
- The 96-cubed representation is selected for prototype engineering evaluation.
  It is not approved for fit, clearance, safety, or clinical use; quantitative
  domain requirements and relevant geometry evidence are needed before making
  those claims.
