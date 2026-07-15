# Commercial Face Model Runtime Implementation Plan

> **Required sub-skill:** Use `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Add a reproducible, commercially deployable YuNet + SFace TensorRT runtime foundation without changing the active OpenCV face backend.

**Architecture:** Official OpenCV model artifacts are pinned by repository revision and SHA-256. A small downloader verifies models and license files before installation. Independent TensorRT adapters handle YuNet's multi-output detector and SFace's 128-dimensional embedding, while OpenCV is limited to crop resize, NMS, and five-point affine alignment. Runtime activation and the shared multi-camera scheduler are deliberately deferred until these model primitives pass parity and Jetson smoke tests.

**Tech Stack:** Python 3, NumPy, OpenCV, TensorRT 10, `trtexec`, pytest.

---

## Task 1: Pin and verify official model artifacts

**Files:**
- Create: `config/models/commercial_face_models.json`
- Create: `scripts/models/fetch_commercial_face_models.py`
- Create: `tests/test_fetch_commercial_face_models.py`

1. Write failing tests covering manifest parsing, fixed revision URLs, SHA-256 success/failure, required LICENSE verification, and rejection of unknown artifacts.
2. Run `rtk pytest tests/test_fetch_commercial_face_models.py -q` and confirm the tests fail because the fetch module does not exist.
3. Add the manifest entries below:
   - YuNet revision `3cc26e7f1014a5ee5d74a42acee58bafc9d0a310`, ONNX SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`, LICENSE SHA-256 `c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958`.
   - SFace revision `3d7082438a6e4551e840c9b2bb60b71e8da4b524`, ONNX SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`, LICENSE SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
4. Implement atomic downloads to a caller-selected directory, verify every byte before replacement, and avoid logging model contents or credentials.
5. Re-run the focused tests and commit the task.

## Task 2: Add deterministic TensorRT conversion commands

**Files:**
- Create: `scripts/convert/convert_commercial_face_models_to_engine.py`
- Create: `tests/test_convert_commercial_face_models.py`

1. Write failing tests for YuNet input `input:1x3x640x640`, SFace data input `data:1x3x112x112`, FP16 flags, explicit output paths, missing artifact errors, and zero-byte engine errors.
2. Run the focused test and verify the expected failure.
3. Implement separate command builders and a CLI that converts one or both models with the installed `trtexec`; keep engine files out of Git.
4. Do not pass SFace initializer tensors as runtime shape inputs; validate the ONNX graph has one image input plus embedded initializers.
5. Re-run tests and commit.

## Task 3: Implement SFace preprocessing and embedding runtime

**Files:**
- Create: `src/core/ai/_commercial_face_tensorrt.py`
- Create: `tests/test_commercial_face_tensorrt.py`

1. Write failing tests for BGR-to-SFace preprocessing, contiguous NCHW float32 output, 128-value output validation, finite/zero-norm checks, and L2 normalization.
2. Confirm focused test failure.
3. Implement `TensorRTSFaceEmbedder` behind an injected runtime factory so unit tests do not require CUDA.
4. Preserve the existing InsightFace research adapter unchanged and expose a distinct model ID, `opencv-sface-tensorrt-v1`.
5. Re-run tests and commit.

## Task 4: Implement YuNet output decoding

**Files:**
- Modify: `src/core/ai/_commercial_face_tensorrt.py`
- Modify: `tests/test_commercial_face_tensorrt.py`

1. Add failing synthetic tests for the 12 fixed outputs (`cls/obj/bbox/kps` at strides 8, 16, 32), score filtering, NMS, landmark ordering, clamping, and ROI-to-frame coordinate restoration.
2. Confirm focused test failure.
3. Implement preprocessing and decode logic using NumPy/OpenCV CPU post-processing only; TensorRT remains responsible for inference.
4. Reject missing, duplicate, or unexpected output shapes with actionable errors.
5. Re-run tests and commit.

## Task 5: Implement five-point SFace alignment

**Files:**
- Modify: `src/core/ai/_commercial_face_tensorrt.py`
- Modify: `tests/test_commercial_face_tensorrt.py`

1. Add failing tests for the official SFace five-landmark template, deterministic 112x112 alignment, invalid landmark count, degenerate transforms, and non-finite points.
2. Confirm focused test failure.
3. Implement the smallest OpenCV affine alignment helper; do not save aligned crops to disk.
4. Re-run tests and commit.

## Task 6: Add artifact/readiness safeguards

**Files:**
- Create: `scripts/health/check_commercial_face_models.py`
- Create: `tests/test_check_commercial_face_models.py`
- Modify: `.gitignore`

1. Add failing tests proving readiness fails for missing/changed ONNX, engine, or LICENSE files and rejects InsightFace pretrained paths in a commercial manifest.
2. Implement a read-only health command with machine-readable JSON and non-zero failure status.
3. Ignore generated `.engine` files and downloaded model binaries while retaining manifest/license metadata.
4. Re-run tests and commit.

## Task 7: Jetson conversion and runtime smoke verification

**Files:**
- Create: `scripts/smoke/smoke_test_commercial_face_tensorrt.py`
- Create: `tests/test_smoke_test_commercial_face_tensorrt.py`
- Create: `docs/operations/commercial-face-models.md`

1. Write unit tests for smoke-test result validation before accessing CUDA.
2. Fetch pinned artifacts into the ignored local model directory and verify hashes/licenses.
3. Convert YuNet and SFace to FP16 engines on the target Jetson.
4. Run one warm-up and repeated inference for each engine; verify all outputs are finite and match fixed shapes.
5. Compare SFace ONNX reference output and TensorRT output by cosine similarity, if a stable ONNX reference provider runs on this Jetson; otherwise record this parity item as unverified rather than passing it.
6. Document exact commands, measured latency, engine portability limits, licenses, and rollback (`FACE_RECOGNITION_BACKEND=opencv`).
7. Run the focused suite, then `rtk pytest tests/test_face_tensorrt.py tests/test_commercial_face_tensorrt.py tests/test_fetch_commercial_face_models.py tests/test_convert_commercial_face_models.py tests/test_check_commercial_face_models.py tests/test_smoke_test_commercial_face_tensorrt.py -q`.
8. Run the existing face pipeline regression tests and commit.

## Task 8: Final branch verification

1. Run the complete test suite with `rtk pytest -q`.
2. Run `rtk git diff --check` and inspect `rtk git status --short`.
3. Confirm the active `.env.jetson` backend was not changed and the main worktree's fall-detection edits were not touched.
4. Record verified results and remaining blockers: real enrolled identities, camera threshold calibration, shared multi-camera scheduler, and production activation.

