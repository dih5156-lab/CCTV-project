# Public Fall Data Package Integration Notes

## Conclusion

The `falldata/` package is usable as a POC reference package, but it is not a
drop-in replacement for the current runtime fall detector.

The current project detects falls from YOLO pose keypoints through
`src/core/ai/_fall_detector.py` and the DeepStream/OpenCV event pipeline. The
public package uses 600-frame feature sequences and sklearn/Keras models, so an
input adapter is required before runtime integration.

## Confirmed Package Shape

- `falldata/낙상방향-003.tar`
  - Contains `SCH_FN`
  - Contains `NIA_FNF_TEST.py`
  - Appears to be the 2-class fall / non-fall package despite the filename.
- `falldata/낙상유무탐지-002.tar`
  - Contains `SCH_FNF`
  - Contains `NIA_FD_TEST.py`
  - Appears to be the 3-class fall direction/type package despite the filename.
- `falldata/2. AI학습모델파일`
  - Contains video RandomForest `.pkl` files.
  - Contains sensor `.sav` and `.h5` files.

The bundled `test.zip` files contain precomputed `.npy` feature sequences, not
raw videos.

## Model Probe Result

Checked with:

```bash
.venv/bin/python scripts/datasets/probe_falldata_models.py
```

Current `.venv` has `sklearn 1.7.2`, while the public package was saved around
`sklearn 1.3.2`. Some models load with compatibility warnings, but this should
not be treated as production-safe.

Confirmed input sizes:

- Video RandomForest `.pkl` models: `n_features_in_ = 997200`
  - This matches `600 x 1662` flattened frame features.
  - The `1662` frame feature is MediaPipe Holistic:
    `pose 33*4 + face 468*3 + left hand 21*3 + right hand 21*3`.
  - It is not directly compatible with the current YOLO pose keypoints.
- Sensor spatio-temporal `.sav` models: `n_features_in_ = 2484`
  - These expect engineered sensor features, not CCTV frame keypoints.

Observed load status in current `.venv`:

- Loadable with warnings: RandomForest, AdaBoost, MLP, SVM models.
- Failed without matching legacy deps: GBC/ensemble models referencing
  `sklearn.ensemble._gb_losses`.
- Failed without optional deps: XGBoost models need `xgboost`.
- Skipped/unsupported in current `.venv`: Keras `.h5` models because TensorFlow
  is not installed.

Observed runtime smoke result:

- Current `.venv` can load some RandomForest models, but prediction failed with
  a sklearn compatibility error:
  `DecisionTreeClassifier` missing `monotonic_cst`.
- A separate `.venv-falldata` environment with `scikit-learn==1.3.2` can load
  and call the video RandomForest model successfully.
- Smoke input used a synthetic zero vector, so this confirms interface
  compatibility only, not detection accuracy.

## Practical Integration Path

1. Keep the existing `FallDetector` as the first-stage detector.
2. Buffer pose/keypoint features for a short window only after a fall candidate
   is detected.
3. Add or reproduce the public package's 1662-dimension frame feature extractor.
   The current YOLO pose output alone is not enough.
4. Convert the buffered frame features into `600 x 1662`, then flatten to
   `997200`.
5. Run the public video RandomForest model as a second-stage verifier.
6. Keep sensor `.sav`/`.h5` models out of the CCTV runtime unless real 108-axis
   sensor data is available.
7. Publish `fall_detected` only when the current detector and verifier agree, or
   use the verifier confidence as metadata during field testing.

## Risks

- The public package expects Python 3.9-era dependencies.
- The active `.venv` currently has newer sklearn/numpy versions, so loading the
  `.pkl`/`.sav` files in the main runtime can fail or behave differently.
- The package naming is confusing; do not wire model paths by filename alone.
- The test data is feature data, so it cannot directly validate raw CCTV video
  performance.

## Validation Command

```bash
.venv/bin/python scripts/datasets/check_falldata_package.py --root falldata
```

Model loading probe:

```bash
.venv/bin/python scripts/datasets/probe_falldata_models.py
```

Isolated sklearn POC environment:

```bash
python3.10 -m venv .venv-falldata
.venv-falldata/bin/pip install numpy==1.26.1 scipy==1.11.3 scikit-learn==1.3.2 joblib==1.3.2
.venv-falldata/bin/python scripts/datasets/probe_falldata_models.py
.venv-falldata/bin/python scripts/datasets/smoke_falldata_video_model.py
```

Observed smoke output for `FNF_RF_SMOTE_CAM_1.pkl`:

- sklearn: `1.3.2`
- input shape: `(1, 997200)`
- prediction: `[0]`
- probability: approximately `[[0.9337, 0.0663]]`
- The package's 2-class labels appear to map `Fall` to class index `0`, so the
  runtime auxiliary verifier defaults `FALLDATA_AUX_FALL_CLASS_INDEX=0`.

Video-to-feature POC uses a separate MediaPipe environment because current
MediaPipe wheels pull `numpy>=2`, while `scikit-learn==1.3.2` requires
`numpy<2`.

```bash
python3.10 -m venv .venv-mediapipe
.venv-mediapipe/bin/pip install opencv-python-headless mediapipe

.venv-mediapipe/bin/python scripts/datasets/extract_falldata_mediapipe_features.py \
  --video path/to/sample.mp4 \
  --output-dir /tmp/falldata_sample_features

.venv-falldata/bin/python scripts/datasets/smoke_falldata_video_model.py \
  --sequence-dir /tmp/falldata_sample_features
```

This path is still a POC. It reproduces the public package's expected feature
shape, but real CCTV accuracy must be checked with labeled fall/non-fall clips.

Observed end-to-end shape smoke using
`external/OpenPAR/VTFPAR++/demo/video.mp4`:

- decoded frames: `60` when run with `--max-frames 60`
- saved frames: `600` after zero padding
- frame feature size: `1662`
- nonzero feature frames: `40`
- RF model input shape: `(1, 997200)`
- prediction: `[0]`
- probability: approximately `[[0.9185, 0.0815]]`

This demo clip is not a labeled fall clip, so the prediction is only an
interface check.

## Runtime Auxiliary Verifier

The OpenCV/`AIAnalyzer` path now has a disabled-by-default auxiliary verifier.
It buffers recent frames and, when the pose detector emits a fall candidate,
runs the public MediaPipe/RF path in subprocesses.

Default mode is safe:

```bash
FALLDATA_AUX_ENABLED=false
```

Shadow mode keeps the original `fall_detected` event and adds metadata only:

```bash
FALLDATA_AUX_ENABLED=true
FALLDATA_AUX_MODE=shadow
FALLDATA_AUX_THRESHOLD=0.7
FALLDATA_AUX_FALL_CLASS_INDEX=0
FALLDATA_AUX_MIN_NONZERO_FRAMES=30
```

Confirm mode drops pose fall events unless the auxiliary model confirms them:

```bash
FALLDATA_AUX_ENABLED=true
FALLDATA_AUX_MODE=confirm
```

Use confirm mode only after checking field logs in shadow mode.

Deep zip inspection is available but heavy:

```bash
.venv/bin/python scripts/datasets/check_falldata_package.py --root falldata --deep-test-zip
```
