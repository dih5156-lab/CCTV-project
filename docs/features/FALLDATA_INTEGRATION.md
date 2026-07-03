# falldata 낙상 보조 검증 통합

## 현재 적용 상태 (2026-07-03 코드 기준)

결론부터 말하면 OpenCV `AIAnalyzer` 경로는 `FallDataAuxVerifier`를 실제로 초기화하고 `shadow`/`confirm` 정책을 적용합니다. DeepStream 경로에는 비동기 검증 큐, shadow 기록, borderline 확인, compare veto 메서드가 구현되어 있지만, 현재 `DeepStreamProcessor` 초기화 코드에서는 관련 객체와 정책 환경변수를 생성하는 연결이 확인되지 않습니다.

따라서 현재 문서 기준은 다음과 같습니다.

| 경로 | 확인된 상태 | 운영 판단 |
|---|---|---|
| OpenCV + YOLO pose | 환경변수 기반 verifier 연결 완료 | `shadow` 우선, 검증 후 `confirm` |
| DeepStream shadow/confirm | 처리 메서드와 테스트 존재, 초기화 연결은 미확인 | 활성화 완료로 간주하지 말고 Jetson 로그와 실제 이벤트 metadata 확인 |
| compare 모델 | 기본 verifier 결과에 비교 결과 기록 가능 | 운영 판정이 아닌 실험·shadow 분석용 |
| compare veto | DeepStream 이벤트 차단 메서드 존재, 초기화 연결은 미확인 | 운영 사용 보류 |

OpenCV 경로의 현재 주요 설정:

| 환경변수 | 기본값 | 설명 |
|---|---:|---|
| `FALLDATA_AUX_ENABLED` | `false` | 보조 검증 활성화 |
| `FALLDATA_AUX_MODE` | `shadow` | `shadow` 또는 `confirm` |
| `FALLDATA_AUX_THRESHOLD` | `0.7` | 낙상 class 확률 임계값 |
| `FALLDATA_AUX_MIN_NONZERO_FRAMES` | `30` | 유효 MediaPipe feature 최소 프레임 수 |
| `FALLDATA_AUX_BUFFER_FRAMES` | `600` | 메모리에 유지할 입력 프레임 수 |
| `FALLDATA_AUX_MAX_EXTRACT_FRAMES` | `120` | 실제 MediaPipe 추출에 사용할 최대 프레임 수 |
| `FALLDATA_AUX_TIMEOUT_SECONDS` | `30` | subprocess 제한 시간 |
| `FALLDATA_AUX_COOLDOWN_SECONDS` | `10` | 연속 검증 억제 시간 |
| `FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE` | `true` | 의존성·프레임·실행 오류 시 원본 알람 유지 |
| `FALLDATA_AUX_COMPARE_MODEL_PATH` | 미설정 | shadow 결과에 함께 기록할 후보 모델 |

아래의 패키지 분석 내용은 현재 모델의 입력 구조와 호환성 판단 근거를 남긴 기록입니다.

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
.venv-falldata/bin/pip install -r requirements/falldata-model.txt
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
.venv-mediapipe/bin/pip install -r requirements/falldata-mediapipe.txt

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
FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true
```

Confirm mode drops pose fall events unless the auxiliary model confirms them:

```bash
FALLDATA_AUX_ENABLED=true
FALLDATA_AUX_MODE=confirm
```

Use confirm mode only after checking field logs in shadow mode.
Confirm mode is fail-open by default for verifier failures such as missing
dependencies, no buffered frames, cooldown skips, or subprocess errors. It only
suppresses fall events when the auxiliary verifier runs and returns
`status=ok, confirmed=false`.

Deep zip inspection is available but heavy:

```bash
.venv/bin/python scripts/datasets/check_falldata_package.py --root falldata --deep-test-zip
```

## Runtime Policy Standard

Use these modes deliberately. Do not mix them without recording the exact
environment variables in the test report.

| Policy | Environment | Behavior | Recommended stage |
| --- | --- | --- | --- |
| Disabled | `FALLDATA_AUX_ENABLED=false` | YOLO/DeepStream fall detector decides alone. | Default operation |
| Shadow | `FALLDATA_AUX_ENABLED=true`, `FALLDATA_AUX_MODE=shadow` | Original fall event is published; falldata result is logged for review. | Field data collection |
| OpenCV confirm | `FALLDATA_AUX_MODE=confirm` on the OpenCV analyzer path | Pose fall event is dropped unless falldata confirms it. | Lab only until recall is proven |
| DeepStream borderline confirm | `FALLDATA_AUX_CONFIRM_BORDERLINE=true` | Only configured borderline DeepStream fall events wait for falldata confirmation. | Controlled pilot |
| Compare veto | `FALLDATA_AUX_COMPARE_VETO_ENABLED=true` | A candidate model can veto confirmed fall events when its result is clearly non-fall. | Lab only |

DeepStream borderline confirm is fail-open by design: if the falldata work queue
is full, the pending marker is removed and the original fall event is published.
This prevents auxiliary verifier overload from silently dropping safety alarms.

## Model Promotion Checklist

Before using a falldata-compatible RF model outside shadow mode, keep the
following artifacts together:

- Training command and git commit.
- `*_metrics.json` from `scripts/datasets/train_falldata_video_rf.py`.
- Manifest readiness output from `scripts/health/check_fall_manifest_readiness.py`.
- `dataset_version`, manifest path, feature cache path, and `max_frames`.
- `holdout_split` with `group_by=scene_base` or a documented reason for another
  grouping.
- `holdout_errors.false_positives` and `holdout_errors.false_negatives` reviewed
  by scene/video.
- `cross_validation.aggregate` confusion matrix.
- Promotion check from `scripts/health/check_falldata_model_report.py`.
- Runtime smoke from `scripts/health/check_falldata_aux.py`.
- Runtime policy check from `scripts/health/check_falldata_aux.py`, including
  fail-open status for confirm mode.

Minimum rule for a field pilot: train/test split must not mix camera variants of
the same scene, and false negatives must be manually reviewed before enabling
any confirm/veto policy.

Promotion check example:

```bash
python scripts/health/check_fall_manifest_readiness.py \
  --manifest data/fall_eval/sample_manifest.jsonl

python scripts/health/check_falldata_model_report.py \
  --metrics-json models/experiments/falldata_sample_rf_metrics.json \
  --require-cross-validation
```

When the report passes, record it in the model manifest:

```bash
python scripts/health/check_falldata_model_report.py \
  --metrics-json models/experiments/falldata_sample_rf_metrics.json \
  --require-cross-validation \
  --update-manifest \
  --model-name falldata_sample_rf
```

`check_falldata_aux.py` validates required paths, isolated venv package versions,
and the synthetic RF smoke. For path/smoke-only troubleshooting, pass
`--skip-version-check`, but do not use that output as a promotion artifact.

After a shadow run, summarize field review records before enabling any blocking
policy:

```bash
python scripts/ops/summarize_fall_shadow_review.py \
  --review-log data/logs/fall_shadow_review.jsonl
```

Use `--strict` in CI or release checks. It exits non-zero when there are no
records, pending unconfirmed fall candidates, aux runtime failures, or parse
errors.
