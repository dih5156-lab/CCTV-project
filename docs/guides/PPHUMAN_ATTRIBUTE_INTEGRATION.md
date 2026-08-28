# PP-Human 속성 모듈 통합 메모

## 현재 구조

현재 외형 분석은 두 경로를 지원합니다.

| 경로 | 용도 | 흐름 |
|---|---|---|
| OpenCV/Python | 개발 PC와 기능 검증 | `YOLO person crop -> AppearanceAnalyzer -> AppearancePipeline` |
| DeepStream SGIE | Jetson 운영 | `PGIE person ROI -> PP-Human/PA100K SGIE tensor -> AppearancePipeline` |

`AppearancePipeline`은 bbox와 keypoint 좌표 스케일을 원본 프레임에 맞추고, track 단위로 색상·성별·boolean 속성을 smoothing한 뒤 `appearance_log`에 저장합니다. crop 영역은 `APPEARANCE_CROP_CONTEXT_RATIO`로 주변 문맥을 포함할 수 있습니다.

적용 원칙:
- 기본 백엔드는 `hsv`로 유지한다.
- `APPEARANCE_BACKEND=pphuman` 설정 시 PP-Human 계열 백엔드를 주입할 수 있다.
- 모델 어댑터가 없거나 모델이 준비되지 않으면 자동으로 HSV 결과로 폴백한다.
- 속성 분석용 bbox는 `APPEARANCE_BBOX_EXPAND_RATIO` 만큼 확장해 가방·하의 누락을 줄인다.
- ONNX Runtime 사용 시 provider는 `TensorRT -> CUDA -> CPU` 순으로 선택한다.
- 라벨 인덱스 해석은 JSON으로 분리해 모델별 출력 순서 차이에 대응한다.
- DeepStream SGIE metadata가 있으면 `attribute_backend=pa100k_sgie` 또는 설정된 SGIE backend 이름으로 저장한다.
- 단일 프레임 결과보다 track 누적 결과를 우선해 색상과 boolean 속성의 흔들림을 줄인다.
- `APPEARANCE_SAVE_CROPS=true`일 때 JPEG는 카메라·track별 대표 이미지 1장만 저장하고, 같은 track의 이후 외형 로그는 해당 경로를 재사용한다.

환경 변수 예시:
- `APPEARANCE_BACKEND=pphuman`
- `APPEARANCE_MODEL_PATH=models/pphuman_attribute.onnx`
- `APPEARANCE_LABEL_MAP_PATH=config/appearance_pphuman_labels.example.json`
- `APPEARANCE_RUNTIME=auto`
- `APPEARANCE_INPUT_SIZE=224`
- `APPEARANCE_SCORE_THRESHOLD=0.5`
- `APPEARANCE_BBOX_EXPAND_RATIO=0.15`
- `APPEARANCE_COLOR_SMOOTHING_WINDOW=12`
- `APPEARANCE_COLOR_MIN_SAMPLES=3`
- `APPEARANCE_BOOL_SMOOTHING_WINDOW=8`
- `APPEARANCE_BOOL_MIN_SAMPLES=3`
- `APPEARANCE_BOOL_TRUE_RATIO=0.6`
- `APPEARANCE_CROP_CONTEXT_RATIO=0.6`

Jetson Orin 권장 순서:
1. 현 구조에서 정확도 검증은 `hsv` 대비 `pphuman` 백엔드 A/B 비교로 진행한다.
2. PoC 단계에서는 Python 어댑터를 붙여 속성 라벨 맵을 검증한다.
3. 배포 단계에서는 현재 지원하는 `YOLO TensorRT + attribute TensorRT SGIE` 2단 엔진으로 정리한다.
4. 최종 운영에서는 Paddle 런타임 전체보다 TensorRT 엔진 단일화 구성을 우선한다.

## TensorRT / DeepStream 전환 메모

`models/pphuman_attribute.onnx`는 ONNX Runtime CPU 세션에서 네이티브 크래시가
발생할 수 있으므로 운영 경로로 사용하지 않는다. Jetson에서는 TensorRT 엔진을
만들어 DeepStream secondary GIE로 붙이는 방향을 우선한다.

확인된 모델 입출력:

- 입력: `x`, `1x3x256x192`
- 출력: `fetch_name_0`, `1x26`

엔진 생성:

```bash
python scripts/convert/convert_onnx_to_engine.py
```

PP-Human용 DeepStream 설정 템플릿:

- `config/deepstream/config_infer_pphuman.txt`

현재 템플릿은 `process-mode=2`, `operate-on-gie-id=1`, `operate-on-class-ids=0`로 person ROI에만 PP-Human을 수행합니다. `fetch_name_0` tensor는 설정된 label map으로 디코딩되어 기존 appearance metadata와 저장 파이프라인으로 전달됩니다.
