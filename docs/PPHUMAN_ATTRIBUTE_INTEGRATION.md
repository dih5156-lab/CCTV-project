# PP-Human 속성 모듈 통합 메모

현재 구조는 `YOLO person crop -> AppearanceAnalyzer` 직렬 파이프라인이다.

적용 원칙:
- 기본 백엔드는 `hsv`로 유지한다.
- `APPEARANCE_BACKEND=pphuman` 설정 시 PP-Human 계열 백엔드를 주입할 수 있다.
- 모델 어댑터가 없거나 모델이 준비되지 않으면 자동으로 HSV 결과로 폴백한다.
- 속성 분석용 bbox는 `APPEARANCE_BBOX_EXPAND_RATIO` 만큼 확장해 가방·하의 누락을 줄인다.
- ONNX Runtime 사용 시 provider는 `TensorRT -> CUDA -> CPU` 순으로 선택한다.
- 라벨 인덱스 해석은 JSON으로 분리해 모델별 출력 순서 차이에 대응한다.

환경 변수 예시:
- `APPEARANCE_BACKEND=pphuman`
- `APPEARANCE_MODEL_PATH=models/pphuman_attribute.onnx`
- `APPEARANCE_LABEL_MAP_PATH=config/appearance_pphuman_labels.example.json`
- `APPEARANCE_RUNTIME=auto`
- `APPEARANCE_INPUT_SIZE=224`
- `APPEARANCE_SCORE_THRESHOLD=0.5`
- `APPEARANCE_BBOX_EXPAND_RATIO=0.15`

Jetson Orin 권장 순서:
1. 현 구조에서 정확도 검증은 `hsv` 대비 `pphuman` 백엔드 A/B 비교로 진행한다.
2. PoC 단계에서는 Python 어댑터를 붙여 속성 라벨 맵을 검증한다.
3. 배포 단계에서는 `YOLO TensorRT + attribute TensorRT` 2단 엔진으로 정리한다.
4. 최종 운영에서는 Paddle 런타임 전체보다 엔진 단일화 구성을 우선한다.
