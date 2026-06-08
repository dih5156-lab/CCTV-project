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

현재 템플릿은 `process-mode=2`, `operate-on-gie-id=1`, `operate-on-class-ids=0`로
person ROI에만 PP-Human을 수행하도록 잡혀 있다. 파이프라인에 붙일 때는
`fetch_name_0` tensor를 `config/appearance_pphuman_labels.example.json` 라벨 맵으로
디코딩해 기존 appearance metadata로 넘기면 된다.
