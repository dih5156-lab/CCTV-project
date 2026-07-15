# 상용 얼굴 모델 운영 절차

YuNet(MIT)과 SFace(Apache-2.0)를 Jetson TensorRT 10.3에서 실행한다. InsightFace pretrained weight는 이 경로에 포함하지 않는다.

## 요구 환경

- JetPack 6.2 / TensorRT 10.3.x
- Python TensorRT binding 10.3.x
- NumPy 1.x (`numpy<2.0`)
- `TENSORRT_EXPECTED_VERSION=10.3`

Python binding이 10.3.x가 아니면 engine deserialize 전에 명시적 오류로 중단한다. pip의 최신 `tensorrt` 패키지로 시스템 binding을 덮어쓰지 않는다.

## 모델 설치와 엔진 생성

```bash
python scripts/models/fetch_commercial_face_models.py
python scripts/convert/convert_commercial_face_models_to_engine.py --model all
python scripts/health/check_commercial_face_models.py
```

모델과 LICENSE는 고정 revision 및 SHA-256으로 검증한다. `.engine` 파일은 생성한 Jetson의 TensorRT/CUDA 조합에 종속되므로 Git이나 다른 장치로 복사하지 않는다.

## 통합 smoke test

얼굴이 한 명 이상 있는 검증 이미지를 지정한다.

```bash
python scripts/smoke/smoke_test_commercial_face_tensorrt.py \
  --image /path/to/face-test.jpg
```

성공 조건은 얼굴 한 명 이상 검출, 모든 임베딩 shape `(128,)`, finite 값, L2 norm 1이다. 이 검사는 모델 실행 검증이며 실제 동일인/타인 정확도나 운영 threshold를 보장하지 않는다.

## 현재 Jetson POC 결과

- TensorRT: 10.3.0
- YuNet engine: 약 609 KiB, 랜덤 입력 평균 GPU compute 약 18.08 ms
- SFace engine: 약 19.2 MiB, 랜덤 입력 평균 GPU compute 약 10.88 ms
- 공식 YuNet 예제 이미지: 얼굴 28개, 모든 SFace 임베딩 finite 및 unit norm
- GPU 클럭을 고정하지 않아 latency 변동성이 있으며, engine은 안정적 POC 빌드를 위해 builder optimization level 0으로 생성했다.

## 롤백

새 backend를 운영에 연결하기 전까지 기존 설정을 유지한다.

```bash
FACE_RECOGNITION_BACKEND=opencv
```

얼굴 TensorRT 준비 실패는 얼굴 기능만 unavailable로 처리해야 하며 DeepStream 낙상·헬멧·외형 분석을 중단시키면 안 된다.
