# Jetson Pose 배치 후보 진단 — 2026-09-17

## 결론

**초기 배치 후보는 운영 채택 보류: 처리량 목표 미달, 정확도 합격 미검증.**
기존 운영 엔진과 설정은 유지했다. 아래 결과는 Jetson 전체의 최종 성능 한계나
배치 방식 자체가 부적합하다는 판정이 아니다.

- 배치 1 고정 엔진과 별도로 배치 1~4를 지원하는 TensorRT 후보를 생성했다.
- 서로 다른 입력 4개에서 후보의 배치 1/2/4 Pose 출력이 동일했다.
  기존 엔진과 후보의 출력은 수치 차이가 있었다. 품질 동등성은 미확정이다.
- 후보로 DeepStream 공유 mux/infer를 실행했지만 추가 4채널은 각각 5.59FPS였다.
- 운영 서비스는 중단하지 않았다. 시험 4채널과 운영 실물 1대가 함께 GPU를 썼다.
- 디코딩 + Primary Pose만 측정했다. 모든 AI의 매 프레임 30FPS 및 상용 인수는
  **NOT_EVALUATED**다.

## 조건과 산출물

기존 시험 조건과 한계는 [Primary 부하 시험](JETSON_PRIMARY_LOAD_TEST_20260917.md)
참조. Jetson AGX Orin의 기존 30W 모드, DeepStream 7.1, TensorRT 10.3.0을
유지했다. 컨테이너 이미지 ID:
`sha256:1ace5b7832a50d6675b66146b0edc307021795732fc7c9a770e2cd628d7fd261`.

산출물 루트:
`data/benchmarks/pose_batch_candidate_20260917.vHdXep/`

| 파일 | 용도 |
| --- | --- |
| `pose_b1_4_fp16_l0.engine` | 운영과 분리된 초기 후보 |
| `layers_l0.json` | TensorRT layer 정보 |
| `paired_tensor_comparison.json` | 동일 입력의 기존/후보 및 배치별 raw tensor 비교 |
| `batched_1ch_interval0/report.json` | 공유 경로 1채널 결과 |
| `batched_2ch_interval0/report.json` | 공유 경로 2채널 결과 |
| `batched_4ch_interval0/report.json` | 공유 경로 4채널 결과 |

각 부하 시험 폴더에는 원본과 분리된 `primary.txt`도 있다.

### 후보 생성

원본 ONNX의 중간 출력 6개를 제거하지 않았다. 입력은 `input`, 최종 Pose 출력은
`output`이며, 기존/후보 엔진 모두 총 출력 7개다. 이번에는 출력 제거까지 동시에
바꾸지 않고 배치 실행 가능 여부를 먼저 확인했다.

- 첫 시도: builder optimization level 3, 외부 600초 제한에서 종료 코드 124.
  엔진 생성 완료가 아니며 운영 모델에 덮어쓰지 않았다.
- 두 번째: level 0, avgTiming 1. 엔진 생성 약 58.75초, trtexec 전체 약 58.97초.
  후보 파일 저장 및 역직렬화에 성공했다.
- min/opt/max: `[1,3,640,640]` / `[4,3,640,640]` / `[4,3,640,640]`.
- `--fp16`을 허용했지만 모든 계층이 FP16이라는 뜻은 아니다. 입출력은 float32다.
- 낮은 탐색 수준으로 만든 기능 확인용 후보다. 충분히 최적화한 배포 엔진이 아니다.
  운영 부하가 있는 상태에서 빌드했으므로 최적 조건의 tactic 탐색도 보장하지 않는다.

격리 컨테이너 내 재현용 빌드 인자:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=/input/model.onnx \
  --saveEngine=/candidate/pose_b1_4_fp16_l0.engine --fp16 \
  --minShapes=input:1x3x640x640 --optShapes=input:4x3x640x640 \
  --maxShapes=input:4x3x640x640 --memPoolSize=workspace:1024 \
  --builderOptimizationLevel=0 --avgTiming=1 --profilingVerbosity=detailed \
  --exportLayerInfo=/candidate/layers_l0.json --skipInference
```

빌드 컨테이너는 네트워크 없음, 원본 읽기 전용, private 결과만 쓰기 허용,
메모리/스왑 합계 12GiB·CPU 4개 제한, 외부 `timeout -k 10 600`을 사용했다.
아래 부하 측정기의 온도/프레임 감시는 빌드 과정에 적용된 것으로 해석하지 않는다.

## 동일 입력의 출력 비교

원본 녹화물 `left_fall_standing_raw_120s.mp4`의 요청 위치 5/15/25/35초에서
각각 프레임을 읽었다. 같은 전처리 결과를 기존 엔진과 후보에 넣었다.
입력 SHA-256을 JSON에 남겼으며 배치 2/4도 이 네 개의 서로 다른 입력을 사용했다.

비교 전처리: 640×640 RGB float32, /255, 비율 유지 resize, 중앙 114 padding.
이는 오프라인 공통 입력 조건이며 DeepStream 전처리와의 수치적 동일성까지
검증한 것은 아니다. 엔진의 다른 출력도 버퍼를 할당하되 `output`만 비교했다.

| 비교 | 최대 절대 차이 | 평균 절대 차이 | person score 최대 차이 |
| --- | --- | --- | --- |
| 기존 batch 1 vs 후보 batch 1 | 8.75 | 0.04353 | 0.003338 |
| 후보 batch 1 vs 후보 batch 2 | 0 | 0 | 0 |
| 후보 batch 1 vs 후보 batch 4 | 0 | 0 | 0 |

모든 Pose 출력은 finite, 형상은 합산 `[4,56,8400]`이었다.
기존/후보 중 하나라도 person score ≥0.25인 proposal 23개에서 최대 절대 차이는
1.5였다. 이 값은 좌표/점수 등의 서로 다른 필드를 포함한 raw 차이이므로
정확도 변화율이나 탐지 오차 픽셀의 단일 합격 기준으로 사용하지 않는다.
`atol=rtol=0.001`의 전체 raw tensor allclose는 기존 vs 후보에서 false였다.
batch 크기에 따른 후보 자체 일관성은 이 네 입력에서만 확인했다.

4개 프레임 비교는 낙상 사건 단위 recall/오경보/시간 지연 평가가 아니다.
35초 입력 하나를 포함했다고 영상의 30초 이후 전체 구간을 평가한 것도 아니다.
NMS 이후 박스/관절 매칭, 낙상 시계열, 야간·가림·원거리 검증은 남아 있다.

## 공유 배치 부하 결과

2026-09-17 13:59~14:02 KST의 짧은 진단이다. 입력은 H.264 1920×1080,
약 29.97FPS 녹화물. 각 채널은 별도 decoder와 source ID를 사용하고, 하나의
`nvstreammux(batch=N) → nvinfer(batch=N)`로 합쳤다. 출력 복제는 하지 않았다.
interval=0, 모든 채널 첫 프레임 이후 워밍업 5초, 측정 약 20.03초씩이다.

| 추가 재생 채널 | 후보 채널별 tensor FPS | 각 채널 추론 프레임 | 최고 관측 온도 | 최소 MemAvailable |
| --- | --- | --- | --- | --- |
| 1 | 19.67 | 394 | 53.468°C | 35.75GiB |
| 2 | 13.08 / 13.08 | 262 | 53.562°C | 35.62GiB |
| 4 | 5.59 / 5.59 / 5.59 / 5.59 | 112 | 53.750°C | 35.50GiB |

세 실행 모두 종료 코드 0, `finished=true`, `errors=[]`. 측정 구간의 관측
프레임마다 해당 Primary의 tensor meta가 있었고 skipped/unknown은 0이었다.
이는 입력 속도를 따라갔거나 원본 프레임 유실이 없었다는 뜻이 아니다.
`identity sync=true`는 최대 공급 속도를 제한하며, 처리 부족 시 재생이 뒤처진다.
카메라→경보 지연이나 큐 지연 분포는 측정하지 않았다.

이전 기존 엔진의 독립 batch-1 경로는 추가 1채널 29.96FPS, 2채널 약
29.23/28.96FPS, 4채널 14.43~14.78FPS였다. 이번 후보는 그 결과보다 낮았다.
다만 엔진/토폴로지/측정 시각이 함께 달라졌다. 운영 부하를 고정한 반복 A/B
시험이 아니므로 감소 원인을 배치 자체 또는 특정 빌드 옵션 하나로 단정하지 않는다.
짧은 단회 결과이며 신뢰구간·장시간 SLA 합격 판정은 제공하지 않는다.

## 안전과 변경 범위

- 시험 컨테이너 `--network none`, 인터페이스 `lo`만 허용.
  운영 runner, MQTT, EdgeX, 장치 클라이언트는 시작하지 않았다.
  시험발 실물 명령 경로는 없으며 기존 운영 서비스의 정상 명령은 별개다.
- 비특권 UID/GID 1000, read-only rootfs, cap-drop ALL, no-new-privileges.
  모델/영상/설정/런타임은 읽기 전용, 후보·결과 디렉터리만 쓰기 가능.
- 각 부하 실행은 메모리/스왑 합계 8GiB, 외부 150초 timeout.
  측정기에서 80°C 이상, 호스트 가용 메모리 8GiB 미만, 프레임 정지 등의 중단
  조건을 유지했다. 모든 경고가 해소된 것은 아니다. 비사용 Rivermax/Triton/
  Argus plugin 경고가 있고 기존 엔진에는 다른 장비 plan 재사용 경고가 있었다.
- 운영 서비스·전력 모드·Compose·카메라 설정·모델을 변경/재시작하지 않았다.
- `benchmark_deepstream_primary.py`: 기존 독립 모드 유지, 공유 배치 모드 추가,
  source ID별 계측으로 혼합 집계를 방지. 대응 테스트 7개 추가.
- `compare_tensorrt_pose_candidate.py`: 격리된 raw 출력 비교 도구 및 테스트 5개 추가.
  형상 불일치/NaN 거부, 수치 차이, 전처리 조건을 검증한다.
- 이 보고서 및 기존 성능 문서의 후속 결과 링크. 운영 API/DB 변경 없음.

SHA-256:

| 대상 | SHA-256 |
| --- | --- |
| 원본 ONNX (유지) | `67ae9087ca8ed813f934660f5323842a96dc2e884e37547862061217b398ee9d` |
| 운영 엔진 (유지) | `cc6cd0ee490e4f2ae46acf853989cf8c8806aadab47853e39c1e97cdf6732a74` |
| 후보 엔진 | `8205a516a93b6892f2437a5c15982f1dc61f579ac6044fc7ebb3ad6f87c625c9` |

### 종료 후 검증

- 최종 연관 회귀 테스트: **146 passed, 5 skipped**. skip은 호스트 Python에
  pyds가 없는 기존 Jetson 전용 테스트이며, 실제 GPU 시험은 위 격리 컨테이너에서
  별도로 수행했다. 테스트 통과가 성능 목표 통과를 뜻하지 않는다.
- 이번 변경 Python 4개 파일 Ruff check/format 검사, `git diff --check` 통과.
- `cctv-pose-` 이름의 시험 컨테이너는 최종 조회에서 남아 있지 않았다.
- AI/API/action-layer는 running, restart count 모두 0. AI 시작 시각은 시험 전과
  같은 `2026-09-16T23:45:46.903505745Z`였다. 실행 상태 확인만으로 UI 지연이나
  전체 운영 기능에 영향이 없었다고 보장하지 않는다.
- 원본 엔진/ONNX의 SHA-256 유지. Primary 설정
  `277ab2e76c826804b82fa7c56b9e8edc4a59037fbfe01c23b014c3263e2fa2ae`, 카메라 설정
  `0b1d8087d0dd9b50eea2cb2d899ab995847d834a7024e3b515349e3ec29a42b5`도 이전과 동일.

검증 명령:

```bash
rtk proxy .venv/bin/python -m pytest tests/test_compare_tensorrt_pose_candidate.py tests/test_benchmark_deepstream_primary.py tests/test_deepstream_primary_metrics.py tests/test_deepstream_pipeline_builder.py tests/test_deepstream_processor.py tests/test_deepstream_event_factory.py tests/test_deepstream_element_config.py tests/test_cctv_metrics.py -q --tb=short
rtk proxy .venv/bin/ruff check scripts/ops/benchmark_deepstream_primary.py scripts/ops/compare_tensorrt_pose_candidate.py tests/test_benchmark_deepstream_primary.py tests/test_compare_tensorrt_pose_candidate.py
rtk proxy .venv/bin/ruff format --check scripts/ops/benchmark_deepstream_primary.py scripts/ops/compare_tensorrt_pose_candidate.py tests/test_benchmark_deepstream_primary.py tests/test_compare_tensorrt_pose_candidate.py
rtk git diff --check
```

## 다음 판단

후속 [출력·빌드 최적화 분리 시험](JETSON_POSE_OUTPUT_OPTIMIZATION_20260917.md)에서
출력 제거 단독 변경과 Softmax FP32 제약은 개선책으로 채택하지 않았다.
계층별 프로파일에서 초기 후보의 DFL Softmax 비용을 확인했으며, 충분히 최적화된
빌드와 품질/전체 파이프라인 검증은 계속 남아 있다.

1. 이 후보를 운영 모델로 승격하지 않는다. 빌드 최적화/중간 출력 비용을 각각
   분리해 프로파일링하고, 동일 부하의 기존 엔진 대조군과 반복 비교한다.
2. 기존과 후보의 후처리 결과 및 30초 이후 낙상 시계열을 같은 입력으로 비교한다.
   raw tensor 차이만으로 합격/불합격 모델 품질을 단정하지 않는다.
3. 유효한 개선 후보가 나온 뒤 전체 운영 AI 구성과 실제 720p 입력을 검증한다.
   운영 중단이 필요한 통제 시험, 모델 교체, 전력 모드 변경은 별도 승인 사항이다.

이번 결과는 [4카메라 인수 기준](MULTICAMERA_30FPS_ACCEPTANCE.md)의 일부 준비
증거이며 실제 4대 UAT 및 전체 AI 매 프레임 30FPS는 여전히 미완료다.
