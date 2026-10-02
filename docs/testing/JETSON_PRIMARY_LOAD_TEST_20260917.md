# Jetson 격리 Primary 추론 부하 시험 — 2026-09-17

## 결론

운영 카메라 1대와 다른 서비스를 유지한 상태에서 **추가 재생 채널**을
1 → 2 → 4개로 늘렸다. 추가 4채널은 각각 14.43~14.78FPS로 입력 속도를
따라가지 못했다. 이 조건에서는 4채널 매 프레임 30FPS 목표 미달이다.
단, 운영 부하를 제외한 최대 4카메라 구성의 성능 한계로 일반화할 수 없다.
단일 배치 4 최적화 구성도 아니며 전체 AI 기능이나 상용 합격을 검증하지 않았다.

## 시험 조건

- 사용자 승인: 현재 Jetson GPU 부하 시험 허용, 시험발 실제 장치 명령 차단.
- Jetson AGX Orin, `nvpmodel -q`: `MODE_30W`, mode 2. 전력 모드 변경 없음.
- DeepStream 7.1, TensorRT 10.3.0.
- 기존 운영 AI 서비스는 중단/재시작하지 않았다. 즉 재생 4채널 시 실물 입력까지
  합하면 영상 입력 경로는 5개다. 운영과 시험 경로의 AI 구성도 서로 다르다.
- 입력: `data/fall_test/rerecord_20260914/left_fall_standing_raw_120s.mp4`.
  ffprobe: H.264, 1920×1080, `r_frame_rate=30000/1001`,
  `avg_frame_rate=514000/17139`, 3598 frames, 119.973초.
- 동일 파일을 각 채널에서 별도로 디코딩/추론. `identity sync=true`로 입력 PTS에
  맞춰 공급한다. 재인코딩, 중복 프레임 생성, 추론 결과 복제는 하지 않았다.
  처리량이 부족하면 역압력으로 재생이 뒤처진다. 프레임 유실이나 E2E 지연의
  별도 계측 없이 이 결과를 무손실/저지연 동작이라고 부르지 않는다.
- 운영 Pose 엔진의 input shape는 `(1, 3, 640, 640)`, profile 1개였다.
  배치 1 고정이므로 각각 별도 decoder → mux(batch 1) → nvinfer → fakesink를
  사용했다. 기존 multi-source mux(batch N) 코드 자체의 동작 검증은 아니다.
- 포함: 파일 디코딩, 1080p mux, 전처리, Primary Pose 엔진, 출력 계측.
- 제외: Pose tensor 후처리/낙상 판단, tracker, 안전모/외형/얼굴 보조 모델,
  UI/인코딩/전송, EdgeX 명령 지연, 물리 카메라 재접속, 장시간 안정성.
- 워밍업: 모든 채널 첫 프레임 이후 5초. 이후 실제 monotonic 경과 시간으로
  프레임 카운터 차이를 나눔. 측정 10/20/30초의 초기 진단이며 soak 시험 아님.

## 결과

| 추가 재생 채널 | interval | 측정 시간 | 채널별 실제 tensor 출력 FPS | 판정 범위 |
| --- | --- | --- | --- | --- |
| 1 (대조) | 1 | 10.01초 | 14.98 | 입력 통과 29.97FPS, 추론은 약 절반 |
| 1 | 0 | 20.02초 | 29.96 | 해당 구간에서 입력 속도와 비슷함 |
| 2 | 0 | 30.04초 | 29.23 / 28.96 | 채널별 입력 속도 미달 |
| 4 | 0 | 20.03초 | 14.43 / 14.58 / 14.78 / 14.53 | 채널별 입력 속도 크게 미달 |

interval 0 측정 구간에서는 관측한 각 프레임마다 tensor metadata가 있었다.
관측 구간 이전/이후나 다른 모델의 추론 완료를 의미하지 않는다.
원본이 정확히 30.000FPS 고정 영상이 아니므로 정밀한 30FPS 인수 기준은
별도의 검증 영상/카메라와 허용 오차를 정해 다시 측정해야 한다.

4채널 시험의 관측 온도 최댓값은 54.625°C(노출된 thermal zone 중 최대),
최소 MemAvailable은 약 37.49GiB였다. 12:55:15~12:55:32 KST tegrastats
18개 표본의 시스템 전체 GPU 사용률은 98~99%, 평균 98.78%였다.
이는 GPU 경합/처리 여유 부족을 시사하지만 단계별 프로파일링 없이 유일한
병목으로 확정하지 않는다. 이 GPU 수치에는 운영 서비스도 포함된다.

## 발견한 계측 결함

현재 raw-tensor(`network-type=100`) 경로의 프레임은 tensor metadata가 있어도
`bInferDone=False`로 관측됐다. 따라서 신규 PrimaryFrameMetrics의
`frames_inferred=0`, `frames_skipped=frames_seen`은 실제 추론/생략 수가 아니다.
기존 단위 테스트가 통과해도 이 실장비 동작은 보장하지 못한다.

이번 도구는 Primary 출력 직후 `NVDSINFER_TENSOR_OUTPUT_META`의 존재를
프레임당 한 번 집계한 `tensor_frames`/`tensor_fps`를 별도 기록했다.
시험에는 nvinfer가 채널당 하나뿐이므로 보조 모델 결과가 섞이지 않는다.
실제 운영 계측 교정 시에는 GIE component ID를 확인하고
출력 메타 미설정/확인 불가를 생략으로 오인하지 않도록 처리해야 한다.
이 초기 시험 시점에는 운영 계측 교정/배포를 하지 않았다. **과거 flag 지표로
판정 금지.** 이후의 소스 교정 및 격리 GPU 재검증은 아래 후속 기록 참조.

## 안전 조치와 운영 영향 확인

- 별도 컨테이너 `--network none`, 포트 공개/호스트 네트워크/장치 제어 소켓 없음.
  실행 시작 시 인터페이스가 `lo`뿐인지도 검사했다.
- MQTT/EdgeX 클라이언트와 프로젝트 운영 runner를 시작하지 않았다.
  시험발 전광판·스피커·사이렌 명령을 전달할 네트워크 경로가 없다.
  기존 운영 서비스의 정상 이벤트 처리까지 차단한 것은 아니다.
- 모델/설정/원본 영상/DeepStream 런타임은 읽기 전용 마운트.
  파생 설정/결과만 `data/benchmarks/primary_20260917`에 기록했다.
- 비특권 UID/GID 1000, cap-drop ALL, no-new-privileges, 읽기 전용 rootfs,
  컨테이너 메모리/스왑 합계 제한 8GiB. 시험 종료 후 컨테이너 자동 제거.
- 입력/채널/측정 시간 제한, 첫 프레임 45초 제한, 프레임 정지 5초 감시,
  온도 80°C 이상 또는 호스트 MemAvailable 8GiB 미만이면 예방적 중단.
  80°C는 이번 시험의 보수적 중단값이며 하드웨어 공식 한계 온도라는 뜻이 아니다.
  별도 `timeout -k 10 150`으로 시험 프로세스 전체 시간도 제한했다.
- 운영 AI/API/action-layer 실행 상태 유지, 확인 시 restart count 모두 0.
  AI StartedAt은 시험 전후 `2026-09-16T23:45:46.903505745Z`로 동일.
- 운영 AI 로그: 03:55:07→03:55:27 UTC 프레임 434383→434980,
  약 29.85FPS. 03:56:17→03:56:27에는 436490→436790, 약 30FPS.
  앱의 `frame_dropped=0`만으로 원본 입력 무손실/화면 지연 없음을 증명하지 않는다.
- 엔진/설정/카메라 파일 SHA-256 시험 전후 동일:
  - `models/fall/yolov8n-pose.engine`: `cc6cd0ee490e4f2ae46acf853989cf8c8806aadab47853e39c1e97cdf6732a74`
  - `config/deepstream/config_infer_primary.txt`: `277ab2e76c826804b82fa7c56b9e8edc4a59037fbfe01c23b014c3263e2fa2ae`
  - `cameras.json`: `0b1d8087d0dd9b50eea2cb2d899ab995847d834a7024e3b515349e3ec29a42b5`

초기 실행은 결과 폴더 UID 불일치, 이후 DeepStream 호스트 런타임 마운트 누락으로
추론 시작 전에 실패했다. UID와 읽기 전용 런타임 마운트를 맞춘 후 표의 4회가
정상 종료됐다. 비사용 Rivermax/Triton/Argus plugin 경고와 TRT 엔진 다른 장비
재사용 경고는 남았다. 모든 경고가 해소된 배포 검증으로 표시하지 않는다.

## 결과 파일과 변경 범위

- `scripts/ops/benchmark_deepstream_primary.py`: 신규 격리 Primary-only 시험 도구.
- `tests/test_benchmark_deepstream_primary.py`: 네트워크 차단, 시간/채널 제한,
  원본 설정 보호, 누적값이 아닌 구간 차이 계산, 불완전 범위 표시 검증.
- `docs/testing/MULTICAMERA_30FPS_ACCEPTANCE.md`: flag 계측 한계 경고 및 결과 연결.
- 이 문서. 운영 앱 코드/Compose/전력 모드/카메라 설정은 이번에 변경하지 않음.

원본 결과는 `data/benchmarks/primary_20260917/` 아래에 있다:

- `control_1ch_interval1_v3/report.json`
- `load_1ch_interval0/report.json`
- `load_2ch_interval0/report.json`
- `load_4ch_interval0/report.json`
- `tegrastats_4ch.log`

JSON의 `observed_decode_caps`에는 pad 연결 시 협상 전 capability 범위가 기록됐다.
확정 입력 해상도 증거로 사용하지 않는다. 이번 원본 파일 해상도/FPS는 위 ffprobe
결과로 확인했고 mux 크기는 파생 설정 및 report에 별도 기록했다.

검증: 신규 안전/집계 테스트는 구현 전 12개 실패를 확인한 뒤 모두 통과했다.
연관 테스트 최종 결과는 **125 passed, 5 skipped**. skip 5개는 호스트 Python의
`pyds` 부재로 인한 기존 Jetson 전용 테스트이며, 위 실제 GPU 시험은 별도
DeepStream 컨테이너에서 수행했다. 신규 Python 2개 파일의 Ruff check/format
검사 및 `git diff --check`도 통과했다. 단위 테스트 통과가 앞서 발견한 기존
   flag 계측 결함의 해결을 뜻하지는 않는다.

## 후속 계측 교정과 GPU 재검증

2026-09-17 13:23~13:25 KST에 교정한 계측 클래스를 같은 격리 시험에서
재검증했다. 운영 서비스는 유지했고 후보 모델/배치 크기/전력 모드는 바꾸지 않았다.

교정 내용:

- `PrimaryFrameMetrics.record_batch`가 실제 Primary element의
  `output-tensor-meta`와 `unique-id`를 확인하도록 변경했다.
- 프레임 user meta에서 해당 ID의 tensor 출력만 집계한다. 같은 프레임에
  메타가 중복돼도 한 번만 세며 다른 GIE 출력은 주 추론으로 세지 않는다.
- 메타 출력이 꺼졌거나 속성/메타를 읽을 수 없으면 `frames_inference_unknown`.
  이 경우를 `bInferDone` 값으로 추정해 성공/생략으로 표시하지 않는다.
- 손상된 메타가 있는 프레임은 unknown으로 남기고 이후 프레임 집계를 계속한다.
- 프로세서 probe는 parent nvinfer element를 계측기에 전달한다.
  이번에 변경한 운영 코드 범위는 계측 함수와 그 연결뿐이다. 기존 사용자 변경 유지.
- 시험 JSON은 이제 `metric_evidence=primary_tensor_meta`, `inferred_fps`를
  기록한다. 초기 파일의 `inferred_flag_fps`는 비교 증거로 원형을 보존했다.

| interval | 관측 프레임 | 추론 출력 확인 | 주 추론 출력 없음 | unknown | 수정 inferred FPS | 독립 tensor FPS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 600 | 300 | 300 | 0 | 14.9808 | 14.9808 |
| 0 | 600 | 600 | 0 | 0 | 29.9611 | 29.9611 |

각각 워밍업 5초 후 약 20.03초를 측정했다. 원본 결과:

- `data/benchmarks/primary_20260917/metrics_fix_1ch_interval1/report.json`
- `data/benchmarks/primary_20260917/metrics_fix_1ch_interval0/report.json`

관련 테스트 최종 **132 passed, 5 skipped**, 변경 Python 파일 Ruff check 통과.
5 skip은 호스트 Python의 pyds 부재로 인한 기존 Jetson 전용 테스트다.
회귀 테스트는 false flag + 실제 tensor 존재, GIE ID 격리, 중복 방지,
메타 비활성/조회 실패/손상, 실제 프로세서 probe 연결을 포함한다.
실제 GPU에서는 위 동일 계측 클래스를 시험 도구에서 호출해 검증했다.

이번 교정의 변경 파일: `src/core/_deepstream_primary_metrics.py`,
`src/core/deepstream_processor.py`, `scripts/ops/benchmark_deepstream_primary.py`,
대응 테스트 2개, 이 보고서와 `MULTICAMERA_30FPS_ACCEPTANCE.md`.
Prometheus 필드 이름/운영 설정/모델은 변경하지 않았다.

이 결과는 계측 오류 수정의 검증이지 추론 처리량 개선이나 상용 합격이 아니다.
운영 서비스는 시작 시각 `2026-09-16T23:45:46.903505745Z`, restart count 0을
유지했다. **소스 교정만 완료했고 운영 프로세스 반영/재시작은 하지 않았다.**

## 다음 순서

후속 결과: [배치 1~4 후보 생성·출력 비교·공유 배치 시험](JETSON_POSE_BATCH_CANDIDATE_20260917.md).
초기 level-0 후보는 실행 가능했지만 추가 4채널 각각 5.59FPS로 성능 미달이었다.
운영 엔진은 유지했으며, 아래 후보 최적화와 전체 파이프라인 검증은 아직 남아 있다.

1. Primary tensor 기반 계측 교정/격리 1채널 interval 0/1 재검증 완료.
   운영 반영 및 전체 파이프라인·다중 소스 검증은 별도로 남아 있다.
2. 배치 1 고정 엔진을 그대로 4채널 mux에 넣지 않도록 엔진/배치 계약 검증.
   별도 후보 엔진으로 배치 최적화, 미사용 중간 출력 여부/전송 비용 등을 확인.
   사전 확인 당시 `models/fall/yolov8n-pose.onnx`의 `input`은
   `[batch_size, 3, height, width]`로 동적 축이 있고 출력은 `output` 외
   중간 출력 6개가 더 있다. 엔진과 ONNX의 수치적 동등성, 배치 2/4에서의
   변환 성공 및 정확도/성능은 미검증이었다. 이후 후보 변환/배치 실행 및 짧은
   처리량 측정은 위 후속 보고서에 기록했다. 품질 합격은 여전히 미검증이다.
   원본 엔진은 보존한다.
3. 실물 1대에서 전체 운영 AI의 매 프레임 완료·출력·지연을 계측한 뒤, 같은 모델
   구성으로 전체 4채널 시험. 운영 부하를 제거하려면 별도 중단 승인이 필요하다.
4. 실제 720p 녹화 입력으로 비교하고 정확도/원거리 사람 성능을 함께 검증.
   mux만 720p로 바꾸고 1080p 디코딩을 유지하는 실험은 720p 입력 시험과 구분.
   모델 입력이 계속 640×640이면 영상 해상도 하향만으로 추론 비용이 크게 줄지
   않을 수 있으므로 개선 폭은 아직 미확정이다.

관련 동작 근거:
[NVIDIA nvinfer tensor metadata](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvinfer.html),
[GStreamer identity clock synchronization](https://gstreamer.freedesktop.org/documentation/coreelements/identity.html).
