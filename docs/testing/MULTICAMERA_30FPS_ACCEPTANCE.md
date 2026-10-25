# 다중 카메라 영상·AI 처리율 기준과 계측

작성일: 2026-09-17 · 기준 갱신: 2026-10-07

## 2026-09-17 실장비 시험 및 계측 교정

사용자 승인 후 운영 서비스를 유지한 상태에서 **추가** 1/2/4채널의
디코딩 + Primary Pose 추론 격리 시험을 수행했다.
[실측 결과와 재현 조건](JETSON_PRIMARY_LOAD_TEST_20260917.md)을 먼저 확인한다.
이 시험은 모든 운영 모델을 포함한 4카메라 인수 시험이 아니다.

후속 [배치 후보 시험](JETSON_POSE_BATCH_CANDIDATE_20260917.md)에서 공유 배치
1/2/4 실행을 확인했지만 초기 후보의 처리량은 미달이었다. 운영 모델은 교체하지
않았으며, 이 후보의 생성 성공을 다중 카메라 성능 합격으로 표시하지 않는다.

초기 시험: 현재 `network-type=100` 경로에서는 실제 tensor output이 존재해도
`bInferDone=False`였다. interval 1에서 tensor 약 15FPS, interval 0에서
약 30FPS를 확인했지만 기존 flag 기반 inferred는 둘 다 0이었다.
이후 소스의 계측을 주 추론 component ID에 해당하는 tensor metadata 기반으로
교정했다. 격리 컨테이너의 interval 1/0 재검증에서 각각 14.98/29.96FPS를
집계했고 독립적인 `tensor_fps`와 일치했다. 각 측정 구간은 20초다.
**운영 서비스는 재시작하지 않았으며, 수정 소스의 운영 적용은 아직 안 했다.**
초기 시험 JSON의 `inferred_flag_fps`와 과거 flag 기반 Prometheus 지표는
계속 실제 추론 FPS로 해석하면 안 된다. 아래 지표 설명은 교정 후 소스 기준이다.

## 현재 제품 범위와 시험 목표

- 동시 카메라 최대 4대.
- 10-07 사용자 합의로 AI 실제 추론은 우선 카메라별 약15FPS를 시험 목표로 한다. 영상 입력/출력 FPS와 분리하며, 지연 누적·낙상 품질을 함께 판단한다. 종전30FPS 미달 기록은 당시 기준의 결과로 보존한다.
- 15FPS는 현장 합격 보장이 아니다. 각 모델의 실제 완료·프레임 선택률·시간창을 기록하고 오탐/미탐·경보 지연 기준은 별도로 확정한다.
- 입력 영상은 1920×1080(1080p)을 우선 사용한다. 성능 부족 시
  1280×720(720p)로 낮추는 것은 사용자에게 허용받았다.
- AI 분석 프레임 생략으로 30FPS라고 보고하지 않는다. 합산 120FPS만으로
  특정 카메라의 저성능을 가리지 않는다.
- 입력 해상도와 모델 입력 해상도는 별개다. 입력 영상만 낮춰도 추론 비용이
  동일할 수 있으며, 해상도/모델 변경 후 원거리 사람·자세 품질을 다시 검증한다.
- 해상도 하향은 시험 결과를 보고 적용할 설정 변경이지, 이번에 구현한
  자동 전환 기능이 아니다. 전력 모드도 이번 작업에서 변경하지 않는다.

## 보유 카메라 조건의 시험 계획

10-07 사용자 최신 확인은 **현재 CCTV1대만 연결 가능**이다.
실물2대 동시 시험은 잔여다. [최신 장비 범위](../planning/POC_SCOPE.md)와
[성능 검증 경로](../planning/PERFORMANCE_DECISION.md)를 따른다. 4대 지원 목표는
유지하며 다음 결과를 서로 다른 검증 단계로 기록한다.

| 단계 | 입력 | 확인 가능한 범위 |
| --- | --- | --- |
| 실물 1대 | 실제 RTSP 카메라 | 현재 장비의 입력·추론·출력, 연결·재접속 |
| 재생 2/4채널 | 녹화 영상의 독립 입력 2/4개 | 디코딩·배치·추론 처리량, 카메라별 상태 격리, 자원 사용량 |
| 실물 4대 | 향후 확보한 실제 카메라 4대 | 실제 네트워크·인코더·인증·동시 재접속을 포함한 현장 인수 시험 |

재생 시험의 조건:

- 각 입력에 별도 카메라 ID와 디코딩 경로를 부여하고 같은 운영 모델 구성을
  적용한다. 추론 결과나 이미 디코딩된 화면을 네 번 복제하는 방식은 4채널
  디코딩·추론 시험으로 인정하지 않는다.
- 실제로 초당 30개의 영상 프레임을 공급하는 녹화물을 우선 사용한다.
  파일은 실시간 속도로 재생하고 입력 PTS와 실제 수신 FPS를 확인한다.
  15FPS 영상을 단순 중복해 30FPS 영상 품질을 검증했다고 표시하지 않는다.
- 하나의 영상을 여러 입력으로 사용할 수 있지만, 결과는 반복 입력의 처리량
  시험이다. 독립된 네 현장의 정확도 검증으로 표본 수를 부풀리지 않는다.
  가능하면 서로 다른 구간과 사람 수·가림·움직임이 다른 녹화물을 사용한다.
- 입력 파일 시험과 RTSP 재생 시험을 구분한다. RTSP 송출은 별도 PC가 있으면
  그쪽에서 수행한다. 같은 Jetson을 사용할 때에는 송출/인코딩 부하를 별도
  기록하여 카메라 수신·추론 비용과 혼동하지 않는다.
- 실물 카메라 한 대에 RTSP 연결 네 개를 무작정 열지 않는다. 카메라의 동시
  접속 제한과 송출 부하가 Jetson의 성능 한계처럼 보일 수 있다.
- 시험 ID·이벤트·명령 경로를 운영과 분리해 녹화된 낙상이나 위험 장면이
  실제 전광판·스피커·사이렌을 작동시키지 않게 한다.
- 재생 시험이 통과해도 실물 4대 시험은 미완료로 남긴다. 신규 GPU 부하 시험은
  운영 영향에 대한 승인을 받은 뒤 실행한다. 카메라 수 확인을 시험 실행 승인으로
  간주하지 않는다.

NVIDIA도 녹화 파일을 활용한 DeepStream 성능 평가를 제시한다. 해당 벤치마크의
모델·출력 렌더링 등 조건은 우리 시스템과 다르므로 수치를 그대로 인용해 합격
판정하지 않는다.
[NVIDIA DeepStream 성능 평가 조건](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Performance.html)

## 교정된 계측 구현 범위

Primary GIE의 출력 패드에서 `NvDsInferTensorMeta`를 관측한다.
후속 SGIE의 결과나 추적 결과를 주 추론 완료로 잘못 집계하지 않도록
주 추론기 바로 뒤에 별도 읽기 전용 probe를 연결한다.
실제 element의 `output-tensor-meta` 활성화 여부와 `unique-id`를 확인하고,
그 ID와 일치하는 출력만 프레임당 한 번 집계한다. `bInferDone`에 의존하지 않는다.

| 필드 | 의미 |
| --- | --- |
| `frames_seen` | 주 추론기 출력에서 관측한 프레임 수 |
| `frames_inferred` | 주 추론기 ID와 일치하는 tensor 출력이 있는 프레임 수 |
| `frames_skipped` | tensor 출력이 활성화된 관측 가능한 경로에서 주 추론 출력이 없는 프레임 수 |
| `frames_inference_unknown` | 출력 비활성/설정 조회 불가/잘못된 메타 등으로 추론 여부를 확인할 수 없는 프레임 수 |
| `last_frame_age_seconds` | 마지막 관측 이후 경과 시간. 한 번도 관측하지 못했으면 null |

`frames_skipped`는 주 추론 출력 미관측 수다. 출력 없음의 모든 원인을 interval
정책으로 단정하거나, 입력 영상의 유실 프레임 수와 같다고 해석하지 않는다.

`get_stats()`에 `primary_camera_stats`를 추가했다. 카메라 ID는 등록된 목록으로
제한하며, 소스가 끊긴 등록 카메라도 0 카운터로 남겨 둔다. 파이프라인 재생성 시
카운터를 초기화한다. 계측 예외는 `primary_metrics_errors`에 기록하고 영상 흐름은
중단하지 않는다. 기존 통계 필드는 유지했다.

Prometheus에는 카메라별 Counter와 마지막 관측 경과 시간을 추가했다.
실제 조회는 이 소스 변경이 적용된 AI 프로세스가 실행된 이후에만 가능하다.

```promql
# 교정된 tensor 기반 계측이 적용된 프로세스에서만 사용한다.
# 다른 카메라/Jetson을 합산해 개별 저성능을 숨기지 않는다.
rate(cctv_camera_primary_frames_inferred_total[1m])

# 주 추론기 통과 FPS. 위 추론 FPS와 다른 값일 수 있다.
rate(cctv_camera_primary_frames_seen_total[1m])

# 추론 생략과 근거 부족은 별도 확인한다.
increase(cctv_camera_primary_frames_skipped_total[1m])
increase(cctv_camera_primary_frames_inference_unknown_total[1m])

cctv_camera_primary_last_frame_age_seconds
cctv_processor_primary_metrics_errors_total
```

최소 두 번의 scrape가 없는 경우, 필요한 카메라의 시계열이 없는 경우,
미관측/오래된 관측/추론 확인 불가/계측 오류가 있는 구간은 합격 증거로 사용하지 않는다.
파이프라인 재시작이 있는 구간은 연속 처리 성능 시험으로 승인하지 않는다.

### 이 지표만으로 증명할 수 없는 것

- 주 추론기에서 30FPS가 나와도 낙상 후처리·Pose SGIE·안전모·외형 등
  전체 운영 모델이 매 프레임 완료됐다는 뜻은 아니다.
- 원본 수신/디코딩 이전에 유실된 프레임, 최종 영상 출력 FPS,
  센서/EdgeX/실물 장비의 실행 지연은 측정하지 않는다.
- `last_frame_age_seconds`는 마지막 관측 이후 시간이지 카메라→경보 지연이 아니다.
- 기존 `yolo_postprocess_avg_ms`는 YOLO 후처리 시간이다. 이를 GPU 추론이나
  종단 간 지연시간으로 사용하지 않는다.
- 최초 계측 구현에는 자동 합격 판정, 모델 변경, 현장 배포가 포함되지 않는다.
  이후 승인된 Primary-only 1/2/4채널 부하 시험은 상단 연결 보고서에 별도 기록했다.

NVIDIA는 Primary의 raw tensor 출력이 각 프레임의 `frame_user_meta_list`에
첨부됨을 명시한다. 이 경로의 실제 interval 0/1 동작은 격리 GPU 시험에서
교차 확인했지만, 전체 AI/실물 4카메라 합격을 뜻하지는 않는다.
[NVIDIA nvinfer Tensor Metadata](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvinfer.html#tensor-metadata)

## 다음 검증 순서

1. 장비 명령이 전달되지 않는 격리된 재생 시험에서 신규 probe와 기존 이벤트
   경로의 동작을 확인한다. 운영 컨테이너를 재시작하거나 장비를 작동시키지 않는다.
   같은 Jetson에서 실행하는 GPU 부하 시험도 운영 영상에 영향을 줄 수 있으므로
   별도 시험 장비 또는 승인된 유지보수 시간에 진행한다.
2. Primary interval 0/1 계측 대조는 격리 1채널에서 완료했다. 주 추론 이외
   각 운영 모델의 실행 완료·입력 품질을 별도로 계측하고 운영 적용 후도 확인한다.
3. 실물 1대 기준 시험 후, 1080p 재생 입력을 1채널→2채널→4채널로 늘려
   시험한다. 카메라별 입력·추론·출력 FPS,
   누락, 큐 적체, 지연 분포, CPU/GPU·메모리·온도·전력 모드를 함께 기록한다.
4. 사람이 많은 장면, 야간·가림·원거리, UI 다중 조회, EdgeX/센서 동시 운영,
   단일 카메라 단절·재접속 및 장시간 동작을 포함한다.
5. 4대 1080p가 목표를 충족하지 못하면 병목을 분리한 뒤 720p 조건으로 비교한다.
   성능뿐 아니라 모델 정확도·사건 검출 지연도 다시 평가한다.
6. 양쪽 조건 모두 미달이면 프레임을 숨겨 생략하지 말고 모델/하드웨어/분산 구성을
   다시 결정한다. 시험 길이·카메라당 인원·지연 상한·측정 오차 허용치는 부하 시험 전에
   명시해야 하며, 아직 상용 성능 합격이 확정된 상태가 아니다.

## 변경 파일과 검증

- `src/core/_deepstream_primary_metrics.py`: 독립적인 카메라별 관측 카운터.
- `src/core/_deepstream_pipeline_builder.py`: 주 추론 출력에 선택적 계측 probe 연결.
- `src/core/deepstream_processor.py`: 초기화·probe·통계 노출. 기존 사용자 변경 유지.
- `src/services/processor_metrics.py`: 카메라별 Prometheus 지표 노출.
- `tests/test_deepstream_primary_metrics.py`: 분리 집계, 추론 확인 불가, 미관측,
  재생성, 지표 export, probe 위치 및 계측 오류 격리 테스트.

검증 명령:

```bash
rtk proxy .venv/bin/python -m pytest tests/test_deepstream_primary_metrics.py tests/test_deepstream_pipeline_builder.py tests/test_deepstream_processor.py tests/test_deepstream_event_factory.py tests/test_cctv_metrics.py -q --tb=short
rtk proxy .venv/bin/python -m ruff check src/core/_deepstream_primary_metrics.py src/core/_deepstream_pipeline_builder.py src/core/deepstream_processor.py src/services/processor_metrics.py tests/test_deepstream_primary_metrics.py
```

단위 테스트의 메타데이터 대역은 실제 GPU 추론·RTSP·4대 동시 성능 검증을 대체하지 않는다.
