# Jetson Pose 최적화 빌드 및 재생 지연 진단 — 2026-09-17

후속: [1080p·720p 실제 입력 반복 비교](JETSON_POSE_RESOLUTION_COMPARISON_20260917.md).
후속 원본 재측정에서도 처리량 변동이 관측돼 아래 단회 값을 고정 성능 보장으로 사용하지 않는다.

## 결론

**level 1 후보 빌드 성공, 운영 교체 없음. 4채널 각각 30FPS는 이번 조건에서 미달.**
기존 5분 빌드 제한은 부족했다. 상세 로그를 유지한 새 빌드는 377.226초에 성공했다.
후보 1채널은 60초 동안 약 29.97FPS였지만, 2·4채널에서는 재생 지연이 누적됐다.
이 결과는 모든 AI를 포함하는 제품 인수나 실물 카메라 4대 검증이 아니다.

| 추가 재생 채널 | warmup / 측정 | 채널별 Pose 추론 FPS | 측정 중 재생 지연 증가 | 결과 파일 |
| --- | --- | --- | --- | --- |
| 1 | 5초 / 60.0868초 | 29.9733 | -0.0046초 | `l1_1ch_timing_60s/report.json` |
| 2 | 10초 / 30.0390초 | 각 25.5002 | 각 +4.4867초 | `l1_2ch_timing_30s/report.json` |
| 4 | 10초 / 30.0384초 | 각 20.1409 | +9.8535~9.8545초 | `l1_4ch_timing_30s/report.json` |

모두 finished=true, errors=[]였고, 관측된 Primary 출력 프레임의 skipped/unknown은
0이었다. **skipped=0은 원본 입력 유실 0이나 지연 없는 실시간 처리를 의미하지 않는다.**
4채널 마지막 출력의 재생 지연은 약 15.72~15.75초였다. `last_frame_age_seconds`는
마지막 처리 이후 경과 시간일 뿐이며, 이 적체를 대신 설명할 수 없다.

각 구성의 DeepStream 수치는 단회 진단이다. FPS 개선율·SLA의 95% 신뢰 보장으로
제시하지 않는다. 이전 후보 수치와 비교할 때에도 빌드뿐 아니라 시점과 공유 부하의
영향이 있다. 현재 시점의 엔진 연산 반복 비교는 아래 프로파일 절을 별도로 참조한다.

## 시험 범위와 격리

- Jetson AGX Orin, TensorRT 10.3.0, DeepStream 7.1, 기존 MODE_30W(2) 유지.
- 운영 실물 카메라 1대와 데스크톱/다른 서비스가 실행 중인 상태의 **추가** 재생 부하.
  4채널 시험은 운영 1대에 더한 4개 독립 디코딩 경로이며, 4대 전용 용량이 아니다.
- 입력은 기존 1920×1080 H.264, 30000/1001 FPS, 약 120초 녹화물
  `data/fall_test/rerecord_20260914/left_fall_standing_raw_120s.mp4`.
- 각 채널을 독립 디코딩하고 공유 mux/nvinfer에 batch 1/2/4로 전달한다.
  mux 1920×1080, 모델 입력 640×640, interval=0. 출력 복제나 프레임 생략 없음.
- 디코딩·mux·전처리·Pose Primary tensor 출력만 포함한다. 낙상 시계열 판단,
  NMS/객체 후처리, tracker, 보조 AI, UI, EdgeX 명령/응답 시간은 포함하지 않는다.
- 모든 시험은 `--network none`, read-only rootfs, UID/GID 1000, cap-drop ALL,
  no-new-privileges. 운영 모델/설정/영상은 읽기 전용, 시험 결과 폴더만 쓰기 가능하다.
  앱 runner·MQTT·EdgeX·장치 제어 소켓은 사용하지 않아 시험발 물리 명령 경로가 없다.
- DeepStream 시험은 외부 timeout 150초, 메모리/스왑 합계 8GiB. 스크립트가 온도
  80°C 이상, 호스트 가용 메모리 8GiB 미만, 입력 정지를 감시한다.
  최대 온도는 1/2/4채널 각각 54.750/55.031/55.218°C, 최소 가용 메모리는
  각각 33.382/33.295/32.526GiB였다. 안전 중단 조건은 발생하지 않았다.

## 빌드와 산출물

결과 루트: `data/benchmarks/pose_optimized_build_20260917.A24MGb/`.
이하 파일명은 이 루트를 기준으로 한다.

- 입력: 이전 실험에서 최종 출력만 노출한
  `data/benchmarks/pose_output_ablation_20260917.iyeGAL/pose_output_only.onnx`.
  763개 노드·145개 가중치가 원본과 같은지 확인한 근거는
  [출력 축소 보고서](JETSON_POSE_OUTPUT_OPTIMIZATION_20260917.md)에 있다.
- 빌드 옵션: `--fp16 --minShapes=input:1x3x640x640 --optShapes=input:4x3x640x640
  --maxShapes=input:4x3x640x640 --memPoolSize=workspace:1024
  --builderOptimizationLevel=1 --avgTiming=1 --profilingVerbosity=detailed
  --timingCacheFile=/candidate/dynamic_l1.cache --skipInference --verbose`.
- 격리 빌드 제한: 외부 timeout 1800초, CPU 4개, 메모리/스왑 합계 12GiB.
  timeout을 기다려 강제 종료한 것이 아니라, 14:46:38 KST 정상 종료 코드 0을 확인했다.
  빌드에는 DeepStream 시험의 자동 온도 중단 로직을 적용했다고 주장하지 않는다.
- `dynamic_l1_build.log`: 전체 진행 로그. 계층별 autotuning 진행 후 생성 완료.
  `Engine built in 377.226 sec`, 크기 9.23619MiB, 역직렬화 성공.
- 엔진: `pose_output_only_l1_dynamic.engine`, SHA-256
  `d417343ad0bf183747c846e3e7fdee737a27526be62b0a327b2b75d6ee06a6b4`.
- `dynamic_l1_layers.json`, `dynamic_l1.cache`: 계층 정보와 후속 빌드용 timing cache.
- 사용 이미지 ID:
  `sha256:1ace5b7832a50d6675b66146b0edc307021795732fc7c9a770e2cd628d7fd261`.

이전 300초 제한의 종료 코드 124 기록은 그대로 보존한다. 이번 성공은 새 실행의
결과이며, 과거 실패 기록을 덮어쓰거나 도구 응답 대기 만료를 빌드 실패로 취급하지 않았다.

## 수치 비교 — 품질 인수 아님

`paired_production_vs_l1.json`에서 녹화물 요청 위치 5/15/25/35초의 동일 입력 4개를
운영 엔진 사본과 비교했다. 640 RGB float32 /255, 비율 유지·114 padding 전처리다.

- 기존 엔진 대비 raw 최대 차이 6.125, 평균 0.041716, p99 0.5.
- person score 최대 차이 0.002751. score≥0.25인 proposal 합집합 23개의 최대 차이 0.75.
- 전체 allclose(atol=rtol=0.001)는 false. 후보 내부 batch 1 대 2/4 차이는 모두 0.
- NaN/형상 오류는 없었지만, 네 프레임의 수치 비교를 낙상 정확도 합격으로 간주하지 않는다.
  30초 이후 전체 시계열, 정답 라벨, NMS 및 낙상 판단 비교는 여전히 필요하다.

## 초기 몰림을 구분하기 위한 시험 도구 보강

최초 `l1_1ch_run1/report.json`의 20초 구간은 915개 추론, 45.6816FPS였다.
입력 약 29.97FPS를 넘고 시작 지연이 관측되어, 누적 프레임을 따라잡는 구간일
가능성이 있다. 이 실행에는 PTS 계측이 없어 정확한 지연 변화는 소급 확정하지 않는다.
이 값을 실시간 처리 합격 근거로 채택하지 않았다.

`scripts/ops/benchmark_deepstream_primary.py`에 시험용 관측만 추가했다.

1. 각 source ID의 `NvDsFrameMeta.buf_pts`와 해당 Primary의 clock-base_time을 기록.
2. 측정 구간의 영상 시간 진행량/실제 경과 비율 및 시작·끝·증가 지연을 JSON에 표시.
3. 유효한 PTS 0과 미상 값/uint64 sentinel/역행을 구분. 미상 값을 0초 지연으로 표시하지 않음.

GStreamer의 running time과 clock/base-time 관계는
[공식 동기화 설명](https://gstreamer.freedesktop.org/documentation/application-development/advanced/clocks.html)을 따른다.
여기서 `output_lag`는 **seek/loop 없는 이 로컬 재생의 PTS와 파이프라인 clock 차이**다.
카메라 촬영 시점~AI~UI/장치까지의 E2E 지연이 아니며, 임의 SEGMENT offset/rate를
가진 스트림의 일반적인 지연 계측기로 사용하지 않는다. 재생 속도/프레임 정책은 바꾸지 않았다.

회귀 테스트에서 몰림(실제 20초에 영상 30초 진행), 잘못된 PTS, PTS 역행 및
채널별 기록을 검증했다. 새 테스트 6개 실패를 먼저 확인한 뒤 최소 구현을 추가했다.
관련 비교/부하/계측 테스트 **53 passed**, 변경 스크립트·테스트 Ruff check 통과.
운영 앱·API·DB·Compose·모델/카메라 설정을 수정한 것은 아니다.

`tegrastats_4ch.log`는 15:11:48~51 KST 표본이다. 수집 시점이 4채널 실행과
정확히 겹침을 보장하지 않으므로 4채널 GPU 점유율이나 최대 온도의 근거로 쓰지 않는다.
위 온도/메모리 표는 해당 시험 자체의 `report.json` 표본에서 계산했다.

## 같은 ONNX의 level 0 / level 1 반복 프로파일

빌드 최적화 수준별 엔진을 L1→L0→L1→L0→L1→L0 순서로 3회씩 실행했다.
각각 `--shapes=input:4x3x640x640 --warmUp=1000 --duration=5 --iterations=10
--separateProfileRun --dumpProfile --exportProfile=... --exportTimes=...`을 사용했다.
network-none 컨테이너, 외부 timeout 90초, 메모리/스왑 합계 8GiB다. 6회 모두
종료 코드 0과 trtexec PASSED를 확인했다. 엔진을 반복 간 재빌드하지 않았다.

| 엔진 | GPU compute 실행별 평균(ms/batch) | 실행 평균들의 평균 | 참고용 95% t 구간 | Softmax 계층 평균 범위 |
| --- | --- | --- | --- | --- |
| 1출력 level 0 | 179.779 / 169.236 / 171.399 | 173.471ms | 159.638~187.304ms | 49.729~53.277ms |
| 1출력 level 1 | 70.551 / 67.361 / 66.429 | 68.114ms | 62.745~73.483ms | 7.935~9.229ms |

근거 파일: `l0_b4_control{1,2,3}_{times,profile}.json`,
`l1_b4_run{1,2,3}_{times,profile}.json`, 각 `_profile.log`,
`repeated_l0_l1_summary.json`. 구간은 실행 단위 평균 3개에 대해
`mean ± t(0.975, df=2) × sample_sd / sqrt(3)`로 계산했다.
한 실행 안의 query 수를 독립 반복 수로 부풀리지 않았다.

이 구간은 소수 반복의 편차를 기술하는 참고치다. 공유 GPU/CPU 부하·클럭을
고정하지 않았고, L1 세 번째 실행에는 짧은 CPU 회귀 테스트도 겹쳤다.
실행 독립성·정규성·부하 동일성을 보장하지 못하므로 **제품 SLA, 정확한 인과적
개선율 또는 95% 성능 보장으로 해석하지 않는다.**

이 프로파일은 trtexec의 임의 입력이며 실제 영상/디코딩 없는 엔진 진단이다.
표의 batch GPU compute와 DeepStream 채널별 FPS를 단순 역수로 변환해 비교하지 않는다.
두 엔진 모두 aux stream 1개를 사용하므로 계층 시간 합은 GPU wall time이 아니다.

반복 관측상 L1의 GPU 연산 시간은 L0보다 작았다. D2H는 두 엔진 모두 약 0.63ms로
비슷했고, 기존 주요 병목인 DFL Softmax가 크게 줄었다. 해당 tactic은 L0의
`0x48c115a824ac468d`에서 L1의 `0x4c311f19c8424918`로 바뀌었다.
L1에서도 Softmax는 profile 합산의 12.54~13.69%로 최대 단일 계층이었으며,
다른 convolution/transpose 계층 비용도 남아 있다. 전체 개선을 Softmax 하나의
인과적 효과로 확정하지 않으며, **출력 제거보다 빌드 연산 선택이 중요한 개선 지점**이라는
근거로 사용한다. 이 후보 자체도 4채널 실시간 목표에 미달하므로 배포하지 않았다.

## 최종 확인과 다음 작업

- 관련 53개 테스트 재실행 통과, 변경 스크립트/테스트 및 기존 비교 도구 Ruff 통과.
  GPU에서 추가 시각 계측을 포함한 1·2·4채널 실행도 모두 정상 종료했다.
- AI/API/action-layer는 healthy, running, restart count 0이며 기존 StartedAt 유지.
  AI는 `2026-09-16T23:45:46.903505745Z`. 이것이 UI 지연 무영향을 뜻하지는 않는다.
- MODE_30W(2) 유지. 원본 ONNX/운영 엔진/Primary 설정/카메라 설정의 SHA-256은
  시험 전후 동일하다. 삭제·배포·운영 재시작·전력 모드 변경·시험발 장치 명령 없음.
- 추가 시각 계측 스크립트 SHA-256:
  `0bd3fb9c0de476daf1c2418db5b23e88f522396225b1c4b091684b85681fc3dc`.
- 이번 수정은 시험 스크립트와 테스트, 이 보고서와 이전 보고서의 후속 링크다.
  private 후보 엔진/로그/결과도 보존한다. 운영 소스 변경이나 후보 승격은 없다.

다음 우선순위:

1. 남은 연산 비용을 대상으로 더 높은 최적화 수준/고정 batch 4 진단을 별도 후보로
   비교한다. 고정 batch만 성공해도 일반적인 동적 카메라 운영 지원 완료는 아니다.
2. 실제 720p 파일 입력과 전체 AI 경로를 단계적으로 시험한다. mux 크기만 줄이거나
   프레임을 생략해 목표를 충족했다고 하지 않는다. 모델 입력 크기를 유지하면
   영상 해상도 축소만으로 모델 연산 비용까지 같은 비율로 줄어드는 것은 아니다.
3. 후보 채택 전 30초 이후 낙상 시계열과 정상 행동의 오경보/NMS·키포인트 변화,
   장시간 반복, 실물 1대 및 추후 4대, EdgeX/장치 UAT를 별도로 통과시킨다.
   운영 서비스 중단·전력 모드 변경·모델 배포는 별도 승인 대상이다.
