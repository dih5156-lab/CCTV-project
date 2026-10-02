# Jetson Pose 출력·빌드 최적화 분리 시험 — 2026-09-17

후속: [level 1 빌드 완료 및 재생 지연 진단](JETSON_POSE_OPTIMIZED_BUILD_20260917.md).
아래 5분 빌드 제한 실패는 당시 실행의 기록이며, 후속 실행의 성공과 구분한다.

## 결론

**운영 교체 없음, 4채널 30FPS 목표는 여전히 미달/미검증.**
출력만 줄인 후보는 DeepStream 4채널 각각 5.79FPS였다. 반복 엔진 진단에서
DFL Softmax가 6회 모두 가장 큰 단일 계층이었으나, FP32 제약만 주는 방법도
해결책이 아니었다. level 1 빌드는 5분 제한으로 미완료다.
실험 산출물과 실패한 가설까지 보존하며, 다음 우선순위는 충분한 시간의 빌드
최적화와 고정/동적 batch에 따른 연산 선택 분리다. 전체 모델·현장 UAT는 남아 있다.

## 범위

[초기 배치 후보](JETSON_POSE_BATCH_CANDIDATE_20260917.md)의 4채널 5.59FPS
미달을 조사한다. 이번 실험에서 바꾸는 변수는 출력 노출 수와 TensorRT 빌드
최적화 수준이다. 운영 모델 교체·전력 모드 변경·서비스 중단은 하지 않는다.
원본 모델의 학습 가중치, 연산 그래프, 입력 640×640, FP16 허용 조건은 유지한다.

성능 수치는 운영 실물 1대 및 데스크톱/기타 서비스가 유지되는 Jetson의 **추가**
재생 부하다. 최대 4대 전용 환경이나 모든 운영 AI의 합격 시험으로 일반화하지 않는다.
실행 중인 다른 프로세스를 임의 종료하지 않았다.

## 가설과 변경

1. 중간 출력 6개의 노출이 불필요한 비용을 만드는가?
   `src/core/_deepstream_tensor_utils.py`의 `select_yolo_output`과 Pose 소비 경로는
   최종 `output`을 선택한다. 원본 엔진은 최종 결과 외 중간 출력 6개도 포함한다.
   실제 비용이 전송인지 연산/융합인지 별도 프로파일 없이 단정하지 않는다.
2. 빠른 빌드용 optimization level 0이 낮은 처리량에 영향을 주는가?
   출력 하나인 같은 ONNX에서 level 0과 1을 비교한다. TensorRT는 빌드 탐색
   수준이 빌드 시간과 엔진 성능에 영향을 줄 수 있다고 설명하지만, 개별 엔진의
   개선은 실측해야 한다. [NVIDIA 설명](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html#builder-optimization-level)

## 재현 가능한 산출물

결과 디렉터리: `data/benchmarks/pose_output_ablation_20260917.iyeGAL/`.

- `pose_output_only.onnx`: 원본을 덮어쓰지 않은 시험용 사본.
- `pose_output_only.manifest.json`: 소스/사본 해시, ONNX 버전, 불변성 확인.
- `pose_output_only_l0.engine`, `layers_output_only_l0.json`, `build_output_only_l0.log`.
- `paired_all_vs_output_only_l0.json`: 7출력 level 0과 1출력 level 0의 raw 출력 비교.
- `output_only_l0_4ch_run1/report.json`: 출력 제거 단독 변경의 첫 부하 진단.

ONNX 1.17.0의 `onnx.utils.extract_model(input, output, ['input'], ['output'],
check_model=True)`을 사용했다. API는 [ONNX 문서](https://onnx.ai/onnx/api/utils.html)
참조. 사용 중인 버전의 함수 시그니처를 직접 확인했고 별도 라이브러리를 설치하지
않았다. 사본의 763개 연산 노드와 145개 initializer가 원본과 순서/직렬화 바이트까지
동일한지 assert로 검증했다. 원본 SHA-256이 추출 전후 동일함도 확인했다.

| 대상 | SHA-256 |
| --- | --- |
| 원본 ONNX | `67ae9087ca8ed813f934660f5323842a96dc2e884e37547862061217b398ee9d` |
| 최종 출력만 남긴 ONNX | `98b8711e91ac481545a3d8f661839b36276c84b634ba4680e97a1eb8f3f019bc` |
| 7출력 level 0 엔진 (대조) | `8205a516a93b6892f2437a5c15982f1dc61f579ac6044fc7ebb3ad6f87c625c9` |
| 1출력 level 0 엔진 | `54226e0dc21b3155b8d5ed44ae8e6d2ccfd8b8e22c8c19a8b1d1eeb36e2e7bd5` |

### 빌드 조건

TensorRT 10.3.0. `--fp16`, workspace 1024MiB, avgTiming 1,
profilingVerbosity detailed, skipInference. min/opt/max 입력은
1×3×640×640 / 4×3×640×640 / 4×3×640×640이다.
level 0 사본은 엔진 생성 63.8764초, trtexec의 `Engine built in` 64.1089초,
크기 12.603MiB로 생성·역직렬화 성공했다. 내부 형태/메모리 선택이 달라질 수 있어
출력 수를 줄이면 엔진 파일 크기도 반드시 줄어드는 것은 아니다.

빌드 컨테이너: network none, read-only rootfs, UID/GID 1000, cap-drop ALL,
no-new-privileges, 메모리/스왑 합계 12GiB, CPU 4개 제한, 외부 300초 timeout.
원본/런타임은 읽기 전용, 결과 폴더만 쓰기 가능하다. 실행 로그는 pipefail을 켠
tee로 기록해 로그 저장 성공이 빌드 실패를 가리지 않게 했다.

## 출력 제거 단독 변경의 첫 결과

서로 다른 동일 입력 4개(녹화물 요청 위치 5/15/25/35초)에 대해 비교했다.
비교 전처리와 제한은 기존 후보 보고서와 같다. 7출력 level 0 대조군 vs 1출력
level 0의 raw 최대 차이는 2.4375, person score 최대 차이는 0.00198457이었다.
person score ≥0.25인 proposal 합집합 23개의 최대 차이는 0.25였다.
전체 raw allclose(atol=rtol=0.001)는 false. 새 후보 자체의 batch 1/2/4 결과는
이 네 입력에서 동일했다. NaN/형상 오류는 없었으며 정확도 합격은 미평가다.

DeepStream 4채널, 공유 batch 4, interval 0, 모든 채널 첫 프레임 이후 warmup 5초,
측정 20.0276초에서 각 채널 116개 tensor 출력, **각각 5.7920FPS**였다.
finished=true, errors=[], 관측된 구간의 skipped/unknown=0. 온도 최댓값
54.062°C, 최소 MemAvailable 34.934GiB. 이전 7출력 level 0의 5.5922FPS와
단회 수치가 다르지만 시점/운영 부하가 통제되지 않아 개선율로 확정하지 않는다.
확인 가능한 결론은 **출력 제거만 적용한 이 후보도 30FPS에 크게 미달**한다는 것이다.

## 공통 측정 제한과 안전

- DeepStream 7.1, 기존 30W 모드 및 이미지 ID 유지.
- 1080p H.264 약 29.97FPS 녹화물을 입력별로 독립 디코딩. 원본은 이전과 같은
  `data/fall_test/rerecord_20260914/left_fall_standing_raw_120s.mp4`.
- 범위는 디코딩·mux·전처리·Primary Pose·tensor 메타 계측까지다. 낙상 판단,
  보조 모델, UI 출력, EdgeX 지연, E2E 적체 및 원본 프레임 유실은 미평가다.
- `benchmark_deepstream_primary.py`와 `compare_tensorrt_pose_candidate.py`는 수정
  없이 재사용했다. 운영 코드/API/DB/Compose/카메라 설정 변경 없음.
- 부하 시험은 network none, 장치 제어 소켓/클라이언트 없음. 기존 운영 명령과
  시험발 명령을 구분하며 시험이 실장치를 작동시킬 네트워크 경로는 없다.
- 부하 측정기: 최대 150초 timeout, 메모리/스왑 합계 8GiB, 온도 80°C 이상 또는
  호스트 가용 메모리 8GiB 미만·프레임 정지 시 중단. 이 감시를 빌드에 적용했다고
  표시하지 않는다. 빌드는 별도 시간/자원 제한 및 수동 tegrastats 표본 확인이다.
- 비사용 plugin/Argus 경고는 여전히 존재한다. 비교 도구의 default CUDA stream
  경고가 있어 raw 비교 실행 시간을 추론 성능으로 사용하지 않는다.

## 빌드 수준 1의 결과와 계층별 프로파일

level 1 빌드는 2026-09-17 14:15:33 KST에 시작했고 외부 300초 제한으로
종료 코드 124가 됐다. 이는 관측 도구의 대기 만료가 아니라 빌드 프로세스의
실제 제한 종료다. `build_output_only_l1.log`를 보존했으며 엔진 파일은 없다.
level 1 성능이 나쁘다거나 모델 파싱 오류라는 결론은 낼 수 없다.
실행 중 메모리 표본은 약 1.438GiB/12GiB로, OOM 증거는 없었다.

동일 장치에서 기존 엔진과 후보를 별도 network-none 컨테이너의 trtexec로
프로파일링했다. `--warmUp=1000 --duration=5 --iterations=10 --separateProfileRun
--dumpProfile --exportProfile=... --exportTimes=...`을 사용했다.
실제 영상이 아닌 trtexec 임의 입력이며, 디코딩/DeepStream/후처리 없는 엔진
진단이다. GPU compute는 공유 GPU에서 다른 작업과 경합한 영향도 포함한다.

| 엔진 | batch | GPU compute 평균 | D2H 평균 | DFL Softmax 평균 (별도 profile) |
| --- | --- | --- | --- | --- |
| 기존 운영 엔진 사본 | 1 | 17.8967ms | 0.6304ms | 0.8504ms |
| 7출력 level 0 후보 | 1 | 48.2067ms | 0.7727ms | 11.4230ms |
| 7출력 level 0 후보 | 4 | 145.376ms | 2.6261ms | 43.7538ms |
| 1출력 level 0 후보 | 4 | 150.335ms | 0.6345ms | 45.8285ms |

Softmax는 batch 4 두 후보에서 layer profile 합산 시간의 약 30%로 가장 큰
단일 계층이었다. 엔진 inspector의 해당 계층은 `CaskSoftMaxV2`, 입력 `[-1,16]`,
Axes=2, tactic `0x48c115a824ac468d`였다. level 0 두 후보의 이 값은 같았다.
기존과 후보는 빌드/프로파일 계약도 달라져 정밀도나 dynamic batch 하나를 단독
원인으로 확정하지 않는다. profile은 별도 실행이며 일부 후보는 aux stream을
사용하므로 계층 시간 합을 GPU wall time이나 전체 지연과 동일시하지 않는다.

이 결과는 **출력 전송만 줄이는 것으로는 큰 처리량 부족을 설명/해결할 수 없고,
Softmax를 포함한 엔진 연산 경로의 최적화를 우선 조사해야 한다**는 근거다.
TensorRT가 GPU compute 변동계수 약 20~37% 경고를 출력했다. 짧은 단회 값의
차이를 개선율이나 95% 신뢰 SLA로 표시하지 않는다. 운영 부하/클럭을 임의로
변경하지 않았으며 고정 부하에서의 최종 비교는 별도로 필요하다.

프로파일 파일 접두사: `production_b1`, `all_outputs_l0_b1`,
`all_outputs_l0_b4`, `output_only_l0_b4`. 각각 `_profile.log`, `_profile.json`,
`_times.json`이 결과 디렉터리에 있다.

## Softmax FP32 진단 후보 — 채택 안 함

동일한 1출력 ONNX, level 0 빌드에 다음 제약만 추가한 후보를 만들었다.

```text
--precisionConstraints=obey
--layerPrecisions=/model.22/dfl/Softmax:fp32
--layerOutputTypes=/model.22/dfl/Softmax:fp32
```

계층 정밀도 제약은 [NVIDIA trtexec 문서](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/command-line-programs.html)
참조. 설치된 TensorRT 10.3.0 로그가 해당 계층에 fp32 제약을 적용했음을 표시했다.
실행 컨텍스트의 inspector도 Softmax 입력/출력이 `[134400,16]`의 FP32임을 확인했다.
이 명시적 제약은 하나의 계층을 대상으로 하지만 재빌드의 다른 fusion/tactic까지
동일함을 보장하지는 않는다. 운영 파일은 바꾸지 않았다.

- `pose_output_only_l0_softmax_fp32.engine`: SHA-256
  `3ac78340df3c7cf253d7dd984ed45b668c54a4aad48d0fee989539cd7b37a5bf`.
- `build_output_only_l0_softmax_fp32.log`: 생성/역직렬화 성공. 엔진 생성 60.1328초,
  `Engine built in` 60.3605초, 크기 11.2731MiB.
- `paired_softmax_fp32.json`: 앞선 1출력 level 0 후보 대비 raw 최대 차이 2.4375,
  person score 최대 차이 0.00198457, foreground 최대 차이 0.25. batch 1/2/4는
  같은 네 입력에서 일치. 품질 인수는 미평가다.
- `softmax_fp32_b4_*`: 실제 batch 4 프로파일/타이밍/레이어 정보.
  GPU compute 평균 151.484ms, D2H 평균 0.6241ms, throughput 6.40159 batch/s.
  Softmax 평균 49.7043ms, profile 합산 중 32.6738%. tactic 값은 이전과 같다.

이 단회 진단에서는 엔진 시간과 Softmax 병목이 줄지 않았다. 따라서 FP32 제약을
운영 변경으로 채택하지 않았고 이 후보의 DeepStream 추가 시험은 실행하지 않았다.
정밀도 한 가지 변경이 해결책이라는 가설을 뒷받침하지 못했다는 결과를 보존한다.

## 대조군/출력 축소 각 3회 반복

동일한 두 엔진 파일을 번갈아 각각 총 3회 프로파일링했다. 첫 회는 위 단회 진단,
추가 실행은 파일 접두사에 `_run2`, `_run3`가 붙는다. 반복 간 생성/재빌드는 없다.
각 실행은 엔진 내 배치 4의 5초 이상 trace와 별도 layer profile이다.
비교 과정에 실제 녹화 입력/DeepStream FPS와 임의 입력 trtexec batch/s를 섞지 않는다.

| 엔진 | 각 실행 GPU compute 평균(ms/batch) | 실행 평균들의 평균 | 참고용 95% t 구간 | D2H 실행 평균들의 평균 |
| --- | --- | --- | --- | --- |
| 7출력 level 0 | 145.376 / 164.131 / 185.956 | 165.155ms | 114.704~215.605ms | 2.617ms |
| 1출력 level 0 | 150.335 / 148.225 / 166.333 | 154.964ms | 130.366~179.562ms | 0.635ms |

산출 근거: `repeated_profile_summary.json`과 각 `_times.json`.
단일 trace의 모든 query를 독립 반복처럼 세지 않고 실행 단위 평균 3개를 사용했다.
구간은 `mean ± t(0.975, df=2) × sample_sd / sqrt(3)`이다.
표의 구간은 **소수 표본의 편차를 기술하는 참고치**이며, 실행 독립성/정규성 및
고정 운영 부하를 보장하지 못한다. 드리프트·상관도 가능하므로 제품 SLA의 95%
신뢰 보장으로 해석하지 않는다. 평균 수치만으로 출력 축소의 전체 연산 개선을
확정하지 않는다. 출력 전송은 줄었지만 큰 처리량 부족의 해결 근거는 부족하다.

6회 모두 `/model.22/dfl/Softmax`가 layer profile상 가장 큰 단일 계층이었다.
각 회 profile 비중은 28.59~31.02%, 평균 시간은 43.75~57.11ms 범위였다.
따라서 출력 전달량보다는 이 계층을 포함한 tactic/레이아웃/빌드 조건을 우선
살펴볼 근거는 반복됐다. 모든 비용을 이 계층 하나로 설명하지는 않는다.

## 최종 검증과 다음 작업

- 재사용 비교/부하/계측 테스트: **47 passed**. Python 도구 2개 Ruff check 통과.
  이번에는 운영 앱·시험 도구 소스를 수정하지 않았으므로 해당 결과는 기존 도구의
  회귀 확인이며, 새 엔진 품질/목표 처리량 합격은 아니다.
- 모든 빌드/비교/프로파일링 컨테이너 종료 확인. level 1의 종료 코드 124와
  엔진 미생성을 성공으로 표시하지 않았다. 다른 후보 빌드/진단 실행은 정상 종료.
- 운영 AI/API/action-layer running, restart count 0. AI StartedAt은
  `2026-09-16T23:45:46.903505745Z` 유지. UI 지연 무영향을 검증한 것은 아니다.
- 원본 ONNX/운영 엔진/Primary 설정/카메라 설정의 SHA-256이 시험 전후 동일했다.
- 코드 변경 없음. 이 문서와 이전 후보 보고서의 후속 링크, private 모델/실험
  산출물만 추가했다. 삭제/배포/전력 모드/실장치 명령 변경 없음.

다음 작업:

1. 충분한 빌드 시간과 진행 관측을 확보해 level 1 이상을 완료하고 동일 입력에서
   수치/후처리 결과를 비교한다. 5분 실패를 모델 변환 불가능으로 판단하지 않는다.
2. 필요하면 고정 batch 4 대 동적 batch 1~4를 별도 후보로 비교하여 tactic 선택
   영향을 분리한다. 고정 batch 4만 지원하는 엔진을 일반 운영 다중 카메라 지원의
   완료로 간주하지 않는다. 프레임 생략으로 목표를 맞추지 않는다.
3. 개선이 검증된 후보에서 전체 운영 AI, 장시간 반복, 실물 1대 및 추후 4대,
   실제 720p 입력, 낙상 30초 이후 시계열과 오경보를 검증한다. 운영 중단이나
   모델 배포·전력 모드 변경이 필요하면 별도 승인받는다.
