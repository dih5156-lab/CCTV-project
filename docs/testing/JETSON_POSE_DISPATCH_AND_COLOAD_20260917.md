# Jetson Pose 실행 비용과 운영 AI 부하 분리 진단 — 2026-09-17

## 결론

**4채널×모든 운영 AI 매 프레임 30FPS는 아직 검증되지 않았다.**
CUDA Graph는 CPU 작업 제출 시간을 줄였지만, 이번 반복 표본에서 GPU 처리 시간은
큰 차이가 없었다. 운영 AI 일시 정지 전·중·후 재생 결과도 일관된 개선 패턴이
아니었다. 기존 약 14FPS를 장비의 고정 성능 상한으로, 새 28.79FPS를 합격으로
단정하지 않는다. 운영 엔진·코드·전력 설정은 바꾸지 않았다.

사용자가 별도로 허용한 최대 3분 AI 중지 구간은 **pause/unpause 53초**로 끝났다.
정지 중 health 검사 실패가 발생했으나, 정지 시작 후 140초 이내에 healthy와
카메라 프레임 처리 재개를 확인했다. API·EdgeX·장치 설정은 변경하지 않았다.

선행 자료: [L1/L3 재생 비교](JETSON_POSE_L3_COMPARISON_20260917.md).
이번 원자료: `data/benchmarks/pose_dispatch_profile_20260917.BvPTry/`.

## 1. 엔진 단독 비교

동일 Jetson AGX Orin, MODE_30W(2), TensorRT 10.3.0에서 L1 일반 실행,
L3 일반 실행, L3 CUDA Graph 요청을 각각 3회 수행했다. 순서는
L1→L3→Graph / L3→Graph→L1 / Graph→L1→L3이다. 다른 GPU 시험·빌드·인코딩은
동시에 실행하지 않았지만 운영 AI와 데스크톱 등 기존 작업은 그대로였다.
실행 사이에는 작업·승인 대기로 긴 시간 간격도 있어 시간 효과가 통제되지 않았다.

- 기존 엔진 파일을 read-only로 로딩했으며 재빌드하지 않았다.
- batch 4, 입력 4×3×640×640, trtexec 기본 생성 입력. 영상/디코딩/후처리 없음.
- warmUp=1000ms, duration=5초, iterations=10, infStreams=1.
- H2D/D2H 전송 활성화, separateProfileRun으로 타이밍과 레이어 프로파일을 분리.
- 각 실행은 network none, read-only rootfs, private 결과 디렉터리만 writable,
  메모리 상한 8GiB, 외부 timeout 90초. 앱 runner/EdgeX/장치 제어 소켓 없음.
- CUDA Graph 3회 모두 일반 측정 및 별도 프로파일에서 capture 성공 로그를 확인했다.
  단순히 옵션이 Enabled로 표시된 것만 성공 근거로 쓰지 않았다.

| 실행 조건 | 평균 GPU compute, 배치당 ms | 표본 표준편차 | 기술 통계용 t 95% 구간 | 평균 CPU enqueue, ms |
| --- | ---: | ---: | --- | ---: |
| L1 일반 | 60.2961 | 6.2108 | 44.8676~75.7246 | 3.9604 |
| L3 일반 | 58.8722 | 3.3914 | 50.4476~67.2968 | 4.0857 |
| L3 CUDA Graph | 57.8662 | 4.4519 | 46.8071~68.9253 | 0.0601 |

각 실행 평균 3개에 같은 가중치를 적용했다. 수십 개의 query를 독립된 실행 반복으로
부풀리지 않았다. 위 구간은 실행 편차의 기술 통계이며 SLA 신뢰수준이나 인과 효과의
증명이 아니다. enqueue는 exportTimes의 endEnqMs-startEnqMs에서 계산하여,
로그의 반올림된 값과 미세한 차이가 있을 수 있다.

L3 일반의 평균 H2D/D2H는 약 1.5073/0.6009ms, Graph는 1.3847/0.4773ms였다.
CPU 작업 제출 단축만으로 4채널 실시간 처리가 해결된다는 근거는 확보하지 못했다.
**이 표의 배치 지연/배치 QPS는 카메라 FPS가 아니다.** DeepStream의 실제 적용 여부와
전체 AI 품질·처리량도 평가하지 않았으므로 CUDA Graph를 운영에 추가하지 않았다.

레이어 프로파일에서는 DFL Softmax가 계속 가장 큰 단일 항목이었다.
L1 일반 평균 6.678~7.848ms, L3 일반 7.959~11.367ms,
L3 Graph 6.759~8.424ms로 관측됐다. profile percentage는 약 12~18%였으나,
보조 stream 1개를 쓰므로 레이어 시간의 합/비율을 전체 GPU wall time의 정확한
분해로 해석할 수 없다. Softmax 하나의 최적화가 목표를 보장한다고 판단하지 않는다.

추가 GPU 시험 전 16:44:18~16:44:31 KST tegrastats 14개 표본은
GR3D 45~98%, 산술 평균 70.43%였다. 이는 시스템 전체 사용률이며 특정 프로세스의
사용량을 뜻하지 않는다. GPU 현재/최대 주파수 스냅샷은 각각 612MHz였다.

## 2. 승인된 운영 AI 일시 정지 비교

운영 `/app/src`와 `/app/scripts`가 작업 디렉터리의 bind mount인 것을 실제
docker inspect로 확인했다. stop/start는 미적용 코드까지 새로 읽을 수 있어,
이 위험을 사용자에게 설명하고 실행 상태를 보존하는 pause/unpause로 비교했다.
이 방법은 프로세스의 작업 제출을 멈추지만 GPU context/메모리를 해제하는 시험은
아니며, 완전한 프로세스 제거/독점 GPU 시험으로 확대 해석하지 않는다.

안전 절차:

1. 사용자 systemd 임시 타이머로 Docker 상태 조회를 실행하고, 결과 파일 `true`와
   loaded/active/exited, Result=success, ExecMainStatus=0을 확인했다.
   최초 예행은 journal이 없고 unit이 해제되어 성공 증거로 사용하지 않았다.
2. 정확한 운영 AI 컨테이너 ID에 대해 100초 후 unpause하는 독립 복구 타이머를
   먼저 등록·검증했다. 타이머 준비 실패/지연 시 pause하지 않도록 제한했다.
3. EXIT/INT/TERM 경로의 복구와 시험 내부 70초 timeout(+5초 kill 유예)을 함께 뒀다.
4. 시험 종료 시 pause 상태임을 확인하고 직접 unpause했다. 복구 뒤 타이머를 해제했다.
   실제 자동 복구 시한까지 도달하지 않았으므로 실컨테이너의 watchdog 발동을
   시험했다고 주장하지 않는다. 임시 예행 타이머·서비스도 후속 정리했다.

| 실제 시각, KST | 관측 |
| --- | --- |
| 17:19:56 | pause 요청/완료 |
| 17:20:46 | 격리 재생 시험 완료 |
| 17:20:48 | unpause 요청; 시험 종료 당시 paused=true 확인 |
| 17:20:49 | unpause 완료, 총 pause 약 53초 |
| 17:20:50 | running=true / paused=false, health는 아직 unhealthy |
| 17:21:12~17:21:42 | 운영 프레임 910903→911816, cameras=1, failed=0 |
| 17:22:16 이전 | AI/API/action-layer 모두 healthy 확인 |

정지 구간에는 AI 분석 중단이 있었다. 복구 후 카운터 증가/healthy는 처리 재개의
증거이지 원본 영상 무손실이나 촬영→화면 지연의 검증은 아니다. 최초 종료 로그의
중복 timer 정리에서 `Unit ... not loaded`가 나왔지만 이미 unpause된 상태였고,
후속 조회에서 시험용 timer/service가 남지 않았음을 확인했다.

재생 조건은 동일 원본 1080p, L3 엔진, 독립 decoder 4개, 공유 batch 4,
모델 입력 640×640, interval=0, warmup 10초 + 측정 약 30.04초였다.

| 상태 | 채널당 최저 추론 FPS | 측정 중 최대 재생 지연 증가 |
| --- | ---: | ---: |
| 운영 AI 실행 중, 정지 전 | 28.7907 | +1.1649초 |
| 운영 AI pause 중 | 21.7710 | +8.1908초 |
| 운영 AI 복구 후 | 14.4804 | +15.5249초 |

각 상태 1회이며 다른 공유 작업/시간 효과가 통제되지 않았다. 일관된 개선이
없으므로 운영 AI만이 병목이라고도, 운영 AI를 제거하면 오히려 느려진다고도
단정할 수 없다. pause의 GPU 자원 보존 역시 한계다. 이전 약 14FPS와 이번
28.79FPS의 차이도 하드웨어 교체/엔진 개선 효과로 해석하지 않는다.
세 실행 모두 적체가 늘었으며, 28.79FPS도 지속적인 매 프레임 실시간 합격이 아니다.

## 검증·변경 범위

- 9개 엔진 진단 및 3개 재생 실행 모두 종료 코드 0. 재생은 finished=true,
  errors=[]이며 카메라별 seen=inferred=tensor, skipped=unknown=0.
- 재생 표본 최고 온도 55.031°C, 최소 MemAvailable 31.153GiB.
  엔진 단독 진단에는 연속 thermal watchdog을 적용하지 않았으므로 이 온도를
  해당 진단 전체의 최고 온도로 해석하지 않는다.
- 최종 AI/API/action-layer running, paused=false, healthy 및 기존 StartedAt 유지.
  MODE_30W(2) 유지. 모든 진단 컨테이너와 임시 복구 unit 종료 확인.
- 운영 ONNX/engine/Primary 설정/cameras.json, L1/L3 후보 엔진 및 기존 시험 도구
  3개 파일의 SHA-256 유지. 재생/출력 진단 도구와 앱 소스는 수정하지 않았다.
- 앱 코드 변경이 없어 기존 53개 회귀 테스트를 반복 실행하지 않았다. 이번 검증은
  실제 CLI 실행 결과, 원자료 수치 검산, 복구 상태·프레임 진행, SHA-256 보존이다.
- 변경 파일은 이 보고서, 선행 보고서의 후속 링크와 private 계획·실행 기록·결과다.
  실제 장치 UAT, 후보 정확도 승인, 전체 AI/장시간/실물 4대 합격은 여전히 미완료다.

원자료: `experiment_plan.json`, 9개 `*_times.json`/`*_profile.json`/로그,
`baseline_tegrastats.log`, `maintenance_plan.json`, `maintenance_execution.json`,
`maintenance_lifecycle.log`, `production_*_stats.log`, 세 재생 폴더의 `report.json`,
`dispatch_and_pause_summary.json`.

다음 단계는 운영 코드 적용 범위를 먼저 고정·검증하고 다른 공유 부하를 통제한
성능 기준을 확보하는 것이다. 추가 운영 중지·코드 재로딩·데스크톱 종료는 별도
승인 없이 실행하지 않는다. GPU 성능 합격과 별도로 NMS/낙상 시계열 및 전체
운영 모델의 매 프레임 완료·정확도 검증을 이어가야 한다.
