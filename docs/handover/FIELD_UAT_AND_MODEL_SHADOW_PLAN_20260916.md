# 현장 장치 UAT 및 Candidate shadow 실행 계획

기준일: 2026-09-16 KST

> **2026-09-18 후속 확인:** 아래는 당시 관측과 계획이다. 전광판은 이후
> [09-17 09:14 UAT](EDGEX_SIGNBOARD_UAT_20260917.md)에서 명령·결과 DB 기준 통과했다.
> 사용자는 CCTV 1~2대·온습도/기울기·전광판·스피커 보유, 사이렌 미보유를 확인했다.
> 현재 라우팅은 EdgeX이며, 과거 Direct 복구 예시는 현행 검증이 필요하다.
> 최신 일정·장비 조건은 [POC 범위](../planning/POC_SCOPE.md), 상태·이슈는
> [기존 성과·이슈](../planning/STATUS.md)를 따른다. 과거 timeout은 현재 연결 판정이 아니다.

## 현재 판정

- Candidate temporal 모델은 `shadow` 모드이며 장비 제어 veto는 비활성화되어 있다.
- AI Engine, Alert API, DeepStream은 정상이고 frame drop 및 failed는 0이다.
- Candidate는 5분 간격의 연속 temporal shadow window를 수집 중이다. 9월 16일
  14:34 KST부터 9월 17일 08:29 KST까지 216개가 기록됐고, 215개는 사람이 없는
  구간이라 `insufficient_pose_frames`로 제외됐다. pose 30프레임인 1개 결과는
  검수 clip 저장 시점이 추론 시점과 달랐던 구버전 기록이므로 승격 평가에서
  제외한다. 동일 프레임을 추론·저장하도록 수정한 9월 17일 08:40 KST 이후
  기록부터 유효 현장 증거로 사용한다.
- 전광판은 과거 EdgeX 단독 canary를 통과했지만 현재 `192.168.88.91:5000` 연결은 timeout이다.
- 스피커는 설정돼 있으나 현재 `192.168.88.92:80` 연결이 timeout이다.
- 사이렌은 `SIREN_HOST`, `SIREN_USER`, `SIREN_PASSWORD`가 설정되지 않았다.
- 장치 네트워크 route는 스피커와 전광판 모두 존재한다. 현재 장애는 Jetson route 부재보다 장치 전원·현장망·포트 응답 상태 쪽을 먼저 확인해야 한다.
- 세 Device Service health는 모두 정상이다. UAT `dry-run`은 물리 명령을
  보내지 않고 health만 확인하도록 보강했으며 세 장치 모두 PASS했다.

2026-09-17 08:43 KST 재점검에서도 세 Device Service dry-run은 모두 PASS했다.
다만 스피커 `192.168.88.92:80`은 timeout, 전광판 `192.168.88.91:5000`은
`No route to host`로 실장비 연결은 회복되지 않았다. 두 목적지의 커널 route는
`wlP1p1s0`을 통해 존재하므로 Jetson 설정 누락보다는 현장 장비 전원·중간망·
VLAN/게이트웨이 상태를 먼저 확인한다. 사이렌 접속정보 3개도 계속 미설정이다.

## 권장 일정

| 일시 | 작업 | 예상 시간 | 시작 조건 | 완료 기준 |
|---|---|---:|---|---|
| 9월 16일 | 재촬영 영상 구간 분석, appearance 모델 경로 복구, Candidate shadow 기동 점검 | 완료 | - | 분석 문서 생성, TensorRT 색상 모델 로드, 런타임 healthy |
| 9월 16~18일 | Candidate 현장 shadow 관찰 | 2~3일 | 실제 카메라 운영 유지 | 사건별 결과·pose 부족률·FP/day 확보 |
| 9월 17일 오전 또는 다음 현장 가능 시간 | 스피커·사이렌 실장비 UAT | 60~90분 | 장비 전원/망 연결, 사이렌 접속정보, 물리 출력 허용 | 각 장치 명령·실물 동작·결과 DB·롤백 PASS |
| 스피커·사이렌 UAT 직후 | 전광판 EdgeX 단독 운영 라우팅 부분 전환 | 30~60분 | 전광판 포트 회복, 현장 출력 허용 | signboard만 EdgeX, 나머지 Direct, 실패 시 즉시 Direct 복구 |
| 장치 2대 이상 UAT 통과 직후 또는 9월 18일 | 다중 장치 장애 격리·롤백 UAT | 90~120분 | 실제 장치 2대 이상 동시 접근 가능 | 한 장치 장애가 다른 장치 명령·결과에 영향 없음, Direct 복구 PASS |
| 9월 21일 | Candidate 1차 승격 판정 | 30~60분 | shadow 사건 및 라벨 확보 | Baseline 대비 recall·FP/day·사후 자세 미탐 비교 |

날짜는 장치 접근이 9월 17일에 가능하다는 가정이다. 장비가 계속 timeout이면
실장비 UAT 일정은 연결이 복구된 첫 업무일로 자동 순연한다. Candidate shadow는
장비 연결과 무관하게 계속 수집할 수 있다.

## 실행 순서와 안전 기준

1. 현장 UAT 전 `check_alarm_devices.py`와 `check_field_network.py`를 재실행한다.
2. 스피커·사이렌은 개별 장치 UAT부터 수행하고 자동 재시도는 사용하지 않는다.
3. 전광판은 `EDGEX_ALLOWED_DEVICES=signboard`인 부분 전환으로 시작한다.
4. 다중 장치 시험에서는 한 장치만 의도적으로 차단하고 나머지 장치의 성공 여부를 확인한다.
5. timeout, 중복 출력 또는 결과 상관관계 오류가 발생하면 Action Layer를 Direct 기본값으로 복구한다.
6. Candidate 결과는 shadow 기록에만 남기고 알람·스피커·사이렌·전광판 제어에는 사용하지 않는다.

연속 shadow window는 `FALLDATA_AUX_SHADOW_ALL_WINDOWS=true`, 수집 간격은
`FALLDATA_AUX_SHADOW_WINDOW_INTERVAL_SEC=300`으로 설정했다. Shadow 전용
payload는 Candidate가 낙상으로 판정해도 이벤트 큐에 넣지 않도록 차단한다.

2026-09-16 점검에서 주기 window가 temporal Candidate 대신 inline pose-RF로
분기되던 문제를 수정했다. 이전 설정의 997,200차원 영상 RF와 53차원 inline
pose 특징 간 불일치 기록은 Candidate temporal 결과가 아니므로 승격 평가에서
제외한다. 일반 낙상 보조검증용 compare 모델은 프로젝트 기본 pose-RF 번들로
복구했고, 주기 window와 near-miss만 temporal Candidate를 사용한다. Shadow
window는 확정 결과와 fail-open 오류 모두 운영 이벤트 큐로 전달되지 않도록
각 경로를 별도로 차단하고 회귀 테스트로 고정했다.

추가 현장 점검에서 pose가 0프레임인 빈 입력에도 Candidate 원시 확률이
`0.99999583`으로 나오는 취약성을 확인했다. 최소 pose 10프레임 품질 게이트를
추가해 이런 결과는 `insufficient_pose_frames`, `confirmed=false`로 기록한다.
게이트 적용 뒤 운영 이벤트 카운터는 detected/sent/failed 모두 0이며, 저장
clip 재추론 결과가 라이브 확률과 동일함도 확인했다. 운영 점검기의 shadow
신선도 기본값은 5분 수집 주기와 추론 여유를 포함하도록 420초로 조정했다.
추론 프레임과 검수 clip의 시점이 달라질 수 있던 문제도 수정해 동일 프레임을
재사용한다. pose 0인 무효 window는 JSON 기록만 남기고 clip은 저장하지 않아
장기 shadow에서 불필요한 디스크 증가를 막는다. 기존 clip은 삭제하지 않았다.

## Candidate 승격 보류 사유

공개 데이터 C1~C8에서는 Candidate가 Baseline보다 recall이 높았지만 FP가
2건에서 7건으로 늘었다. 재촬영 영상에서는 Candidate가 낙상 직후 30~40초
자세를 놓치는 사례도 확인됐다. 현장 shadow에서는 빈 pose 입력을 낙상으로
강하게 예측하는 취약성도 확인됐다. 따라서 최소 2~3일의 유효 pose 현장
shadow와 사람 검수 결과 없이 Candidate를 운영 판정이나 물리 장치 제어에
연결하지 않는다.
