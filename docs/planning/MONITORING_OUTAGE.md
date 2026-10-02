# P2.4 — DB 잠금 중 혼합 입력 누적·복구 검증

검증: 2026-09-21 KST · 계획 v0.18 · **첫 실행 단위 검증 완료 / P2.4 IN_PROGRESS**

이 문서는 첫 시험 당시의 기록이다. 이후 앱 수정·I22 잔여 범위는
[쓰기 경합 개선 보고서](MONITORING_CONTENTION.md), 노트북 연결 회복은 [후속 안내](WINDOWS_MONITORING_POC.md)를 따른다.

수신 DB 잠금 중 센서 관측 12건과 영상 이벤트 12건을 누적하고 복구 후
ID·고정 bytes·SHA-256·관제 저장 ACK를 대조했다. 두 시나리오 모두 24건이 일치했고
누락·충돌·추가 레코드·논리 레코드 중복·잘못된 저장 ACK는 0건이다.
관련 회귀 **158 PASS / 0 FAIL / 0 SKIP**, 신규 시험은 4개다.

## 1. 시험 범위와 결과

운영망과 분리한 Mosquitto 2.0 컨테이너(`--network none`, 포트 공개 없음,
64MiB/0.5 CPU 상한)의 Unix 소켓과 임시 SQLite DB를 사용했다.
모형 영상 이벤트는 `VideoEventIngest`의 기록 전용 복구로, 센서는
`SensorRecordWorker`로 정규화했다. 실제 CCTV 추론·센서 패킷 수신 시험은 아니다.

| 시나리오 | 장애 중 확인 | 복구 결과 |
|---|---|---|
| 수신 DB `BEGIN IMMEDIATE` 쓰기 잠금 | 수신 저장 실패 관측, 미확인 전달 24건, 저장 ACK 0건 | 잠금 해제 후 서비스 자동 복구, 24건 일치 |
| 수신 DB `BEGIN EXCLUSIVE` 읽기·쓰기 잠금 | 저장 상태 조회 실패 관측, 미확인 전달 24건, 저장 ACK 0건 | 서비스 종료 후 같은 DB·binding으로 객체 재생성, 24건 일치 |
| 송신 DB 정규화 쓰기 잠금 | 영상 원본/접수 ID 보존, 정규화 pending, 공통 레코드 0건 | 객체 재생성 후 같은 원본으로 복구, 두 번 재처리해도 레코드 1건 |
| 송신 DB 원본 접수 쓰기 잠금 | 접수 호출은 예외, 저장된 접수 0건 | 해제 후 재접수 성공, 접수 1건 |

혼합 시험마다 최종 원본 접수 24건, 관제 레코드 24건, 수신 영수증 24건을 확인했다.
저장 성공은 broker PUBACK가 아닌 `consumer_committed` ACK로 판정했다.
원래 영상 출력 경로의 작업 12건은 `held`로 유지했다. 물리 출력은 실행하지 않았다.
두 번째 시나리오의 재시작은 **동일 Python 프로세스 안에서 서비스·DB 객체 재생성**이며,
새 OS 프로세스 강제 종료/재기동 시험으로 표현하지 않는다.

최종 실행의 잠금 유지/해제 후 복구 시간은 쓰기 잠금 **3.408초/2.466초**,
읽기·쓰기 잠금 **9.969초/3.832초**였다. 작은 시험 표본의 관측값이며 장기 장애·처리량·현장 지연 합격 수치가 아니다.

## 2. 발견한 경합과 시험 수정

- 최초 관련 회귀는 129 PASS/1 FAIL이었다. 새 시험이 영상 정규화 함수를 직접 호출했고,
  송신 DB 쓰기 경합 시 기존 영상 재처리 경로를 거치지 못했다.
  [초기 결과](evidence/monitoring_outage_initial_regression_20260921.xml)를 보존했다.
- 실제 영상 복구 경로로 바꾼 후 156 PASS/1 FAIL에서 센서 원본 접수의
  `database is locked`를 관측했다. 송신 저장소의 SQLite 대기 상한은 250ms이고
  시험 전달 worker 주기는 20ms다(서비스 기본 주기는 1초).
  [접수 경합 결과](evidence/monitoring_outage_ingest_contention_20260921.xml)를 보존했다.
- 시험 입력자는 명시적 SQLite 잠금 실패에 한해서 최대 10초 안에 재시도한다.
  이미 발급된 접수 ID의 후속 저장은 같은 ID로 재시도하며, 실패한 호출을 접수 성공으로
  계산하지 않는다. 최종 원본 24건 대조로 불필요한 재접수 중복도 확인한다.
- 최종 실행에서 시험 입력자의 잠금 재시도는 두 혼합 시나리오 모두 0회였다.
  이는 앞선 경합이 해결됐다는 뜻이 아니다. 강제 잠금 시험으로 접수 실패와
  원본 보존/재처리 경계를 따로 검증했다. **애플리케이션 코드는 변경하지 않았다.**
- [I22](STATUS.md#3-미해결-이슈-대장)는 OPEN이다. 실제 주기/부하에서 경합 빈도·적체를
  계측하고 MQTT·영상 입력별 실패/재시도 책임을 확인해야 한다.
  시험 입력자의 재시도 기능이 운영 입력에도 구현됐다고 주장하지 않는다.

시험 준비 중 발생한 샌드박스 소켓 차단과 시험 소켓 권한 오류는 환경 문제로 분리했다.
운영 DB·네트워크 설정을 바꾸지 않고 격리 소켓 접근만 맞춰 검증했다.

## 3. 변경·검증·재실행

추가 코드: [혼합 누적/DB 잠금 테스트](../../tests/test_monitoring_outage_mqtt.py).
런타임 코드·DB 스키마·API·배포 설정·의존성은 변경하지 않았다.
기존 P2.3 증거의 소스/테스트 12개 해시도 일치한다.
Ruff·Black check·Python compile은 모두 통과했다.

[최종 JUnit](evidence/monitoring_outage_20260921.xml)에는 두 혼합 시험의
입력 건수·대조 결과·시간·재시도 횟수를 `outage_result` 속성으로 보존했다.
[검증 요약·파일 해시](evidence/monitoring_outage_validation_20260921.json)도 함께 기록한다.

격리 브로커가 준비된 상태에서 재실행한다. 소켓 경로는 `/tmp/cctv-p22-mqtt-`로 시작해야 한다.
이 실행에서 사용한 시험 브로커는 검증 후 종료했다. 운영 브로커를 지정하지 않는다.

```bash
rtk proxy env CCTV_TEST_MQTT_SOCKET=/tmp/cctv-p22-mqtt-rerun/broker.sock \
  .venv/bin/python -m pytest tests/test_monitoring_*.py \
  tests/test_sensor_record.py tests/test_record_delivery.py \
  tests/test_record_delivery_mqtt.py tests/test_video_event_ingest.py \
  -q -o addopts=--tb=short -o junit_family=xunit1 \
  --junitxml=/tmp/monitoring-outage-rerun.xml
```

새 실행 결과로 기존 날짜의 증거를 덮어쓰지 않는다. 브로커 미지정 시 실제 MQTT 시험은
SKIP되므로 동일 검증 완료로 계산하지 않는다.

## 4. 남은 P2.4 작업

1. I22: 입력별 접수 실패/재시도 경로 확인, 실제 서비스 주기와 부하별 재시도율·적체량 측정.
2. 격리 환경에서 대기 누적량·용량 한계/저장 실패·복구 시험. 운영 디스크를 채우지 않는다.
3. 오래된 데이터·미확인 데이터 표시와 장기 장애 보관 조건 검증.

D04/D06의 실제 부하·지연 상한·디스크 예산·보관시간은 미확정이다.
노트북 연결 I21은 사용자 요청으로 보류한다. 실제 외부 관제 인증/화면,
운영 배포·실장비·GPU 품질/속도는 이번 결과의 범위가 아니다.
P2.4 완료 조건 전체를 채운 후 P1.3 우선순위/만료 계약, P3.1 영속 장치 대기로 연결한다.
