# 현장 수용 테스트(UAT) 체크리스트

## 판정 기준

- **PASS**: 기대 결과와 실제 결과가 모두 일치
- **FAIL**: 기능 미동작, 누락, 중복, 지연 초과 또는 안전상 위험
- **BLOCKED**: 장비·네트워크·권한 문제로 검증 불가

모든 결과에는 시각, 장비명, camera_id/device_id, event_id, 로그 경로를 남긴다.

## 기본 연결

| ID | 시나리오 | 기대 결과 | 결과 | 증거 |
|---|---|---|---|---|
| U-01 | 모든 Compose 서비스 기동 | 필수 서비스 Up |  |  |
| U-02 | Public/Alert/Action health | 정상 JSON 응답 |  |  |
| U-03 | RTSP 연결 | 영상 프레임 수신 |  |  |
| U-04 | 외부 MQTT uplink | parser 수신 |  |  |

## AI 이벤트

| ID | 시나리오 | 기대 결과 |
|---|---|---|
| AI-01 | 사람 검출 | bbox와 person 이벤트 기록 |
| AI-02 | 헬멧 착용/미착용 | helmet/head가 구분되고 중복 과다 없음 |
| AI-03 | 낙상 영상 | fall_detected, score, event_id 기록 |
| AI-04 | 정상 앉기/쪼그리기 | 불필요한 낙상 알람 없음 |
| AI-05 | 수동 모드 | pending 생성, 승인 전 장치 미동작 |
| AI-06 | confidence threshold 미달 | filtered 기록, 장치 미동작 |

## 센서·EdgeX

| ID | 시나리오 | 기대 결과 |
|---|---|---|
| S-01 | 정상 Base64/TLV uplink | `aiot/sensors/#` 발행 |
| S-02 | tilt/temperature/vibration 임계 초과 | 규칙 이벤트 발행 |
| S-03 | 잘못된 payload | parser 오류 기록, 프로세스 유지 |
| S-04 | EdgeX 전달 | Reading 저장·조회 가능 |
| S-05 | MQTT 일시 단절 후 복구 | outbox/retry 후 누락 최소화 |

## 장치 출력

| ID | 시나리오 | 기대 결과 |
|---|---|---|
| D-01 | 스피커 fall 이벤트 | TTS 방송, 실행 결과 acknowledged |
| D-02 | 전광판 이벤트 | 한글 문구 표시, EUC-KR 깨짐 없음 |
| D-03 | 사이렌 이벤트 | ON 후 `SIREN_AUTO_STOP` 뒤 OFF |
| D-04 | 한 장치 네트워크 단절 | 해당 장치만 failed, 다른 장치·DB는 계속 |
| D-05 | 같은 이벤트 반복 | cooldown 동안 중복 출력 억제 |

## 성능·안정성

- [ ] Jetson warm-up 이후 latency 평균과 p95 기록
- [ ] 30분 이상 frame drop·메모리 증가 확인
- [ ] MQTT retry/outbox pending이 계속 증가하지 않음
- [ ] 장치 응답 timeout이 전체 이벤트 처리를 막지 않음
- [ ] 재시작 후 모델·카메라·장치 설정이 복원됨

## 승인 기록

| 항목 | 담당자 | 일시 | 결과 | 비고 |
|---|---|---|---|---|
| AI | `<입력>` | `<입력>` | PASS/FAIL |  |
| 센서/EdgeX | `<입력>` | `<입력>` | PASS/FAIL |  |
| 장치 | `<입력>` | `<입력>` | PASS/FAIL |  |
| 운영 책임자 | `<입력>` | `<입력>` | 승인/보류 |  |

