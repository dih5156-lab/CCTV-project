# CCTV·AIoT 아키텍처·데이터 흐름·시퀀스

2026-09-22 · 현재 작업 소스와 계획 v0.26을 읽는 보조 자료.

**[시작 페이지 열기](index.html)** → 원하는 그림을 선택한다. HTML을 브라우저로 직접 열 수 있고,
폴더를 노트북으로 복사해서 볼 수도 있다. 별도 서버 실행은 필요 없다.

| 그림 | 내용 | 편집 원본 |
|---|---|---|
| [01 전체 장치 제어](01-system.html) | CCTV·센서, 기존 AI 직접 구독/규칙 경유, TLV·EdgeX·Action Layer·장치 | [architecture JSON](01-system-r2.architecture.json) |
| [02 공통 저장·데이터 흐름](02-storage.html) | 선택 경로의 Raw → 정규화 → 공통 기록 → 관제 전송 | [dataflow JSON](02-storage.dataflow.json) |
| [03 관제 저장 확인 시퀀스](03-monitoring.html) | 저장 여부 선조회 → MQTT → 수신 DB commit → HTTPS ACK 재조회 → 송신 DB 완료 | [sequence JSON](03-monitoring.sequence.json) |

## 읽는 방법과 적용 상태

- 그림 01의 일부 상자는 관련 모듈/논리 분기를 묶은 것이다. 프로세스·컨테이너와 1:1 대응하지 않는다.
  AI 어댑터/규칙을 거치는 경로 외에 Action Layer가 `cctv/ai/events/...`를 직접 구독하는 경로도 있다.
- 센서 규칙 브리지는 `aiot/sensors/#`를 구독한다. TLV의 `EdgeXForwarder`가 MQTT 발행과 Core Data HTTP push를 나누므로, 센서 규칙을 Core Data 뒤의 필수 단계로 해석하지 않는다.
- 그림 02는 `AIOT_INGEST_DB`로 선택하는 새 경로다. 모든 운영 서비스가 이 경로로 전환됐다는 뜻은 아니다.
  Raw와 canonical 상자는 같은 source SQLite 안의 논리 구분이다. PostgreSQL 센서 투영은 기존 용도와 함께 유지된다.
- 직접 TLV 입력은 원본 commit 이후 파싱/준비와 기존 실시간 EdgeX 출력 시도를 수행하고 handler가 반환될 때 MQTT 수신 ACK를 보낸다. Raw commit 직후 모든 후속 처리를 비동기로 분리한 구현은 아니다.
- 그림 03은 가능한 정상 실행 순서다. 브로커 PUBACK와 receiver commit의 상대 도착 순서는 고정되지 않는다.
  broker ACK, 관제 DB 저장 ACK, 실제 장치 작동 성공은 각각 별도다.
- 수신 DB에 저장된 레코드는 HTTPS/FastAPI에서 ACK를 조회한다. HTTP API가 관제 레코드 입력 경로인 것은 아니다.
- 운영 상태는 09-21 기록을 참고했고 이번에는 장비/컨테이너를 재점검하거나 배포하지 않았다. 실제 외부 관제/화면, 우선순위·만료·장치 장애 격리와 장시간 부하는 계속 후속 작업이다.

노드 선택·확대/축소·테마·내보내기는 Archify 기본 기능이다. 설명은 한국어이며 고정 메뉴와
생성 HTML의 `lang`은 스킬의 지원 범위에 따라 영어로 남는다. 자동 애니메이션은 켜지 않았다.

## 설치·검증

[Archify 원본](https://github.com/tt-a1i/archify)을 요청한 전역 범위에 설치했다.
Codex 대상 `archify` 한 스킬만 선택했으며 설치 위치는 `/home/sawwave/.agents/skills/archify`다.
스킬 메타데이터는 2.17, 실행 환경은 Node 20.20.2다. 다음 사용자 턴부터 스킬 검색 대상이 된다.
이번에는 설치된 `SKILL.md`를 직접 읽어 적용했다.

```bash
rtk proxy npx --yes skills add tt-a1i/archify -g --skill archify --agent codex --yes
```

- `doctor`: PASS.
- 세 그림의 `validate`와 `deliver`: 각각 **showcase 9/9, 오류 0, 경고 0**.
- 산출물/명세 SHA-256·바이트 수: [인계 기록](evidence/archify_handoff_20260922.json), [구조도](evidence/01-system.delivery.json), [데이터 흐름](evidence/02-storage.delivery.json), [시퀀스](evidence/03-monitoring.delivery.json).
- `visual-check`: **SKIPPED**. Chrome/Chromium이 설치돼 있지 않아 화면 크기별 자동 측정을 수행하지 못했다.
  각 HTML 옆 `.visual-check.json`에 아티팩트 해시와 생략 이유가 있다.
- Firefox 추가 캡처: 첫 시도는 Snap의 `/tmp` 프로필 접근 실패, 작업 폴더의 독립 프로필로 재시도했으나 45초 시간 제한에 도달했다. 화면 캡처·육안 검토 및 실제 메뉴/내보내기 작동을 검증 완료로 표시하지 않는다.
- 런타임 앱·기존 테스트·설정·DB를 수정하지 않았다. 이전 개발 단위 앱/테스트 해시도 동일함을 확인했다.

수정은 JSON에서 한다. HTML을 직접 수정하면 명세와 검증 해시가 달라진다. 검증 후 전달하는 명령 예:

```bash
rtk proxy node /home/sawwave/.agents/skills/archify/bin/archify.mjs validate architecture docs/architecture/archify/01-system-r2.architecture.json --quality showcase --json
rtk proxy node /home/sawwave/.agents/skills/archify/bin/archify.mjs deliver architecture docs/architecture/archify/01-system-r2.architecture.json docs/architecture/archify/01-system.html --quality showcase --json
```

`01-system.architecture.json`은 첫 구조도 명세로 보존했다. 실제 최신 HTML은 `-r2` 명세에서 생성한다.
Archify는 레이아웃을 검증하며 코드·배포 동작 자체를 입증하지 않는다. 미커밋/미추적 변경이 있어
공개 Git commit에 고정된 `SRC` 배지를 사용하지 않았고, 아래 로컬 소스와 해시를 근거로 남겼다.

## 소스 근거

| 내용 | 확인한 코드/기록 |
|---|---|
| AI 이벤트 직접 구독·센서 규칙 구독 | [Action 토픽](../../../src/services/_action_bridge_topics.py), [AI 어댑터 실행](../../../runners/run_edgex_adapter.py), [센서 브리지 실행](../../../runners/run_sensor_rule_bridge.py) |
| TLV Raw 우선 저장·handler 반환·복구 출력 제외 | [SensorService](../../../parser-python/service/sensor_service.py), [TLV ACK 경계](../../planning/TLV_MQTT_RECOVERY.md) |
| 센서 MQTT와 Core Data 분기 | [EdgeXForwarder](../../../parser-python/mqtt/edgex_forwarder.py) |
| 영상 원본·정규화·BUSY 1회 재시도 | [VideoEventIngest](../../../src/core/video_event_ingest.py), [최신 200개 시험 보고](../../planning/VIDEO_INTAKE_CONTENTION.md) |
| 관제 선조회·발행·ACK 저장 | [MonitoringDeliveryWorker](../../../src/monitoring_worker.py), [Monitoring API](../../../src/monitoring_api.py) |
| 출력 명령 발행과 실물 성공의 차이 | [장치 전송](../../../src/services/device_command_transport.py), [현재 상태/이슈](../../planning/STATUS.md) |
| 노트북 전달 근거 | [실제 LAN 정상/중지/복구 기록](../../planning/MONITORING_REMOTE.md) |

지식 그래프는 기존 구조 탐색에 사용했지만 신규 저장/관제 모듈 일부는 포함하지 않았다.
해당 부분은 실제 파일과 검증 기록으로 보완했다. 다이어그램을 개발 완료율로 환산하지 않으며
WBS는 DONE 8/33, P2.4 IN_PROGRESS를 유지한다. 다음 개발 작업은 기존 혼합 적체·재시작 시험이다.
