# P2.4 — 노트북 원격 수신·저장 ACK

2026-09-21 · 계획 v0.24 · **실제 노트북 정상 전달·수신기 중지/재기동 후 총 4건 저장 ACK 대조 완료**.

흐름은 `Jetson 기록 DB → MQTT/TLS → 노트북 수신 DB commit → HTTPS 조회 → Jetson ACK 영속화`다.
기존 schema 1.0 envelope·ID·bytes/hash·consumer_committed ACK를 사용한다.
노트북과 Jetson의 SQLite 파일은 각각 로컬에 두며 공유하지 않는다.

## 인터페이스

| 요청 | 인증/범위 | 응답 |
|---|---|---|
| `GET /v1/health` | Bearer token | 200 ready 또는 503 waiting_for_mqtt, 등록 binding 포함 |
| `GET /v1/deliveries/{delivery_id}` | Bearer token, origin_id·consumer_id·destination query가 등록 binding과 일치 | 200 기존 저장 ACK, 404 미저장, 401 인증 실패, 403 binding 불일치, 503 저장소 조회 실패 |

404는 `{status: not_found, binding: 등록 binding, delivery_id: 요청 ID}`로만 미저장 판정한다.
그 외 오류는 `{error: {code: 코드}}`다. 잘못된 인증·대상은 자동으로 다른 대상으로 우회하지 않는다.
알 수 없는 응답, 잘못된 ACK, timeout, 5xx는 저장 성공으로 처리하지 않는다.
송신 worker는 200 응답도 기존 ID/hash·binding 검증을 통과해야 ACK를 영속화한다.
목적지는 시험 전용으로 새로 등록하고 이미 고정된 목적지/endpoint를 바꾸지 않는다.

## 실행 구성

- 노트북: 전용 Mosquitto TLS broker + 수신/HTTPS API, 별도 DB volume.
- Jetson: 기존 전달 worker + MQTT TLS 전송 + HTTPS ACK 조회.
- 전송 사용자와 수신 사용자를 분리하고 MQTT topic ACL을 둔다.
- CA/서버 인증서·MQTT 비밀번호·API token은 노트북에서 생성한다. 소스 패키지에 비밀값을 넣지 않는다.
- Jetson에는 CA·전송용 자격·API token만 전달한다. 서버 개인키/수신용 자격은 노트북에 둔다.
- 기본 시험 포트: MQTT TLS 18883, HTTPS 18443. nginx 18080과 별도다.

## 전달할 패키지와 실행

사용자 확인: 노트북 일반 Ubuntu에는 CCTV-project 소스가 없다. 필요한 Python 소스 9개와
수신 실행 파일·Dockerfile·Compose·초기화 도구만 포함하는 압축 파일을 만들었다.
프로젝트 전체·AI 모델·기존 DB·운영 설정·인증 정보는 포함하지 않는다.

- [노트북 실행 패키지](../../dist/monitoring/cctv-monitoring-poc.tar.gz)
- [복사·설치·송신·종료 설명](../../deploy/monitoring-poc/README.md)
- Jetson 전용 시험 DB: `data/runtime/monitoring_remote_20260921/source.db`
- SHA-256: `b0816a6beae99d49505252b4b50ea18527bcdb515e9d66b2239870d2844a843e`

압축 파일을 노트북 Ubuntu 홈 폴더로 복사하고 실행한다. 최초 이미지/패키지 다운로드가 필요하다.

```bash
tar -xzf cctv-monitoring-poc.tar.gz
cd cctv-monitoring-poc
python3 init_poc.py --host 192.168.2.79 --lan
docker compose up -d --build
docker compose ps
```

생성된 `client/`만 Jetson의 `data/runtime/monitoring_remote_20260921/client/`에 복사한다.
이후 [송신 CLI](../../runners/run_monitoring_remote_poc.py)의 안내 명령으로 `total=2 / stored=2`를 확인한다.
인증서·서버 키를 포함한 `state/`는 노트북에 남긴다. 기존 nginx 성공만으로 새 두 포트의
Windows 방화벽·WSL 전달이 준비됐다고 판정하지 않는다. 노트북 접근은 사용자 실행이 필요하다.

### 노트북 기동 확인 후 client 전달 — 09-21 16:07 KST

사용자가 노트북 `broker`와 `receiver` 모두 정상 실행됐다고 알려주었다.
Jetson에서 `192.168.2.79:18883`, `:18443`의 TCP 접속을 직접 확인했다.
Jetson의 로컬 SSH 22번 포트도 연결 가능하다. [확인 근거](evidence/monitoring_laptop_startup_20260921.json).
이 확인은 TLS 인증·MQTT 수신 준비·DB 저장 ACK까지 검증한 결과는 아니다.
당시 Jetson 시험 경로에 `client/`가 없어 데이터 발행은 실행하지 않았다. 이후 아래 16:19 실측에서 전달·인증·저장 확인을 완료했다.

**노트북 일반 Ubuntu**의 패키지를 풀어 둔 폴더에서 실행한다.

```bash
scp -r ./client sawwave@192.168.2.242:/media/sawwave/Learning11/CCTV-project/data/runtime/monitoring_remote_20260921/
```

암호를 요청하면 Jetson의 `sawwave` 계정 비밀번호를 입력한다. 채팅으로 비밀번호나 token을 보내지 않는다.
복사 후 Jetson에서 4개 파일(`ca.crt`, `config.json`, `api-token`, `sender-password`)의 존재와
binding 일치·TLS 인증을 확인하고 합성 2건을 발행한다. `state/`는 복사 대상이 아니다.

09-21 16:14 확인: 사용자가 scp 비밀번호 거부를 보고했다. SSH 로그에 노트북에서 전달된
사용자명이 선행 ASCII 공백을 포함한 ` sawwave`로 기록돼 `Invalid user`로 거절됐다.
정상 계정 `sawwave`는 존재하며(uid 1000, 비밀번호 상태 P), 이 로그는 정상 계정의 비밀번호가
틀렸다는 근거가 아니다. [사용자명 확인 근거](evidence/monitoring_scp_username_20260921.json).
여러 줄 명령의 역슬래시 뒤 공백/줄바꿈 처리 문제가 가능한 원인이지만 노트북 입력 명령 자체는 확인하지 않았다.
위 scp 명령을 **역슬래시 없이 한 줄로** 다시 실행한다. 비밀번호 초기화·SSH 설정 변경은 하지 않았다.

### 두 장비 간 실제 전달 완료 — 09-21 16:19 KST

Jetson의 client 4개 파일(권한 0600)과 전용 시험 DB의 origin/binding 일치를 확인했다.
노트북 `https://192.168.2.79:18443/v1/health`가 CA 검증·토큰 인증을 통과하고
등록 binding과 `ready`를 반환했다. 기존 송신 CLI를 한 번 실행해 센서 모형 1건과
영상 이벤트 모형 1건을 MQTT/TLS로 전달했다.

- CLI exit 0, `total=2 / stored=2`, source 공통 레코드 2건.
- 각 레코드의 고정 wire hash, source에 영속화된 `consumer_committed` ACK,
  노트북 HTTPS 재조회 ACK의 ID/hash·receiver_record_id·stored_at이 일치했다.
- 물리 출력용 전달 작업 0건. 실제 CCTV/센서 입력이나 물리 출력은 실행하지 않았다.
- [실측 원자료](evidence/monitoring_laptop_delivery_20260921.json). 전체 확인 스크립트 소요 2.277초이며 지연 벤치마크/운영 상한 판정은 아니다.

인증된 저장 ACK 재조회까지 확인한 것으로, 노트북 DB 파일 직접 조회·실제 관제 화면·
장시간 연결 안정성·원격 중복 재전송/단절 복구를 이번 결과로 완료 처리하지 않는다.

## 구현과 영향 범위

| 파일 | 변경과 이유 |
|---|---|
| [monitoring_api.py](../../src/monitoring_api.py) | 인증된 binding 범위의 읽기 API. DB 조회 실패는 503, 정확한 scoped 404만 미저장 |
| [monitoring_http.py](../../src/protocols/monitoring_http.py) | CA 검증·토큰·timeout·redirect 차단·64KiB 응답 제한. 기존 worker가 ACK ID/hash를 최종 검증 |
| [monitoring_local.py](../../src/protocols/monitoring_local.py) | 기존 Unix 수신기에 선택형 TCP/TLS/계정 옵션 추가. 기본 로컬 동작 유지 |
| [receiver.py](../../deploy/monitoring-poc/receiver.py) | MQTT 재연결과 HTTPS 조회를 독립 실행. MQTT 단절 중에도 이미 저장된 ACK 조회 가능 |
| [compose.yaml](../../deploy/monitoring-poc/compose.yaml), [Dockerfile](../../deploy/monitoring-poc/Dockerfile), [requirements.txt](../../deploy/monitoring-poc/requirements.txt) | 노트북 전용 2서비스. 기존 사용 라이브러리의 경량 의존성만 설치, UID·메모리·CPU 한도 지정 |
| [init_poc.py](../../deploy/monitoring-poc/init_poc.py), [.dockerignore](../../deploy/monitoring-poc/.dockerignore) | 노트북에서 키 생성·topic ACL·송수신 계정 분리. 비밀 파일을 이미지 빌드 문맥에서 제외 |
| [build_monitoring_poc_bundle.py](../../scripts/ops/build_monitoring_poc_bundle.py) | 허용 파일 목록·hash manifest로 최소 배포물 생성. 새 시험 DB·binding 발급, 기존 파일 덮어쓰기 거부 |
| [run_monitoring_remote_poc.py](../../runners/run_monitoring_remote_poc.py) | 표시된 전용 POC DB에서만 합성 센서/영상 전달. 운영 DB·물리 명령 경로와 분리 |

공통 payload·source schema v4·receiver schema v1은 변경하지 않았다. 운영 Compose, GPU,
실센서, 카메라, 출력 장치를 변경/기동하지 않았다. 새 서버는 관제 수신/ACK 시험용이며 화면은 없다.
수신기 이미지 빌드는 ARM64 Jetson에서 검증했다. 노트북 기동은 사용자 확인에 이어
TLS 인증·health·실제 저장 ACK 응답을 직접 확인했다. 노트북 이미지 버전/로그·DB 파일 직접 조회는 별도다.

## 검증 근거

| 검증 | 결과 |
|---|---|
| API·HTTP reader·패키지·기존 수신/송신/worker/service 회귀 | **73 PASS / 0 FAIL / 0 SKIP**, [JUnit](evidence/monitoring_remote_unit_20260921.xml) |
| 기본 Unix 소켓 수신/DB 실패 복구/CLI 종료·재시작 | **3 PASS**, [기존 경로 회귀](evidence/monitoring_remote_unix_regression_20260921.xml) |
| 압축 해제 후 최소 소스만으로 import·Docker build·Compose 기동 | 성공. 실제 TLS Mosquitto + Uvicorn, 루프백 21883/21443 |
| 센서/영상 각 1건 실제 MQTT/TLS → receiver commit → HTTPS 조회 → source ACK | **2/2 stored**, ID/hash·consumer_committed 일치 |
| 같은 bytes/ID 재전송 | 수신 업무 레코드 2건·receipt 2건 유지 |
| 잘못된 HTTPS token/binding/CA, MQTT password | 성공으로 인정하지 않음. [TLS 통합 1 PASS](evidence/monitoring_remote_tls_20260921.xml) |
| 수신 컨테이너 실제 재시작 후 기존 ACK 조회 | [1 PASS](evidence/monitoring_remote_restart_20260921.xml), 기존 2건 ID/hash 일치 |
| 사용자용 CLI 재실행 | exit 0, total=2 / stored=2 |
| Ruff·Black·Python compile | 변경 Python 11개 통과 |
| 실제 Jetson → 노트북(192.168.2.79) 합성 2건 | 16:19 KST TLS 인증·ready·2/2 stored·source/remote ACK 대조 성공, [원자료](evidence/monitoring_laptop_delivery_20260921.json) |
| 실제 수신기 중지 중 새 2건 보존 | 16:29 KST 기존 stored 2건+새 uncertain 2건, 각각 4회 재시도·잘못된 ACK 0건, [원자료](evidence/monitoring_laptop_outage_20260921.json) |
| 실제 수신기 재기동 후 송신 재개 | 16:32 KST 새 접수 없이 총 4건 stored, 원래 ID/bytes/hash·기존 ACK 보존, [원자료](evidence/monitoring_laptop_recovery_20260921.json) |

전체 실행 결과는 **78 PASS / 0 FAIL / 0 SKIP**이며 서로 다른 4개 실행의 합계다.
[검증 환경·파일/패키지 hash·한계](evidence/monitoring_remote_validation_20260921.json)를 기록했다.

TLS 통합 시험의 첫 실행은 테스트 예상값을 `observation`으로 적어 실패했다.
실제 기존 계약은 `sensor_observation`이므로 테스트를 바로잡고 통과했다. 앱 payload는 바꾸지 않았다.
샌드박스 안 TestClient 정지는 해당 시험 프로세스를 종료한 뒤 승인된 외부 실행으로 검증했다.
코드 그래프에는 신규 모듈이 없어 `tests_for` not_found였으며 직접 소스/시험으로 영향 범위를 확인했다.

## 완료 실행 카드 — 노트북 수신기 중지·재시작 복구

전용 POC 수신기 중지/재기동 시험은 완료했다. 용량·오래된 데이터 등 잔여 기준이 있어 P2.4 전체는 IN_PROGRESS다.

1. 노트북 패키지 폴더에서 `docker compose stop receiver`로 시험 수신기만 중지한다.
2. Jetson에서 새로운 센서/영상 모형 각 1건을 별도 ID로 접수한다. 중지 중 저장 ACK가 없고,
   기존 완료 2건과 새 미확인 2건·고정 ID/hash가 source DB에 유지되는지 확인한다.
3. 중지 중 확인을 마친 뒤 노트북에서 `docker compose start receiver`로 재기동한다.
   같은 source DB에서 송신을 재개해 새 2건의 저장 ACK를 대조한다.

합격 조건: 기존 stored 2건 보존, 중지 중 새 2건의 잘못된 완료 0건, 복구 후 총 4건 stored와
동일 ID/hash·저장 ACK 일치. 노트북 DB 파일 삭제나 패키지 재초기화는 필요 없다.
사용자 수신기 중지 후 HTTPS health 연결 실패와 broker TCP 접속 유지를 확인했다.
새 센서/영상 모형 각 1건을 별도 ID로 접수하고 기존 송신 CLI를 8초간 실행했다.

- 기존 `stored` 2건의 ACK 보존, 새 `uncertain` 2건은 각각 4회 재시도·마지막 오류 `ConnectionError`.
- 새 2건의 ACK/acked_at은 비어 있고, 최초 ID·wire bytes·hash는 동일하다. 잘못된 완료 0건.
- source 공통 기록 총 4건, 물리 출력 전달 작업 0건. CLI exit 1은 이번 중지 중 시험에서 기대한 미완료 보고다.
- [중지 중 보존 근거](evidence/monitoring_laptop_outage_20260921.json). `offline_phase_pass`는 이 단계만의 합격이며 복구 완료는 아니다.

16:32 KST 사용자 재기동 제보 후 CA/토큰 인증 health=ready를 확인했다.
기존 송신 CLI를 `--demo` 없이 다시 실행해 같은 source DB의 대기 작업만 재개했다.
노트북 재기동만으로 종료된 Jetson CLI가 자동 시작된 것은 아니며, 상시 운영 서비스 적용은 별도다.

- CLI exit 0, `total=4 / stored=4`, 대기 2건 복구. 새 데이터 접수 없이 완료했다.
- 중지 전에 저장한 새 2건의 ID·wire bytes·hash가 일치하고, 기존 2건의 source ACK도 그대로다.
- 4건 모두 source 영속 ACK와 노트북 HTTPS 재조회 ACK의 ID/hash·receiver_record_id·stored_at이 일치한다.
- source 공통 기록 4건, 물리 출력 전달 작업 0건. 노트북 수신기는 재기동/ready 상태로 유지했다.
- [복구 근거](evidence/monitoring_laptop_recovery_20260921.json). 전체 확인 스크립트 2.428초는 운영 지연 벤치마크가 아니다.

이 시험은 짧은 수신기 중지와 기존 DB를 사용한 송신 재개를 검증했다. 수신 DB 파일 직접 조회,
장시간 장애·노트북 재부팅·네트워크 단절·디스크 부족·운영 상시 실행까지 검증한 결과는 아니다.

## 후속 검증 — 격리 용량 부족·내부 센서값 상태 조회 완료

[별도 격리 SQLite 시험](MONITORING_CAPACITY.md)에서 FULL 오류 시 접수 실패·기존 원본/대기 보존·ACK 복구를 검증했다(신규 6개 포함 71 PASS).
노트북 디스크나 LAN을 사용한 용량 시험은 아니다. 후속 [오래된 센서값·지연 도착 내부 조회](MONITORING_FRESHNESS.md)는 131 PASS이며 노트북 수신기에는 미배포다.
09-22 후속 [영상 최초 잠금 재시도·활성 쓰기](VIDEO_INTAKE_CONTENTION.md)는 200 PASS로 내부 검증했다. 다음은 혼합 적체 누적·재시작 복구/저장량 계측이다. D02/D04 실센서 유효시간과 D04/D06 실제 보관량·시간 기준은 대기다.
정식 서버 이전 상세는 사용자 지시로 관제 대상 결정 시 검토하고 현 시험 서버로 개발을 이어간다. DB/ACK·신원/주소·인증과 미전송 작업의 이전 절차는 D06 잔여로 유지한다.
D04/D06 실제 부하·용량·보관/오래된 데이터 기준, I22 운영 부하·영상 지연은 이후 잔여다. 재시도 소진으로 미접수된 영상의 영구 재전송 보장은 추가하지 않았다.
토큰 오류로 `blocked`가 된 작업의 운영자 해제 도구, 인증서 갱신·운영 보관 정책은 이 POC의 범위 밖이다.
HTTP timeout은 requests의 연결/읽기 제한이며 느린 응답 전체의 엄격한 wall-clock 상한 시험은 하지 않았다.
