# P2.4 — SQLite 용량 부족·기존 데이터 보존·복구

2026-09-21 · 계획 v0.23 · **격리 용량 부족 실행 단위 완료, P2.4 전체 IN_PROGRESS**.

임시 DB에서 실제 SQLite 용량 부족 오류를 발생시켜 접수·정규화·전달 대기·수신 저장·송신 ACK 기록의 실패와 복구를 검증했다. 신규 6개를 포함한 관련 회귀 **71 PASS / 0 FAIL / 0 SKIP**다. 애플리케이션 코드는 수정할 필요가 없었으며 테스트와 개발 문서를 보강했다.

## 시험 방법과 범위

`tests/test_monitoring_capacity.py`에서 pytest 임시 DB의 현재 페이지 수를 읽고, 해당 store 객체의 연결에만 `PRAGMA max_page_count`를 적용했다. 기존 크기에서 더 늘어날 수 없도록 한 뒤 64 KiB 합성 값 또는 일반 크기의 ACK를 저장한다. 오류를 임의로 던지는 mock 대신 실제 `sqlite3.OperationalError: database or disk is full`을 확인한다.

SQLite의 페이지 수 제한은 [공식 PRAGMA 설명](https://www.sqlite.org/pragma.html#pragma_max_page_count), 용량 오류의 의미는 [SQLITE_FULL 설명](https://www.sqlite.org/rescode.html#full)을 따른다. 제한은 시험 대상 연결마다 적용하고, 복구 단계에서는 제한을 제거한 후 같은 임시 DB를 새 store 객체로 연다.

- 환경: Python 3.10.12 / SQLite 3.37.2. 이번 실행의 제한값은 DB당 48~328 KiB이며 운영 용량 권장값이 아니다.
- 실제 운영 디스크를 채우지 않는다. 파일시스템 ENOSPC·inode 부족·journal/temp 별도 볼륨 고갈·정전 시험과는 다르다.
- MQTT callback의 ACK/연결 끊기 호출은 test double로 관찰한다. SQLite 저장은 실제 수행하지만 이번 시험에서 실제 broker/LAN은 사용하지 않는다.
- TLS 노트북 정상/중지/재기동 검증은 [별도 실제 LAN 근거](MONITORING_REMOTE.md)다. 두 시험을 합쳐 노트북 디스크 부족까지 실증했다고 해석하지 않는다.

## 결과

| 경계 | 용량 제한에서 확인 | 제한 해제 후 확인 |
|---|---|---|
| 센서 MQTT bytes 원본 접수 | 저장 예외 전파, 신규 부분 행 없음, 기존 접수/상태 보존 | 호출자가 같은 원본을 다시 제출하면 새 접수 1건 저장 |
| 영상 JSON 원본 접수 | 저장 예외 전파, 신규 부분 행 없음, 기존 접수/상태 보존 | 호출자가 같은 원본을 다시 제출하면 새 접수 1건 저장 |
| 영상 정규화 | 이미 commit된 원본 보존, pending 유지, canonical/revision 부분 저장 없음 | 같은 접수 ID로 재처리하여 canonical/revision 각각 1건 |
| 관제 전송 대기 생성 | 신규 job/state/wire 저장 rollback, 기존 wire 및 원본/canonical 보존 | 미완료 레코드 1건만 추가, 기존 대기 1건 유지 |
| 수신 저장 | 신규 레코드·수신 장부 없음, 저장 ACK 없음, callback의 MQTT ACK 호출 없음 | 같은 wire 재전달 후 저장, 중복 재전달에도 같은 수신 ID/저장 시각 유지 |
| 송신 ACK 영속화 | 수신기에 24건 commit 후 송신 ACK 기록 중 실제 FULL 발생, 실패 transaction 전체 rollback | lease 만료 후 worker 재개, 수신 상태 조회만으로 총 24건 stored, 재발행 0건 |

송신 ACK 시험은 이번 실행에서 3건 저장 후 다음 ACK에서 FULL이 발생했다. 제한 해제 후 나머지 21건을 완료했으며 모든 고정 wire bytes/hash가 유지됐다. 3건이라는 값은 시험 DB 페이지 배치의 관측 결과이지 용량 보장값이 아니다. 각 경계의 DB `quick_check`도 `ok`였다.

**접수 commit 전 실패한 데이터는 DB에 없으므로 자동 복구 대상으로 세지 않는다.** 첫 두 시험의 복구는 호출자의 명시적 재제출이다. 운영 영상의 최초 미접수 재시도/실패 정책은 I22에 계속 남긴다. 센서 시험은 원본 접수 저장소 경계이며 실제 TLV 해석·PostgreSQL projection의 용량 부족 시험은 아니다.

## 변경과 재현

추가 파일은 [회귀 테스트](../../tests/test_monitoring_capacity.py), 이 보고서 및 [JSON 근거](evidence/monitoring_capacity_20260921.json)·[JUnit 결과](evidence/monitoring_capacity_20260921.xml)다. README·WBS·PROGRESS·STATUS와 후속 작업 안내를 함께 최신화했다. DB schema/API/설정/배포 변경은 없다. 관련 애플리케이션 8개 파일의 작업 전후 SHA-256이 같음을 확인했다.

```bash
rtk proxy .venv/bin/python -m pytest tests/test_monitoring_capacity.py tests/test_ingest_store.py tests/test_event_ingest_store.py tests/test_monitoring_delivery.py tests/test_monitoring_worker.py tests/test_monitoring_receiver.py tests/test_monitoring_local.py tests/test_monitoring_contract.py -q -o addopts=--tb=short
rtk proxy .venv/bin/python -m ruff check tests/test_monitoring_capacity.py
rtk proxy .venv/bin/python -m black --check tests/test_monitoring_capacity.py
rtk proxy .venv/bin/python -m compileall -q tests/test_monitoring_capacity.py
```

회귀 71개·Ruff·Black 형식·문법 검사는 PASS다. 문서 8개의 로컬 링크 312개와 WBS 33행 상태 집계도 확인했다. 그래프의 신규 테스트 경로가 미등록이라 영향/커버리지 판정은 직접 소스 검토로 보완했다. 운영 패키지 재빌드/배포는 테스트·문서만 변경하여 수행하지 않았다. 기존 다른 검증의 PASS 수와 중복 합산하지 않는다.

## WBS 판정과 다음 단위

이번 **용량 부족 시 transaction 보존·복구 단위는 완료**다. WBS P2.4 전체는 다음 잔여 기준 때문에 IN_PROGRESS로 유지한다.

1. 오래된 센서값·장애 중 지연 도착 데이터를 최신 정상값과 구분하는 표시 및 시험.
2. I22 활성 쓰기 경합과 영상 최초 미접수 이벤트의 책임·재시도/실패 정책.
3. D04/D06 부하·최대 적체량·보관시간·디스크 예산 합의 및 해당 조건의 누적/장시간 시험.

다음 실행 단위는 1번이다. P2.4 종료 뒤 P1.3 우선순위/만료 계약 → P3.1 영속 장치별 대기로 이어간다. 정식 관제 제품 선정·실물 출력·운영 자동 시작 완료를 이번 판정에 포함하지 않는다.
