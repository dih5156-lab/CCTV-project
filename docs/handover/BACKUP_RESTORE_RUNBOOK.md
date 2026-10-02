# 백업·복구 런북

## 2026-10-01 v0.92 — 같은 DB의 후보→이전 구성 전환 확인

고정 후보 종료·컨테이너 제거 후 DB 경로를 유지한 채 이전 이미지·보존 코드로 기동했다.
후보가 추가한 이벤트31건·외형28건을 포함한705517행의 값/상태 보존, 복귀 후 이벤트32건·외형28건
추가 저장과 DB3개 무결성ok를 확인했다. 후보 이벤트1건을 이전 저장 API에 재접수해 같은 행을 반환했고
전체 이벤트173830행의 수·내용도 전후 동일했다. 원자료는 `data/runtime/p51_transition_20261001_v093`이다.

전환 도중 새로 쌓인 데이터가 있으므로 **코드/이미지 복귀와 과거 DB 백업 덮어쓰기를 한 동작으로 묶지 않는다.**
이번처럼 스키마가 호환되면 쓰기 프로세스를 정상 종료하고 DB를 보존한 채 실행 구성을 바꾼다.
DB 손상/비호환으로 과거 스냅샷이 필요한 경우에는 전환 이후 이벤트·ACK 상태를 별도 보존하고
재처리/중복 실행 범위를 검토해야 한다. 이번 검증은 파괴적 스키마 변경의 역이행 시험이 아니다.

후보 SIGTERM6.37초·복귀3.49초, 영상3009프레임 디코드/원본 시간 역행 오류0.
이전 구성 매칭 누락13·TCN Pose 부족 결과와 CUDA700 원인 미확정은 별도 이슈다.
운영 AI·실제 DB·카메라/관제/장치에는 전환을 적용하지 않았다.
[v0.92 상세 근거](../planning/VIDEO_INFERENCE_CONTRACT.md#transition-rollback-v092).

## 2026-10-01 v0.91 — 복구 묶음과 격리 기동 확인

가상환경3개·SDK/NVIDIA 라이브러리·모델/캐시·선택 코드49,917파일(논리9.35GB)을
복사 후 SHA로 대조했다. 이전 이미지6.31GB를 내보내고 동일 이미지 ID로 다시 불러왔다.
설정/실제 컨테이너 구성/얼굴 등록12파일은 AES-256-GCM으로 보관·복호화 일치/변조 거부를 확인했다.
키와 복호화 파일은 비공개 경로에 두며 Git·진행판에는 올리지 않는다.

로컬 복구 묶음은 `data/runtime/p51_restore_20261001_v092/bundle/`, 기존8DB 복원본은
`data/runtime/p51_rollback_20261001_v089/restored/`다. DB는 묶음 안에 중복 포함된 것이 아니다.
키 위치는 시험 폴더의 `key-location.txt`에만 기록했다. 키 분실 시 암호화 설정은 복구할 수 없으며,
현재 둘 다 같은 장비에 있으므로 독립 저장매체를 이용한 재해 복구는 아직 준비되지 않았다.

격리 재실행에서 입력/추론/출력·DB 저장·SIGTERM2.59초 종료를 확인했다.
복제 DB3개의 기존705,398행(원래 열 전체) 보존, 대기 이벤트32건·외형28건 추가·무결성ok.
외형 `track_id_text` 열 추가는 기존 스냅샷 코드의 시작 동작이다. 운영 DB에는 적용하지 않았다.
기존 열의 값과 새 열의 존재를 각각 검사해야 하며 `SELECT *` 길이 차이를 데이터 유실로 판정하지 않는다.

첫 실행은150초 초기 진행 대기 실패, 재실행은 TCN timeout1·매칭 누락17·최대 영상 간격1.936초를 보였다.
원인은 아직 확정하지 않았다. 정상 기동/종료가 전체 모델 품질이나 출력 장애 복구 합격을 뜻하지 않는다.
다음은 **같은 격리 DB에서 후보 실행→이전 구성 복귀 후 신규 대기 이벤트 보존·스키마 호환 대조**다.
실제 운영 전환은 하지 않았다. I07 OPEN·필수 이슈 해결 전 배포 보류.

### 이번 복구본을 재사용하는 순서

1. 운영 writer에 연결하지 않고 별도 시험 디렉터리에서 시작한다. `bundle/manifest.json`의 파일/링크와
   [근거 JSON](../planning/evidence/restore_bundle_20261001.json)의 이미지 해시·DB 참조를 확인한다.
2. `operating-image.tar`를 적재한 뒤 이미지 ID가 `sha256:1ace5b7832a5…`인지 확인한다.
   적재 자체는 기존 운영 컨테이너를 재생성하지 않는다.
3. 비공개 키로 설정을 복호화하고 변조/경로 탈출을 거부한다. 이번 검증 스크립트는 시험 폴더의
   `restore_settings.py`이며 파일600·폴더700을 적용한다. 실제 비밀값을 로그/명령 인자에 출력하지 않는다.
4. DB 원본을 시험용으로 복제하고 가상환경/SDK/모델/선택 코드의 복구 경로를 연결한다.
   `trt_cache` 빈 연결 지점과 별도 쓰기 캐시를 준비한다. 이번에는 `compose_retry.json`의
   network none·내부 녹화 재생으로 시험했으며 실제 카메라/관제 연결이 아니다.
5. 기동/종료 결과, 기존 열/행의 값·새 행/스키마, 영상 PTS/DTS와 전체 디코드를 검사한다.
   최초 실패를 덮어쓰지 않고 원인 미확정 범위를 함께 보존한다.

이전 이미지에 연결한 코드는 v0.88 디스크 선택351파일로,09-17 시작 당시 메모리와 같다는 보장이 없다.
추가 가상환경2개·생략한 학습/샘플 마운트·절대 링크7개를 포함한 범위는
[v0.91 상세 근거](../planning/VIDEO_INFERENCE_CONTRACT.md#restore-bundle-v091)에 기록했다.
OS/드라이버 복원·전체 서비스 동시점 복구·백업 이후 운영 데이터 무손실은 미검증이다.

아래 v0.90 문단은 당시 기록이며, 외부 의존성/설정 묶음 준비 상태는 위 v0.91 결과로 갱신한다.

2026-10-01 v0.90 추가: 후보 전체 격리 기동은 확인했으며, 읽기 전용 모델 루트의
`trt_cache` 연결 디렉터리 누락(I42)을 보완했다. 복구/새 모델 묶음에도 빈 연결 지점을 포함한다.
후보 종료 유예는60초이며 SIGINT25.12초/SIGTERM4.86초를 별도 시험에서 관측했다.
외부 가상환경3개·SDK·NVIDIA 라이브러리 약7.69GB의 식별값은 기록했으나,
이 파일들과 비밀설정/전체 마운트를 포함한 복구 묶음·이전 이미지 전체 기동은 아직 미검증이다.
[최신 근거](../planning/VIDEO_INFERENCE_CONTRACT.md#profile-start-v090). 아래 DB별 복원 결과를 전체 운영 복귀 합격으로 해석하지 않는다.

## 2026-10-01 I07 준비와 I41 보완

운영 AI 설정은 `appearances.db`와 `mqtt_event_outbox.db`를 참조한다.
기존 정기 백업4개에 MQTT/HTTP 전달 대기와 EdgeX 이벤트 대기/명령 결과4개가 빠져 있어
`scripts/ops/backup_runtime_databases.sh`의 명시적 목록을8개로 보완했다.
사용자 정의 DB 경로·새 공통 저장소를 추가하면 이 목록과 실제 환경/마운트를 다시 대조해야 한다.
PostgreSQL·Redis·named volume 전체 백업을 뜻하지 않는다.

새 백업 디렉터리/파일은700/600으로 생성한다. 원본은 읽기 전용 연결과 SQLite backup API를
사용해 WAL 커밋까지 포함하고, `quick_check` 통과 후 `.partial`을 최종 `.db`로 전환한다.
기존 스냅샷은 덮어쓰지 않으며 실패한 복사를 완성된 백업으로 표시하지 않는다.
`CCTV_RUNTIME_BACKUP_PRUNE=0`이면 과거 백업 삭제를 생략한다. 기본값1·기존7일 보관은 유지한다.
설치된 systemd 서비스는 이 저장소의 스크립트를 직접 호출하므로 다음 실행부터 보완된 목록을 사용한다.

프로젝트 루트에서 새 경로로 온라인 백업만 만들 때:

```bash
rtk proxy env CCTV_RUNTIME_BACKUP_PRUNE=0 \
  CCTV_RUNTIME_BACKUP_DIR=/media/sawwave/Learning11/CCTV-project/data/backups/i07-review \
  bash scripts/ops/backup_runtime_databases.sh
```

이번 실제 시험 경로는 `data/runtime/p51_rollback_20261001_v089`이며 운영 DB에 덮어쓰지 않는다.
복원은 읽기 전용 백업에서 별도 `restored/`로 `VACUUM INTO`하고, 스키마·전체 행의
정렬된 내용 해시·행 수·상태 분포·`integrity_check`를 대조한다. 미사용 페이지 제거 때문에
원본/복원 파일 크기나 파일 해시는 달라질 수 있다. 모델/코드 실행 및 메시지 재발행 시험과 구분한다.
실행 결과는 [v0.88 근거](../planning/evidence/rollback_preparation_20261001.json)에 기록한다.

**각 DB의 온라인 스냅샷 시각은 다르다.** 이 시험을 서비스 전체의 같은 시점 복구나
백업 이후 데이터 무손실로 확대하지 않는다. 실제 전환에서는 writer/전달 worker의 처리 경계와
마지막 백업 이후 기록을 보존하고, 과거 ACK 상태 복구에 따른 중복 실행도 확인해야 한다.
`review.jsonl`은 기존 일반 파일 복사이며 SQLite와 같은 일관성 보장을 주장하지 않는다.

후보 이미지와 운영 구성의 차이도 [I07 현재 판정](../planning/VIDEO_INFERENCE_CONTRACT.md#rollback-preparation-v088)에 기록한다.
시험 후보는 네트워크none·임시 DB·비밀값 없는 설정이며 운영용 Compose가 아니다.
실제 운영의 MQTT 재전송 DB 환경변수를 시험 후보에 자동으로 생략해서는 안 된다.
현재 선택된 디스크 파일/모델 보존본은 시작 당시 메모리의 코드, 모든 마운트·비밀값·장치 상태의 복구본이 아니다.

## 목표

컨테이너를 재생성해도 이벤트·명령·센서·EdgeX 데이터를 잃지 않도록 백업 위치와 복구 순서를 고정한다.

## 보존 대상

| 대상 | 예시 | 보존 이유 |
|---|---|---|
| Alert/Action DB | `data/runtime`, action events | 이벤트·장치 명령 이력 |
| AIoT parser DB | `aiot-parser` DB volume | 센서 원본·TLV 결과 |
| EdgeX DB | `edgex-jetson_db-data` | Reading·metadata |
| eKuiper | `kuiper-data`, `kuiper-etc`, `kuiper-log` | rule·상태·로그 |
| 모델·설정 | `models`, `config`, `.env` secret 별도 | 동일 버전 복구 |
| 라벨·평가 결과 | `data/eval`, `data/fall_dataset`, report JSON | 재학습·검증 재현 |

## 백업 전 확인

1. 담당자에게 백업 시작을 알린다.
2. 현재 Compose 파일, git commit, 모델 manifest를 기록한다.
3. 진행 중인 테스트 이벤트가 없는지 확인한다.
4. 백업 저장소의 여유 공간과 접근 권한을 확인한다.

## 권장 절차

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
git rev-parse HEAD
```

실제 volume 백업 명령은 현장 Docker storage 정책에 맞춰 별도 승인 후 실행한다. 백업 파일에는 API key·비밀번호가 포함될 수 있으므로 암호화 저장하고 문서나 Git에 올리지 않는다.

## 복구 순서

1. 장애 시각과 마지막 정상 시각을 기록한다.
2. 현재 컨테이너와 volume 목록을 보존한다. 원인 분석 전에 삭제하지 않는다.
3. 코드·Compose·환경변수·모델 버전을 마지막 정상 조합으로 맞춘다.
4. DB와 named volume을 백업본으로 복구한다.
5. EdgeX Core, MQTT, parser, AI Engine, Alert, Action 순으로 기동한다.
6. health/readiness, MQTT topic, DB row, 장치 상태를 확인한다.
7. 복구 시각·누락 구간·재처리 여부를 기록한다.

## 복구 후 검증

- [ ] `/health`, `/readiness` 응답
- [ ] `cctv/ai/events/#`와 `aiot/sensors/#` 수신
- [ ] EdgeX Reading 생성
- [ ] 이벤트 DB 조회
- [ ] Action Layer pending/command 정상
- [ ] 장치 1회 테스트 출력
- [ ] 중복 이벤트·outbox pending 확인

`docker compose down -v`는 named volume을 삭제할 수 있으므로 운영 데이터 백업과 승인 없이 실행하지 않는다.
