# 백업·복구 런북

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

