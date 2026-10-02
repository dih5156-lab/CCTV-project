# Jetson GPU·Windows 실시간 Shadow 테스트

## 목적

Jetson에서 기존 모델과 새 Temporal 모델을 GPU로 실행하고, Windows 노트북에서는 분석 RTSP 화면을 확인한다. 새 모델은 shadow 결과만 기록하며 장치 제어 이벤트를 직접 발생시키지 않는다.

## Jetson 설정

`.env.jetson`에 아래 값을 추가하거나 기존 값을 임시로 변경한다.

```dotenv
FALLDATA_AUX_MODE=shadow
FALLDATA_AUX_TEMPORAL_COMPARE_MODEL_PATH=/app/models/fall/fall_temporal_same_split_resume_20260908.pt
FALLDATA_AUX_TEMPORAL_NEAR_MISS_SHADOW_ENABLED=true
```

설정 변경 전 `.env.jetson`을 백업하고, 테스트 종료 후 원래 값을 복구한다. 설정 반영에는 `cctv-ai-engine` 재생성이 필요하다.

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --no-deps --force-recreate cctv-ai-engine
```

## Windows 화면 확인

Jetson과 Windows가 같은 네트워크에 있고 TCP 8554가 접근 가능해야 한다.

```powershell
ffplay rtsp://JETSON_IP:8554/camera_1
```

VLC에서는 `미디어 → 네트워크 스트림 열기`에 같은 RTSP 주소를 입력한다.

## 테스트 기록

각 영상은 `data/fall_test/20260910/` 규칙에 따라 저장하고, 기존 모델 결과와 Temporal shadow 결과를 함께 기록한다. 테스트 종료 후 shadow 로그에서 모델별 판정 시각·확률·미탐·오탐을 비교한다.

## 운영 보호

- `FALLDATA_AUX_MODE=shadow`를 유지한다.
- `FALLDATA_AUX_COMPARE_VETO_ENABLED=false`를 유지한다.
- 장치 제어 승격은 하지 않는다.
- 테스트 종료 후 `.env.jetson` 원복 및 컨테이너 healthcheck를 확인한다.
