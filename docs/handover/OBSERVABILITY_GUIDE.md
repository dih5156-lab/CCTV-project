# 모니터링 지표 가이드

## 정상 확인 순서

1. 컨테이너 상태와 health/readiness
2. 영상 프레임 수신·추론 latency·frame drop
3. MQTT publish/subscribe·retry·outbox pending
4. EdgeX Reading·rule 처리
5. Action 장치별 성공/실패·cooldown
6. DB 크기·디스크·메모리

## 주요 지표

| 영역 | 지표/로그 | 이상 징후 |
|---|---|---|
| DeepStream | `yolo_postprocess`, avg/max ms | max 증가, failed/frame_dropped 증가 |
| API | HTTP request/error/latency | 5xx·timeout 증가 |
| MQTT | publish 실패·재시도·outbox | pending 지속 증가 |
| AI | 이벤트 수, detector score | 갑작스런 0 또는 비정상 급증 |
| Action | device command/result | 특정 장치 failed 지속 |
| 시스템 | CPU/GPU/RAM/disk | 메모리·디스크 계속 증가 |

컨테이너가 `Up`이어도 지표가 멈췄거나 이벤트가 흐르지 않으면 정상으로 판정하지 않는다. 장시간 평가에서는 평균뿐 아니라 p95 latency, frame drop, 파일 디스크립터와 메모리 추세를 기록한다.

## 장애 시 증거 수집

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
docker logs --tail 200 cctv-ai-engine
docker logs --tail 200 cctv-action-layer
curl -fsS http://127.0.0.1:9000/health
curl -fsS http://127.0.0.1:9000/api/v1/metrics
```

수집 시 민감한 인증 헤더·비밀번호·얼굴 이미지는 제거한 뒤 공유한다.

