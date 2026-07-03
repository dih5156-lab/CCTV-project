# App Service Configurable (ASC) 기반 Rule/Route 운영 가이드

## 목적
이 문서는 현재 프로젝트의 토픽 체계를 유지하면서, EdgeX `app-service-configurable`를 운영에 활용하는 실무형 방식(ASC + Kuiper 하이브리드)을 정리합니다.

- 입력(기존): `cctv/ai/events/{camera_id}/{event_type}`
- 룰 출력(기존):
  - `cctv/rules/intrusion/filtered`
  - `cctv/rules/intrusion/persisted`
  - `cctv/rules/intrusion/critical`

## 핵심 결론
- 다중 카메라에서 ASC는 **표준 라우팅/전송/운영성 강화**에 매우 유리합니다.
- 다만 "5초 지속 감지" 같은 윈도우 집계는 현 구조 기준으로 Kuiper가 더 적합합니다.
- 따라서 권장안은 **ASC + Kuiper 하이브리드**입니다.

## 권장 역할 분리
1. AI Engine
   - 카메라별 이벤트 생성
   - `camera_id`, `type`, `confidence`, `timestamp`, `object_id` 포함
2. EdgeX Adapter / Device Service
   - `camera-{camera_id}` 메타데이터 정합 유지
   - EdgeX 이벤트 표준화
3. Kuiper
   - 침입 필터링, 지속 조건(5초), 고신뢰 분기
4. ASC
   - 다운스트림 라우팅과 HTTP API 전송
   - 현재 구성에서는 자체 store-and-forward를 사용하지 않음
5. Action Layer
   - AI, zone, intrusion, sensor 토픽 구독
   - 스피커·전광판·경광등 제어, 로컬 DB 기록, 외부 알림 연동

## ASC 적용 포인트 (이 프로젝트 권장)
- 입력: `cctv/rules/intrusion/#`
- 처리:
  - 토픽별 정책 분기(`filtered`, `persisted`, `critical`)
  - 외부 API 전송(HTTP Export)
  - 필요시 별도 MQTT 토픽으로 재발행(멀티 컨슈머 분리)

## 운영 토픽 권장안
- `cctv/rules/intrusion/filtered`  → 저장/대시보드 후보
- `cctv/rules/intrusion/persisted` → 알람 후보(저중요)
- `cctv/rules/intrusion/critical`  → 즉시 알람(스피커/문자/관제)

## 다중카메라 체크리스트
- 이벤트 payload에 `camera_id` 필수
- `camera_id` 기반으로 알람 쿨다운 분리(이미 적용됨)
- NTP 시간 동기화(카메라/서버)
- 카메라 명명 규칙 고정(`camera_1`, `camera_2`, ...)
- 장애 시 카메라별 헬스체크 로그 분리

## ASC 하이브리드 배포 순서
1. AI Engine + Adapter 정상 동작
2. Kuiper 룰 배포(`kuiper/rules/cctv_intrusion_rules.json`)
3. ASC를 후단 라우터로 연결
4. Action Layer에서 필요한 AI/zone/intrusion/sensor 토픽 구독 상태 확인

## 참고 파일
- Kuiper 룰: `kuiper/rules/cctv_intrusion_rules.json`
- 액션 레이어: `src/services/action_bridge.py`
- ASC 환경 템플릿: `edgex/asc/cctv_asc.env.example`
- ASC 커스텀 프로파일: `edgex/asc/cctv-external-http/configuration.yaml`

## 현재 구성 상태

아래 `app-rules-engine` 구성은 `docker-compose.jetson.yml`에 정의되어 있습니다. 일반 `docker-compose.yml`에는 동일 서비스가 없으므로 두 Compose의 구성 범위를 혼동하지 않습니다.

- `app-rules-engine`은 `EDGEX_PROFILE=cctv-external-http`로 실행
- 트리거: `external-mqtt`
- 구독 토픽: `cctv/rules/intrusion/persisted,cctv/rules/intrusion/critical`
- 파이프라인: `HTTPExport -> http://cctv-alert-api:8000/api/alerts`
- 호스트 확인 포트: `127.0.0.1:59701` → 컨테이너 `59707`
- `StoreAndForward.Enabled=false`, `PersistOnError=false`

현재 ASC는 HTTP 전송 실패 payload를 자체 보관하거나 재시도하지 않습니다. 재전송이 필요한 경우 ASC store-and-forward를 별도로 설계해 활성화하거나, downstream 저장·outbox 계층에서 처리해야 합니다.

실행 후 로그 확인 기준:

- `Loading configuration file from /tmp/edgex-res/cctv-external-http/configuration.yaml`
- `External MQTT trigger selected`
- `Connected to mqtt server for MQTT trigger`
- `StoreAndForward disabled. Not running retry loop.`
- `Subscribed to topic(s) 'cctv/rules/intrusion/persisted,cctv/rules/intrusion/critical' for MQTT trigger`

확인 명령:

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml logs --tail 150 app-rules-engine
curl -fsS http://127.0.0.1:59701/api/v3/ping
```

정상 ping 응답의 `serviceName`은 `app-cctv-external-http`입니다. `No cluster leader` 또는 Consul 연결 오류가 반복되면 ASC 자체보다 `edgex-core-consul` 상태를 먼저 확인합니다.
