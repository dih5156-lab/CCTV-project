# Kuiper Rule Engine (3단계)

## 역할

Rule Engine(Kuiper)은 다음만 담당합니다.

- intrusion 이벤트 필터링 (`type == danger_zone`)
- 5초 이상 지속 감지 판단
- confidence 임계값 필터링
- 결과 라우팅 (filtered / persisted / critical)

## 입력 / 출력 토픽

- 입력: `cctv/ai/events/+/+`
- 출력:
  - `cctv/rules/intrusion/filtered`
  - `cctv/rules/intrusion/persisted`
  - `cctv/rules/intrusion/critical`

## 규칙 팩

- 파일: `kuiper/rules/cctv_intrusion_rules.json`
- 포함 룰:
  1. `intrusion_confidence_filter`
  2. `intrusion_5s_persist`
  3. `intrusion_high_confidence_routing`

## 배포 방법

```bash
python run_kuiper_rules.py --kuiper-api http://localhost:9081 --mqtt-broker localhost --mqtt-port 1883 --intrusion-confidence 0.7 --critical-confidence 0.9 --persist-hit-count 5
```

## 임계값 조정 가이드

- `--intrusion-confidence`
  - 낮추면 민감도 증가, 오탐 증가 가능
- `--critical-confidence`
  - 즉시 대응이 필요한 알림 분기 기준
- `--persist-hit-count`
  - 5초 윈도우 내 검출 횟수 기준
  - FPS/발행 주기 높을수록 값을 올리고, 낮을수록 값을 내립니다

## 운영 흐름

1. AI Engine이 MQTT로 이벤트 발행
2. Kuiper가 intrusion 이벤트를 룰로 평가
3. 조건 충족 이벤트를 별도 토픽으로 라우팅
4. 다운스트림(알림, 저장, 대시보드)이 토픽별 소비
