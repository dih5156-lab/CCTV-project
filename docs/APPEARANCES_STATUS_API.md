# `/api/v1/appearances/status` 계약 문서

## 목적

이 엔드포인트는 대시보드에서 외형 검색 기능을 붙일 때

- 어떤 필드를 보여줄지
- 어떤 필드를 비활성화할지
- 값이 비는 원인이 설정 문제인지, 데이터 부족인지

를 한 번에 판단하기 위한 상태 API입니다.

단순히 DB 통계만 보여주는 API가 아니라,
`현재 런타임 설정 + 실제 적재 데이터`를 함께 해석한 결과를 반환합니다.

---

## 엔드포인트

- Method: `GET`
- Path: `/api/v1/appearances/status`
- Auth: `X-API-Key` 또는 `api_key` 쿼리 파라미터

응답 래퍼는 공개 API 공통 형식과 동일합니다.

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2026-04-24T00:00:00Z"
}
```

---

## 응답 구조

### `data.db_path`

- 현재 조회 중인 외형 로그 DB 경로
- 운영에서 `appearance.db`와 `appearances.db` 혼선을 피하기 위한 값

### `data.backend`

- 현재 외형 속성 백엔드
- 예:
  - `hsv`
  - `pphuman`

### `data.fields`

필드별 준비 상태 목록입니다.

각 항목 구조:

```json
{
  "field": "has_backpack",
  "enabled": true,
  "ready": false,
  "source": "yolo_nearby_objects",
  "observed_count": 0,
  "observed_ratio": 0.0,
  "reason": "backend=hsv, bag_labels=none"
}
```

의미:

- `field`
  - 대시보드 검색/필터에서 사용하는 필드 이름
- `enabled`
  - 설정상 이 기능이 켜져 있는지
- `ready`
  - 현재 런타임 기준으로 실제 적재가 가능한지
- `source`
  - 값 생성 경로
- `observed_count`
  - 현재 DB에서 실제로 채워진 건수
- `observed_ratio`
  - 전체 적재 대비 실제 채워진 비율
- `reason`
  - `ready=false`일 때 주된 원인

권장 해석:

- `enabled=false`
  - 기능 토글이 꺼져 있거나 상위 경로가 비활성화된 상태
- `enabled=true`, `ready=false`
  - UI는 켤 수 있지만 현재 모델/라벨/백엔드 구조상 실제 적재는 어려운 상태
- `enabled=true`, `ready=true`, `observed_count=0`
  - 설정은 맞지만 실제 데이터가 아직 없거나 품질 문제가 있는 상태

---

## 주요 필드 해석 기준

### `gender`

- source: `face_recognition`
- 일반적으로 `DS_FACE_ENABLED`와 얼굴 인식 경로에 의존
- `ready=true`인데도 `observed_count=0`이면 얼굴 crop 품질 또는 얼굴 인식 경로를 먼저 확인

### `has_helmet`

- source: `helmet_detection`
- 일반적으로 `DS_APPEARANCE_ENABLED`, `DS_HELMET_ENABLED`에 의존
- `ready=true`인데도 `observed_count=0`이면 헬멧 이벤트 생성 또는 사람 머리와의 정합을 확인

### `has_backpack`, `has_handbag`, `has_suitcase`

- `backend=hsv`일 때는 보통 `yolo_nearby_objects`
- `backend=pphuman` 등 속성 모델일 때는 `attribute_backend`
- `backend=hsv` + bag label 없음이면 `ready=false`

bag label alias 예:

- backpack 계열: `backpack`, `back_pack`, `rucksack`
- handbag 계열: `handbag`, `hand_bag`, `purse`
- suitcase 계열: `suitcase`, `luggage`, `travel_bag`, `carry_on`

---

## `data.data_stats`

DB 기준 누적 적재 통계입니다.

```json
{
  "total_records": 3238,
  "gender_filled": 2609,
  "helmet_true": 0,
  "backpack_true": 0,
  "handbag_true": 0,
  "suitcase_true": 0,
  "latest_timestamp": 1713846825.0
}
```

주의:

- 이 값은 **현재 DB에 쌓인 결과**입니다.
- 코드가 수정되어도 과거 데이터는 자동으로 바뀌지 않습니다.

---

## `data.backend_counts`

`appearance_log.attribute_backend` 기준 집계입니다.

예:

```json
{
  "unknown": 3238,
  "hsv": 120,
  "pphuman": 54
}
```

해석:

- `unknown`
  - 과거 적재 데이터이거나, `attribute_backend`가 비어 있던 시점의 데이터
- `hsv`, `pphuman`
  - 신규 적재분에서 어떤 백엔드가 실제로 사용됐는지 보여줌

---

## `data.warnings`

운영자가 바로 읽을 수 있는 진단 메시지입니다.

예:

- `현재 DB는 과거 적재분 위주라 attribute_backend가 모두 unknown 입니다.`
- `has_helmet는 설정상 활성화되어 있지만 실제 적재 건수가 0입니다.`
- `backend=hsv 환경에서는 bag 값이 detector nearby_objects에 의존합니다.`

프론트 권장 사용:

- 경고 배너
- 필터 옆 보조 설명
- 운영자용 진단 패널

---

## `data.next_steps`

바로 다음에 무엇을 확인해야 하는지 제안하는 운영 가이드입니다.

예:

- `AI 엔진 재시작 후 /api/v1/appearances/status 를 다시 조회해 ...`
- `has_helmet가 계속 0이면 헬멧 이벤트 로그와 사람 머리 bbox 정합 여부를 먼저 확인하세요.`

프론트 권장 사용:

- 운영 진단 화면의 “다음 조치” 섹션

---

## 대시보드 권장 사용 방식

### 1. 검색 필터 표시 조건

- `enabled=true` 인 필드만 기본 표시
- `ready=false` 인 필드는 비활성 상태로 표시하고 `reason` 툴팁 제공

### 2. 운영 경고 표시 조건

- `warnings.length > 0` 이면 상단 배너 또는 진단 패널 표시

### 3. 기능 on/off UI 기준

- 단순 토글 상태는 `enabled`
- 실제 사용 가능 여부는 `ready`

즉, UI에서는 `enabled`와 `ready`를 같은 의미로 다루면 안 됩니다.

---

## 운영 체크 순서

1. `db_path`가 기대한 DB를 보고 있는지 확인
2. `backend`가 현재 배포 설정과 일치하는지 확인
3. `fields[].ready`가 설정/라벨/모델 구조와 맞는지 확인
4. `backend_counts`에서 신규 적재분이 `unknown` 밖으로 나오는지 확인
5. `observed_count`와 `warnings`로 실제 적재 품질 확인

---

## 현재 계약의 한계

- 이 API는 “설정상/구조상 가능한지”와 “현재 DB에 실제로 쌓였는지”를 보여줍니다.
- 하지만 카메라별 세부 품질 편차, 프레임별 누락률, 모델 precision/recall까지 직접 측정하지는 않습니다.
- 정밀 품질 분석은 별도 로그/메트릭/샘플링 확인이 필요합니다.
