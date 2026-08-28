# EdgeX 전광판 프로파일 및 Command 계약

## 목적

전광판을 EdgeX의 표준 장치로 등록하기 위한 프로파일과 Command 이름을 정의한다.
현재 운영 전광판은 Action Layer의 Dabit TCP 드라이버로 직접 제어되며, 이 프로파일은 EdgeX 메타데이터 계약을 먼저 고정하기 위한 것이다.

## 프로파일

`edgex/device-profiles/cctv-signboard-dabit-profile.yaml`

| Command | 리소스 | 용도 |
| --- | --- | --- |
| `display` | `display_text`, `display_color`, `brightness` | 문구 표시 |
| `clear` | `display_text` | 기본 화면 복귀 |
| `power` | `power` | 전원 켜기/끄기 |

## EdgeX API 예시

장치 등록 후 Core Command API는 다음 형식을 사용한다.

```text
GET  /api/v3/device/name/cctv-signboard-01/display
PUT  /api/v3/device/name/cctv-signboard-01/display
PUT  /api/v3/device/name/cctv-signboard-01/clear
GET  /api/v3/device/name/cctv-signboard-01/power
PUT  /api/v3/device/name/cctv-signboard-01/power
```

`display` 요청 값 예시:

```json
{
  "display_text": "낙상 감지 - 즉시 확인",
  "display_color": 1,
  "brightness": 10
}
```

## 운영 전환 조건

프로파일 업로드만으로 Dabit TCP 통신이 실행되지는 않는다. `device-rest`는 메타데이터 등록용으로 사용할 수 있지만, 실제 전광판 명령을 수행하려면 Dabit TCP를 구현한 전용 EdgeX Device Service가 필요하다.

따라서 전환 순서는 다음과 같다.

1. 프로파일·장치 등록
2. 전용 Device Service에서 EdgeX Command를 Dabit 버퍼로 변환
3. `command_id` 기준 `sent/acknowledged/failed` 응답 발행
4. Action Layer Shadow 비교
5. 검증 후 EdgeX Command 경로를 운영 경로로 승격
