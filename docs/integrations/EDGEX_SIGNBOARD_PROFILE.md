# EdgeX 전광판 프로파일 및 Command 계약

## 목적

전광판을 EdgeX의 표준 장치로 등록하기 위한 프로파일과 Command 이름을 정의한다.
현재 `cctv-device-dabit` Device Service가 EdgeX Command를 Dabit TCP 명령으로 변환한다.

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

프로파일 업로드만으로 Dabit TCP 통신이 실행되지는 않는다. 전광판 장치는 범용 `device-rest`가 아니라 Dabit TCP를 구현한 `cctv-device-dabit` Device Service에 연결한다.

따라서 전환 순서는 다음과 같다.

1. 프로파일·`cctv-device-dabit` 서비스·장치 등록
2. Core Command에서 `display`, `clear`, `power` 명령 호출
3. Device Service에서 EdgeX Command를 Dabit 버퍼로 변환
4. `command_id` 기준 `acknowledged/failed` 결과 확인
