# 스피커 API 인수인계 문서

## 1. 문서 목적

이 문서는 CCTV Action Layer에서 사용하는 InterM 스피커의 HTTP API와 프로젝트 내부 제어 방식을 설명합니다.

> InterM API는 장치 직접 호출용입니다. Public API의 `/api/v1/control`은 `ACTION_LAYER_DEVICE_API.md`를 참고합니다.

## 2. 구현 위치

| 항목 | 경로 |
|---|---|
| 스피커 설정/클라이언트 | `src/devices/speaker.py` |
| Action Layer 연결 | `src/services/_action_bridge_executor.py` |
| 실행 진입점 | `runners/run_action_bridge.py` |
| 환경변수 | `.env.example`, `.env.jetson.example` |

## 3. 연결 정보와 인증

```text
http://{SPEAKER_HOST}:{SPEAKER_PORT}/interm-api
```

- HTTP Digest 인증
- 기본 포트 `80`
- `Content-Type: application/json`
- 연결 timeout 3초, 응답 timeout 7초
- host/계정/비밀번호 중 하나라도 없으면 비활성화

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `SPEAKER_HOST` | 빈 값 | 스피커 IP/호스트명 |
| `SPEAKER_PORT` | `80` | HTTP 포트 |
| `SPEAKER_USER` | 빈 값 | Digest 계정 |
| `SPEAKER_PASSWORD` | 빈 값 | Digest 비밀번호 |
| `SPEAKER_VOLUME` | `10` 또는 runner 기본값 `1` | 이벤트 방송 볼륨 |
| `SPEAKER_TTS_LANGUAGE` | `kor` | TTS 언어 |
| `SPEAKER_TTS_GENDER` | `female` | TTS 성별 |
| `SPEAKER_TTS_PITCH` | `100` | TTS 피치 |
| `SPEAKER_TTS_SPEED` | `100` | TTS 속도 |
| `SPEAKER_TTS_VOLUME` | `1` | TTS 음량 |

실제 계정과 비밀번호는 `.env` 또는 `.env.jetson`에만 입력합니다.

## 4. CCTV 이벤트 방송 흐름

`SpeakerDevice.play()`는 다음 순서로 동작합니다.

```text
이벤트 수신
  → /Audio/File/Status에서 TTS BGM 파일 확인
  → 파일이 없으면 TTS 생성
  → /TTS/Status에서 TTS ID 조회
  → TTS를 BGM으로 변환
  → 볼륨 설정
  → BGM 재생
```

이벤트별 문구는 `config/event_type_map.json`의 `tts_message`를 사용합니다. 직접 문구를 전달하면 해당 문구가 우선됩니다.

## 4-1. Action Layer 이벤트 Payload

MQTT 토픽은 `cctv/ai/events/{camera_id}/{event_type}`입니다. 스피커는 `tts_message`를 우선 사용합니다.

```json
{
  "event_id": "evt-20260903-0001",
  "camera_id": "camera_1",
  "type": "fall_detected",
  "severity": "critical",
  "confidence": 0.86,
  "tts_message": "낙상 사고가 감지되었습니다.",
  "metadata": {"fall_score": 5.2, "fall_direction": "back"}
}
```

`tts_message`가 없으면 `message`, 그 다음 이벤트 타입별 기본 문구를 사용합니다. Action Layer는 `{event_id}:speaker` command ID를 만들고 InterM API에 TTS 생성·변환·재생 요청을 순서대로 보냅니다. 전체 계약은 [디바이스 이벤트 Payload 계약](EVENT_PAYLOADS.md)을 참고합니다.

## 5. 주요 InterM API

### 5.1 TTS 생성

```http
POST /interm-api/TTS/Create
Content-Type: application/json
Authorization: Digest ...
```

```json
{
  "Title": "cctv_fall_detected",
  "Text": "낙상 사고가 감지되었습니다.",
  "Language": "kor",
  "Gender": "female",
  "Option": {
    "Pitch": 100,
    "Speed": 100,
    "Volume": 1,
    "SentencePause": 200,
    "CommaPause": 200
  },
  "Storage": "internal"
}
```

### 5.2 TTS 상태/변환/삭제

```http
GET  /interm-api/TTS/Status
POST /interm-api/TTS/ToBGM
POST /interm-api/TTS/Remove
```

변환과 삭제 요청:

```json
{"ID": ["<tts_id>"]}
```

`TTS/Status`의 `result.FileList`에서 제목이 일치하는 항목의 `ID`를 찾습니다.

### 5.3 BGM 파일 조회/삭제

```http
GET  /interm-api/Audio/File/Status
POST /interm-api/Audio/File/Remove
```

삭제 요청:

```json
{
  "Type": "BGM",
  "FileList": [{"FileHash": 12345}]
}
```

프로젝트는 `cctv_<event_type>` 제목의 파일을 재사용하고, 기동 시 오래된 `TTS_cctv_..._<timestamp>.wav` 파일을 정리합니다.

### 5.4 볼륨/재생/정지

```http
POST /interm-api/Audio/Output/PlayCtrl
```

볼륨:

```json
{
  "CHIndex": 1,
  "PlayType": "FilePlay",
  "ActionType": "Volume",
  "Volume": 10
}
```

재생:

```json
{
  "CHIndex": 1,
  "PlayType": "FilePlay",
  "ActionType": "Play",
  "Play": [{"FileHash": 12345, "FileLoopCount": 1}]
}
```

정지:

```json
{
  "CHIndex": 1,
  "PlayType": "FilePlay",
  "ActionType": "PlayStop"
}
```

### 5.5 전원

```http
POST /interm-api/System/Power
```

```json
{"Method": "On"}
```

허용 값은 `On`, `Off`, `Reboot`입니다.

### 5.6 음원 업로드/삭제

```http
POST /interm-api/Audio/File/Upload
Content-Type: multipart/form-data
```

업로드 필드:

| 필드 | 설명 |
|---|---|
| `File` | MP3/WAV 파일 |
| `Type` | `BGM` 또는 `CHIME` |
| `StorageType` | `Internal` 또는 `External` |

삭제는 `/Audio/File/Remove`를 사용합니다. BGM은 파일 해시, Chime은 파일명을 식별자로 사용합니다.

### 5.7 수동 방송

```http
POST /interm-api/Controller/Broadcast/Manual
```

```json
{
  "Action": "Start",
  "Start": {"SourceID": 1, "ZoneList": ["all"]}
}
```

정지는 `{"Action":"Stop","Stop":{"ZoneList":["all"]}}`, 전체 정지는 `{"Action":"AllStop"}` 형식입니다.

### 5.8 DSP

```http
GET  /interm-api/Audio/DSP/Input
POST /interm-api/Audio/DSP/Input
GET  /interm-api/Audio/DSP/Output
POST /interm-api/Audio/DSP/Output
```

IP-Speaker 출력 변경 예시:

```json
{"IsMute": false, "Volume": 50}
```

## 6. 프로젝트 내부 메서드

| 메서드 | 설명 |
|---|---|
| `play(event_type, severity, camera_id, text)` | 이벤트 TTS 생성/재생 |
| `stop()` | 현재 방송 정지 |
| `power_on()` / `power_off()` | 전원 제어 |
| `reboot()` | 스피커 재부팅 |
| `upload_file()` / `remove_file()` | 음원 관리 |
| `replace_file()` | 음원 교체 |
| `broadcast_start()` / `broadcast_stop()` | 수동 방송 |
| `broadcast_all_stop()` | 전체 방송 정지 |
| `broadcast_volume()` | 존별 볼륨 |
| `get_dsp_input()` / `set_dsp_input()` | DSP 입력 |
| `get_dsp_output()` / `set_dsp_output()` | DSP 출력 |

성공 여부는 대부분 `True/False`로 반환합니다. 네트워크 오류는 로그에 남기고 다른 이벤트 처리는 계속합니다.

## 7. 응답과 장애 대응

정상 응답은 장치에 따라 `Execute: OK` 또는 `result.Execute: OK` 형태입니다.

| 증상 | 확인 항목 |
|---|---|
| 비활성화 | host/계정/비밀번호 |
| 연결 timeout | IP/포트/네트워크 |
| TTS ID 없음 | 장치 처리 지연/제목 불일치 |
| 파일 해시 없음 | TTS 변환/파일 목록 반영 |
| 방송 중복 | Action Layer cooldown과 파일 재사용 |

펌웨어 업데이트는 `.imkp` 파일과 실제 장치 상태를 확인한 뒤 별도 승인으로 진행합니다.
