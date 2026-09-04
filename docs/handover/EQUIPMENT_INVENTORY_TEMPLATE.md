# 현장 장비 인벤토리

## 작성 규칙

실제 비밀번호·토큰은 기록하지 않는다. 주소와 ID만 관리하고 인증값은 secret 저장 위치를 기록한다. 변경 시 변경일·변경자·사유를 남긴다.

## Jetson / 서버

| 장비명 | 역할 | IP/호스트 | OS·JetPack | Docker 버전 | 설치 위치 | 담당자 | 상태 |
|---|---|---|---|---|---|---|---|
| `<입력>` | AI/EdgeX 서버 | `<입력>` | `<입력>` | `<입력>` | `<입력>` | `<입력>` | 운영/점검 |

### 저장소에서 확인되는 기본 서비스

| 서비스 | 기본 포트 | 역할 | 확인 방법 |
|---|---:|---|---|
| Public API | 9000 | 외부 조회·제어 | `GET /health` |
| Alert API | 8000 | 알림·이벤트 처리 | 서비스 로그/health |
| Action Layer | 8080 | 장치 제어 | `GET /health` |
| Stream API | 8769 | 영상 스트림 | `GET /health` |
| MQTT | 1883 | 이벤트 전달 | broker 연결/subscribe |
| EdgeX Core Data | 59880 | Reading 저장 | EdgeX health |
| EdgeX Core Metadata | 59881 | 장치·프로파일 | Metadata API |

## 카메라

| camera_id | 제조사/모델 | RTSP 별칭 | 해상도/FPS | 설치 위치 | 사용 모델 | ROI/zone | 상태 |
|---|---|---|---|---|---|---|---|
| `camera_1` | `<현장 확인>` | `camera_1` | `<현장 확인>` | `<현장 확인>` | pose/helmet/intrusion | `<현장 확인>` | enabled |
| `camera_2` | `<현장 확인>` | `camera_2` | `<현장 확인>` | `<현장 확인>` | helmet | `<현장 확인>` | disabled |
| `webcam` | 로컬 웹캠 | `0` | `<개발 PC 확인>` | 로컬 | helmet/fall | 없음 | disabled |

RTSP 계정은 문서가 아닌 secret에 보관한다. `camera_id`는 이벤트, 영상 저장, 검색 API에서 동일하게 유지해야 한다.

## 센서

| device_id | app_eui | dev_eui | 센서 종류 | 설치 위치 | 원본 MQTT | table | 규칙 |
|---|---|---|---|---|---|---|---|
| `<입력>` | `<입력>` | `<입력>` | tilt/temperature/vibration | `<입력>` | `<입력>` | `t<입력>` | `<입력>` |

## 출력 장치

| 논리 ID | 장치 | IP/호스트 | 포트 | 프로토콜 | Action 대상 | 설치 위치 | 담당자 |
|---|---|---|---:|---|---|---|---|
| `<입력>` | speaker | `<입력>` | 80 | InterM HTTP Digest | speaker | `<입력>` | `<입력>` |
| `cctv-signboard-01` | signboard | `<입력>` | 5000 | Dabit TCP | signboard | `<입력>` | `<입력>` |
| `<입력>` | siren | `<입력>` | 80 | InterM HTTP Digest | siren | `<입력>` | `<입력>` |

## 변경 이력

| 일시 | 대상 | 변경 내용 | 변경자 | 승인/티켓 |
|---|---|---|---|---|
| `<YYYY-MM-DD>` | `<입력>` | `<입력>` | `<입력>` | `<입력>` |
