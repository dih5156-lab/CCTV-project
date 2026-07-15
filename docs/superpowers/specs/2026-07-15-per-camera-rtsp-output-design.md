# 카메라별 RTSP 분석 영상 출력 설계

## 목적

DeepStream의 다중 카메라 배치 추론 구조를 유지하면서, 각 카메라의 AI 분석 결과 영상을 독립된 RTSP/WebRTC 경로로 제공한다.

예상 경로는 다음과 같다.

```text
rtsp://cctv-media-server:8554/camera_1
rtsp://cctv-media-server:8554/camera_2
rtsp://cctv-media-server:8554/entrance
```

POC 단계에서는 객체 탐지 박스, 추적 정보, 낙상 표시 등 `nvdsosd` 결과가 포함된 영상을 출력한다.

## 현재 구조와 문제

현재 파이프라인은 여러 입력을 `nvstreammux`에서 배치 처리하지만, 마지막 H.264 인코더와 `rtspclientsink`는 하나뿐이다. 출력 URL도 `DS_RTSP_LOCATION`의 단일 값으로 고정되어 있어 카메라별 라이브 경로를 제공할 수 없다.

단순히 URL 문자열만 카메라 ID로 치환하면 배치 버퍼를 카메라별 영상으로 분리할 수 없으므로 요구사항을 충족하지 못한다.

## 선택한 구조

AI 추론, 추적, OSD까지 기존 배치 파이프라인을 공유하고 그 이후에 `nvstreamdemux`를 배치한다. demux의 `src_<pad_id>`를 해당 카메라 전용 출력 브랜치에 연결한다.

```text
카메라별 nvurisrcbin
        ↓
   nvstreammux
        ↓
추론 → 추적 → OSD
        ↓
  nvstreamdemux
   ├─ src_0 → queue → 변환 → H.264 HW 인코더 → RTSP camera_1
   ├─ src_1 → queue → 변환 → H.264 HW 인코더 → RTSP camera_2
   └─ src_2 → queue → 변환 → H.264 HW 인코더 → RTSP entrance
```

카메라 추가·삭제로 입력 배치가 달라지면 기존 파이프라인 재시작 방식을 그대로 사용한다. 재구축 시 카메라 ID와 pad ID 매핑을 기준으로 출력 브랜치도 함께 다시 만든다.

## URL과 설정

새 기본 설정은 카메라 ID 자리표시자를 포함하는 템플릿이다.

```env
DS_RTSP_LOCATION_TEMPLATE=rtsp://cctv-media-server:8554/{camera_id}
```

`{camera_id}`는 각 활성 카메라 ID로 치환한다. URL 충돌과 잘못된 GStreamer 경로를 막기 위해 카메라 ID는 영문자, 숫자, `_`, `-`만 허용한다. 유효하지 않은 ID가 있으면 파이프라인 생성 전에 해당 ID를 포함한 명확한 오류를 발생시킨다.

기존 단일 카메라 배포와의 호환성을 위해 `DS_RTSP_LOCATION_TEMPLATE`이 없고 `DS_RTSP_LOCATION`만 설정된 경우에는 다음 규칙을 적용한다.

- 활성 카메라가 하나이면 기존 `DS_RTSP_LOCATION` 값을 그대로 사용한다.
- 활성 카메라가 둘 이상이면 URL 충돌을 막기 위해 설정 오류를 발생시키고 템플릿 사용을 안내한다.
- 두 설정이 모두 없으면 기본 템플릿을 사용한다.

Docker Compose의 기본 환경변수는 템플릿으로 변경한다. MediaMTX는 카메라 추가 때마다 설정 파일을 수정하지 않도록 동적 publisher 경로를 허용하되, 기존 `sample_eval` 경로는 유지한다.

## 구성 요소 변경

### 파이프라인 빌더

- 단일 출력 요소 목록 대신 카메라 ID와 pad ID를 가진 출력 브랜치 목록을 생성한다.
- 각 브랜치는 고유한 GStreamer 요소 이름을 사용한다.
- `nvstreamdemux`의 요청 pad와 브랜치 queue의 sink pad를 명시적으로 연결한다.
- 브랜치 queue는 지연 누적을 막기 위해 기존 출력 queue와 같은 leaky 정책을 사용한다.

### DeepStreamProcessor

- 활성 `source_entries`를 출력 브랜치 생성 함수에 전달한다.
- fakesink와 display 등 기존 비-RTSP 모드는 현재 단일 출력 동작을 유지한다.
- `rtsp-publish` 모드에서만 카메라별 demux 구조를 사용한다.
- 기존 추론 probe와 이벤트 처리 경로는 변경하지 않는다.

### MediaMTX 및 Compose

- `DS_RTSP_LOCATION` 기본값을 `DS_RTSP_LOCATION_TEMPLATE`로 교체한다.
- MediaMTX가 동적으로 게시되는 카메라 경로를 수락하도록 설정한다.
- WebRTC URL은 동일한 카메라 ID 경로를 사용한다.

## 오류 처리

- 허용되지 않은 카메라 ID: 파이프라인 시작 전에 `ValueError` 발생
- 다중 카메라에서 단일 legacy URL 사용: URL 충돌을 설명하는 설정 오류 발생
- demux pad 요청 또는 pad 연결 실패: 카메라 ID와 pad ID를 포함한 `RuntimeError` 발생
- 개별 RTSP publisher 연결 실패: GStreamer bus 오류에 카메라별 요소 이름이 남도록 구성

## 성능 영향

추론과 추적은 계속 배치로 공유되므로 모델 메모리는 카메라 수만큼 복제되지 않는다. 다만 변환과 H.264 인코더는 카메라마다 하나씩 필요하므로 카메라 수에 따라 NVENC 처리량과 메모리 사용량이 증가한다.

POC 기본값은 현재 해상도, FPS, 비트레이트를 모든 카메라에 공통 적용한다. 카메라별 화질 프로필이나 원본/분석 이중 스트림은 이번 범위에 포함하지 않는다.

## 테스트와 완료 조건

다음 조건을 자동 테스트로 검증한다.

- 카메라 ID별 템플릿 URL 생성
- 단일 카메라의 legacy `DS_RTSP_LOCATION` 호환
- 다중 카메라에서 legacy 단일 URL 거부
- 잘못된 카메라 ID 거부
- 카메라마다 고유한 H.264/RTSP 요소 이름 생성
- `nvstreamdemux` pad와 올바른 카메라 브랜치 연결
- 비-RTSP 출력 모드 회귀 방지
- Compose와 MediaMTX 설정 가정 검증

구현 후 관련 단위 테스트, 전체 테스트, Ruff, Docker Compose 설정 검증을 실행한다. Jetson 하드웨어가 필요한 실제 다중 RTSP 송출은 로컬 자동 테스트와 구분해 운영 장비에서 확인한다.

## 범위 제외

- 카메라별 비트레이트 및 해상도 프로필
- 원본 영상과 분석 영상의 동시 출력
- HLS 활성화 및 적응형 비트레이트
- 모바일 앱 UI 구현
- 카메라별 인증·접근 제어
