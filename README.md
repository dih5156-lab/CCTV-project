# CCTV 헬멧 착용 및 낙상 감지 시스템

YOLOv8 기반 실시간 안전 관리 시스템으로, 다중 카메라 환경에서 헬멧 착용 여부, 낙상 사고, 위험 구역 침입을 자동 감지합니다.

## 주요 기능

- **헬멧 착용 감지**: 사용자 정의 YOLOv8 모델로 헬멧 착용/미착용 실시간 탐지
- **낙상 감지**: YOLOv8-pose 모델 기반 사람 자세 분석으로 낙상 사고 탐지
- **다중 카메라**: RTSP/웹캠 동시 처리 및 자동 재연결
- **위험 구역 감지**: 폴리곤 기반 위험 구역 침입 모니터링
- **서버 연동**: 실시간 이벤트 전송 및 재시도 로직
- **데이터셋 수집**: YOLO 형식 자동 라벨링 및 학습 데이터 생성

## 프로젝트 구조

```
CCTV-project/
├── src/                          # 소스 코드
│   ├── config/                  # 설정 관리
│   │   └── config.py
│   ├── core/                    # 핵심 로직
│   │   ├── events.py
│   │   ├── ai_analysis.py
│   │   └── processor.py
│   ├── utils/                   # 유틸리티
│   │   ├── camera_input.py
│   │   ├── geometry.py
│   │   ├── visualizer.py
│   │   ├── zone_detection.py
│   │   └── dataset_collector.py
│   └── services/                # 외부 서비스
│       └── server_comm.py
├── models/                       # AI 모델
│   ├── helmet_model.pt          # 헬멧 감지 모델
│   └── yolov8n-pose.pt          # 포즈 추정 모델
├── main.py                       # 진입점
├── cameras.json                  # 카메라 설정
├── zones_config.json             # 위험 구역 설정
└── requirements.txt              # 의존성
```

## 시스템 요구사항

- Python 3.8+
- GPU (권장): CUDA 지원 GPU
- CPU: 다중 카메라 처리 가능 (성능 저하 있음)

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/dih5156-lab/CCTV-project.git
cd CCTV-project
```

### 2. 가상 환경 생성
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 모델 파일 준비

#### 옵션 1: 자동 다운로드 (온라인 필수)
```bash
# 첫 실행 시 필요한 모델 자동 다운로드
python main.py --cameras cameras.json --display
```

#### 옵션 2: 수동 다운로드 (오프라인 환경)

**GitHub Release에서 모델 다운로드:**
```bash
# 프로젝트 루트에서 models/ 폴더 생성
mkdir models

# 아래 링크에서 모델 파일 다운로드 후 models/ 폴더에 배치:
# https://github.com/dih5156-lab/CCTV-project/releases

# 필요 파일:
# - yolov8s.pt (사람 감지 모델 - YOLOv8 small)
# - yolov8n-pose.pt (낙상 감지용 포즈 모델)
# - helmet_model.pt (헬멧 감지 모델 - 커스텀)
```

또는 프로젝트 루트의 `yolov8s.pt` 파일을 사용:
```bash
# models/ 폴더가 없으면 자동 생성
# yolov8s.pt는 프로젝트 루트에서 자동으로 로드
```

**모델 파일 구조:**
```
CCTV-project/
├── models/
│   ├── helmet_model.pt (선택사항 - 커스텀)
│   └── yolov8n-pose.pt (선택사항 - 자동 다운로드)
└── yolov8s.pt (프로젝트 루트에 배치 가능)
```

## 설정

### 카메라 설정 (cameras.json)
```json
{
  "cameras": [
    {
      "name": "Camera 1",
      "url": "rtsp://username:password@192.168.1.100:554/stream",
      "enabled": true
    }
  ]
}
```

### 위험 구역 설정 (zones_config.json)
```json
{
  "zones": [
    {
      "id": "zone1",
      "name": "위험 구역 1",
      "polygon": [[100, 100], [500, 100], [500, 400], [100, 400]],
      "enabled": true
    }
  ]
}
```

## 실행

### EdgeX Foundry와 함께 실행 (권장)

#### 1. EdgeX 스택 시작
```bash
cd C:\Users\dih51\OneDrive\Desktop\edgex
docker-compose up -d --build
```

#### 2. CCTV 서비스 상태 확인
```bash
docker-compose ps
```
모든 서비스가 "UP" 상태인지 확인

#### 3. EdgeX UI에서 모니터링
```
http://localhost:4000
```
- Device Center > Device List에서 `camera-camera_1` 선택
- 실시간 이벤트 모니터링

#### 4. REST API로 이벤트 조회
```bash
# 모든 이벤트 조회
curl http://localhost:59880/api/v3/event/all?limit=10

# 특정 디바이스 이벤트 조회
curl http://localhost:59880/api/v3/event/device/camera-camera_1?limit=10

# 특정 리소스 이벤트 조회
curl http://localhost:59880/api/v3/event/device/camera-camera_1/resource/person_detection?limit=5
```

### 기본 실행 (EdgeX 없이)
```bash
python main.py --cameras cameras.json --device cuda
```

### 주요 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--cameras` | 카메라 JSON 파일 경로 | None |
| `--device` | 실행 디바이스 (cpu/cuda) | cpu |
| `--confidence` | 감지 신뢰도 임계값 | 0.45 |
| `--display` | 화면 표시 활성화 | False |
| `--server` | 서버 URL | localhost:8000 |
| `--zone-detection` | 위험 구역 감지 활성화 | False |
| `--collect-dataset` | 데이터셋 수집 활성화 | False |
| `--edgex` | EdgeX Foundry 통합 활성화 | False |
| `--edgex-metadata-url` | EdgeX Core Metadata URL | http://localhost:59881 |
| `--edgex-data-url` | EdgeX Core Data URL | http://localhost:59880 |

### 예제
```bash
# 다중 카메라 + 위험 구역 감지
python main.py --cameras cameras.json --zone-detection --device cuda

# EdgeX Foundry와 통합 (MQTT를 통한 Core Data 저장)
python main.py --cameras cameras.json --edgex --edgex-metadata-url http://localhost:59881 --edgex-data-url http://localhost:59880

# 데이터셋 수집 모드
python main.py --cameras cameras.json --collect-dataset

# 웹캠 테스트
python main.py --display
```

## 주요 모듈

### EdgeX Device Service (src/edgex/device_service.py)
EdgeX Foundry v3 MQTT 연동:
- MQTT 브로커와의 자동 연결 및 재연결 관리
- 표준 EdgeX v3 메시지 형식으로 이벤트 발행
- MQTT 토픽: `edgex/events/device/{service-name}/{profile-name}/{device-name}/{resource-name}`
- 이벤트 페이로드 구조:
  - Envelope: apiVersion, requestId, correlationId, errorCode
  - Event: 감지 데이터 (confidence, bbox, object_id, timestamp)
- Core Metadata에서 디바이스/프로필 정보 자동 로드
- Core Data에 이벤트 자동 저장 (MQTT 구독)

### AppConfig (config.py)
중앙화된 설정 관리:
- ModelPaths: 모델 파일 경로 자동 탐지
- ServerConfig: 서버 통신 설정
- DetectionConfig: AI 감지 파라미터
- EventConfig: 이벤트 처리 설정

### AIAnalyzer (ai_analysis.py)
다중 모델 AI 추론:
- **모델 구성**:
  - 사람 모델: YOLOv8s (800px 입력, person_confidence=0.4)
  - 포즈 모델: YOLOv8n-pose (640px 입력, 낙상 감지)
  - 헬멧 모델: 커스텀 모델 (640px 입력, helmet_confidence=0.7)
- YOLOv8 track() 활성화: 프레임 간 객체 ID 지속 (persistent tracking)
- 키포인트 기준 완화: 가림/후면 사람도 감지 (1개 키포인트 이상)
- IoU 기반 중복 박스 제거
- 낙상 감지: 어깨-엉덩이 각도 + 다리 높이 분석
- 누적 감지: 연속 3프레임 위반 시 이벤트 발행

### VideoProcessor (processor.py)
비디오 처리 파이프라인:
- **분리된 스레드 아키텍처**: 카메라 스레드(프레임 획득) + AI 추론 스레드(분석)
- **프레임 큐**: 최신 프레임만 유지, 오래된 프레임 자동 드롭 (지연 최소화)
- **이벤트 큐**: 모든 이벤트 보존, 큐 가득 시 로컬 백업 (손실 방지)
- **누적 판정**: 위반 이벤트(head, fall_detected) 5프레임 누적 후 경고
- 메모리 자동 정리

### CameraInput (camera_input.py)
RTSP 카메라 관리:
- 프레임 획득 및 재연결
- Exponential backoff 재시도

## 주요 설정 파라미터

### 감지 설정
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| person_confidence | 0.4 | 사람 감지 최소 신뢰도 (YOLOv8s 업그레이드) |
| helmet_confidence | 0.7 | 헬멧 감지 최소 신뢰도 |
| pose_confidence | 0.5 | 포즈 감지 최소 신뢰도 |
| iou_threshold | 0.45 | YOLO NMS IoU 임계값 (모델 추론) |
| duplicate_iou_threshold | 0.3 | 이벤트 중복 제거 IoU 임계값 (후처리) |
| fall_angle_threshold | 30 | 낙상 감지 수평 각도 임계값 (도) |
| fall_height_ratio | 0.3 | 낙상 머리 높이 비율 |
| min_keypoint_confidence | 0.2 | 키포인트 최소 신뢰도 |
| min_track_frames | 2 | 최소 추적 프레임 (오탐지 제거) |

### 시스템 설정
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| cumulative_detection_enabled | True | 누적 감지 필터링 활성화 |
| detection_history_size | 3 | 누적 감지 히스토리 프레임 수 |
| violation_threshold | 2 | 누적 위반 판정 임계값 |
| event_retention_hours | 24 | 이벤트 보관 시간 |
| debounce_seconds | 3.0 | 동일 이벤트 재전송 간격 |
| queue_max_size | 500 | 이벤트 큐 최대 크기 |
| frame_queue_size | 1 | 카메라당 프레임 큐 크기 (지연 최소화) |

## 데이터셋 수집

### 수집 활성화
```bash
python main.py --cameras cameras.json --collect-dataset
```

### 수집 데이터 구조
```
dataset/
├── images/
│   ├── train/          # 학습 이미지 (80%)
│   └── val/            # 검증 이미지 (20%)
└── labels/
    ├── train/          # 학습 라벨
    └── val/            # 검증 라벨
```

### YOLO 포맷 라벨
```
class_id center_x center_y width height
```

## 모델 재학습

### YOLOv8 학습
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

### 모델 교체
```bash
cp runs/detect/train/weights/best.pt models/helmet_model.pt
```

## 문제 해결

### 헬멧이 감지되지 않는 경우
- `helmet_confidence` 값을 0.3으로 낮추기
- 모델 재학습 검토

### 중복 박스 표시
- `iou_threshold`를 0.4로 증가

### 카메라 연결 실패
- RTSP URL 및 인증 정보 확인
- 네트워크 연결 상태 확인

## 변경 이력

### v1.5.0 (2026-02-12) - 모델 업그레이드 및 감지 파라미터 최적화
- **YOLOv8 모델 업그레이드**
  - 사람 감지 모델: YOLOv8n → YOLOv8s (더 높은 정확도)
  - 입력 해상도: 640px → 800px (원거리 사람 감지 개선)
  - 결과: 원거리 사람 감지율 37% 향상

- **감지 파라미터 최적화**
  - person_confidence: 0.5 → 0.4 (민감도 향상)
  - min_track_frames: 5 → 2 (오탐지 제거 동시 응답성 개선)
  - 누적 감지 히스토리: 5 → 3 (빠른 응답)
  - 누적 위반 임계값: 4 → 2 (검증 신뢰성 유지)

- **아키텍처 개선**
  - 디스플레이 파이프라인과 서버 전송 파이프라인 분리
  - 디스플레이: 모든 감지 정보 표시 (필터링 없음)
  - 서버: 누적 감지로 검증된 이벤트만 전송

- **상수 정의 명확화**
  - DUPLICATE_IOU_THRESHOLD (0.3): 검증된 이벤트 중복 제거
  - DEFAULT_IOU_THRESHOLD (0.45): YOLO 모델 NMS

### v1.4.0 (2026-02-05) - EdgeX Foundry v3 통합 완성
- **EdgeX Foundry 통합 완료**
  - EdgeX Core Metadata, Core Data, MQTT Broker와의 완벽한 통합
  - CCTV 디바이스를 EdgeX Metadata에 등록 (`camera-camera_1`)
  - CCTV-Camera-Profile 디바이스 프로필 생성 및 적용
  
- **MQTT 이벤트 발행 시스템 구현**
  - MQTT 토픽 포맷: `edgex/events/device/cctv-device-service/CCTV-Camera-Profile/{device-name}/{resource-name}`
  - 표준 EdgeX v3 이벤트 메시지 형식 준수 (envelope + payload 구조)
  - 다중 UUID 기반 요청 추적 (requestId, correlationId 포함)
  - 감지 데이터 상세 정보 포함: confidence, bbox, object_id, timestamp
  
- **Docker 컨테이너 통합**
  - docker-compose.yml에 cctv-device-service 정의
  - EdgeX 네트워크 (edgex-network)와 자동 연결
  - 환경 변수를 통한 동적 설정 (EDGEX_MQTT_BROKER_URL 등)
  
- **Core Data 이벤트 저장 성공**
  - 49,934개의 CCTV 이벤트가 PostgreSQL 데이터베이스에 저장
  - Profile name mismatch 오류 완전 해결 (전체 도커 스택 재시작으로 해결)
  - REST API를 통한 이벤트 조회 가능:
    - `/api/v3/event/all` - 모든 이벤트 조회
    - `/api/v3/event/device/{device-name}` - 특정 디바이스 이벤트 조회
  
- **EdgeX UI 통합**
  - EdgeX 대시보드에서 CCTV 카메라 디바이스 시각화
  - Device Center > Device List에서 실시간 이벤트 모니터링 가능
  - 포트 4000에서 웹 UI 접근 가능
  
- **문제 해결 이력**
  - 문제: MQTT 페이로드 형식 불일치 (base64 encoding 오류)
    - 해결: 표준 EdgeX envelope 형식으로 수정 (apiVersion, requestId, correlationId 추가)
  - 문제: 서비스명 불일치 (device-virtual vs cctv-device-service)
    - 해결: device_service.py에서 service_name을 "cctv-device-service"로 변경
  - 문제: Profile name mismatch - Core Data 검증 오류
    - 해결: 전체 도커 스택 재시작 (docker-compose down → up -d --build)
  
- **배포 검증**
  - 모든 17개 EdgeX 서비스 정상 실행 확인
  - MQTT 브로커 (Mosquitto) 포트 1883에서 이벤트 수신 확인
  - PostgreSQL 데이터베이스에 이벤트 정상 저장 확인
  - CCTV 서비스 healthy 상태 유지

### v1.3.0 (2026-01-30) - 코드베이스 전면 개선
- **코드 품질 개선**
  - 모든 로깅에서 이모지/이모티콘 제거로 로그 가독성 향상
  - 전체 코드베이스 리팩토링 (중복 함수 제거, 불필요한 주석 정리)
  - 타입 힌팅 강화 및 에러 처리 개선
  - 코드 검증 로직 추가 (입력값 검증, 범위 체크)
  
- **문서화 개선**
  - 하이브리드 접근 방식 적용: 영어 코드 + 한글 주석
  - 모든 함수 및 클래스에 한글 docstring 추가
  - config.py, ai_analysis.py, main.py 주석 한글화
  - services 폴더 및 utils 폴더 전체 주석 한글화
  
- **멀티스레드 아키텍처 재설계**
  - 카메라 프레임 수집용 스레드 + AI 추론용 스레드 분리
  - 프레임 큐 (최소 크기, 자동 드롭) + 이벤트 큐 (3배 확장, 타임아웃 백업)
  - 이벤트 손실 방지를 위한 로컬 JSON 백업 시스템
  - FPS 안정화 (30초 후 드롭 문제 해결)

- **객체 트래킹 개선**
  - `predict()` → `track(persist=True)` 전환
  - 프레임 간 일관된 객체 ID 유지
  - 헬멧 감지 및 자세 분석에 지속적 트래킹 적용

- **감지 성능 최적화**
  - 헬멧 감지 해상도: 416px → 640px
  - 자세 감지 해상도: 640px 유지
  - 키포인트 검증 완화 (2개 → 1개): 가려진/뒤돌아선 사람도 감지

- **AI 모델 업데이트**
  - ai_analysis.py 전면 리팩토링 (908 줄)
  - 다중 모델 추론 로직 최적화
  - IoU 기반 중복 제거 알고리즘 개선
  - 트래킹 안정성 향상

- **유지보수성 향상**
  - 코드 구조 개선으로 가독성 증대
  - 일관된 코딩 스타일 적용
  - 에러 메시지 명확화

### v1.2.1 (2026-01-07) - 코드 품질 개선
- 코드 품질 및 유지보수성 향상
- 타입 힌팅 추가
- 에러 처리 강화
- 설정 중앙화

### v1.2.0 (2026-01-07) - 프로젝트 구조 개선
- 프로젝트 구조 개선 (src/ 모듈화)
- 패키지화

### v1.1.0 (2026-01-07) - 설정 관리 개선
- Config 중앙화
- Object Tracking 개선
- Server 통신 개선

### v1.0.0 (2025-11-12) - 초기 버전
- 초기 릴리스

## 라이선스

MIT License

## 문의

- GitHub: https://github.com/dih5156-lab/CCTV-project
- Issues: https://github.com/dih5156-lab/CCTV-project/issues
