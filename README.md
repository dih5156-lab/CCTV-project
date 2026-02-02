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
`models/` 폴더에 다음 파일 배치:
- `helmet_model.pt` - 헬멧 감지 모델 (사용자 학습)
- `yolov8n-pose.pt` - 포즈 모델 (자동 다운로드)

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

### 기본 실행
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

### 예제
```bash
# 다중 카메라 + 위험 구역 감지
python main.py --cameras cameras.json --zone-detection --device cuda

# 데이터셋 수집 모드
python main.py --cameras cameras.json --collect-dataset

# 웹캠 테스트
python main.py --display
```

## 주요 모듈

### AppConfig (config.py)
중앙화된 설정 관리:
- ModelPaths: 모델 파일 경로 자동 탐지
- ServerConfig: 서버 통신 설정
- DetectionConfig: AI 감지 파라미터
- EventConfig: 이벤트 처리 설정

### AIAnalyzer (ai_analysis.py)
다중 모델 AI 추론:
- 헬멧(416→640px), 사람 포즈(640px) 모델 동시 실행
- YOLOv8 track() 활성화: 프레임 간 객체 ID 지속 (persistent tracking)
- 키포인트 기준 완화: 가림/후면 사람도 감지 (1개 키포인트 이상)
- IoU 기반 중복 박스 제거
- 낙상 감지: 어깨-엉덩이 각도 + 다리 높이 분석

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
| helmet_confidence | 0.45 | 헬멧 감지 최소 신뢰도 |
| pose_confidence | 0.5 | 사람 감지 최소 신뢰도 |
| iou_threshold | 0.3 | NMS IoU 임계값 |
| fall_angle_threshold | 30 | 낙상 감지 수평 각도 임계값 (도) |
| fall_height_ratio | 0.3 | 낙상 머리 높이 비율 |
| min_keypoint_confidence | 0.2 | 키포인트 최소 신뢰도 |
| detection_history_size | 5 | 누적 판정 프레임 수 |
| violation_threshold | 4 | 누적 판정 위반 임계값 |

### 시스템 설정
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| event_retention_hours | 24 | 이벤트 보관 시간 |
| debounce_seconds | 3.0 | 동일 이벤트 재전송 간격 |
| queue_max_size | 500 | 이벤트 큐 최대 크기 (3배 확대 설정) |
| frame_queue_size | 1 | 카메라당 프레임 큐 크기 (지연 최소화) |
| cumulative_detection_enabled | True | 누적 판정 활성화 |

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
