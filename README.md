# CCTV 헬멧 착용 및 낙상 감지 시스템

건설 현장 안전 관리를 위한 실시간 CCTV 기반 AI 감지 시스템입니다. 헬멧 착용 여부와 낙상 사고를 자동으로 감지하고 서버로 이벤트를 전송합니다.

## 주요 기능

- **헬멧 착용 감지**: 작업자의 헬멧 착용(helmet_wearing) 및 미착용(helmet_missing) 상태 감지
- **낙상 감지**: YOLOv8-pose 모델을 이용한 관절 기반 낙상 감지
  - 몸 각도 분석 (기본: 45도 이하면 낙상)
  - 머리 높이 비율 분석 (기본: 바운딩 박스 하단 30% 이내면 낙상)
- **사람 감지**: YOLOv8-pose로 통합 (관절 정보 포함)
- **다중 카메라 지원**: 여러 RTSP 카메라 동시 처리
- **객체 추적**: 프레임 간 객체 ID 유지 및 추적
- **위험 구역 감지**: 지정된 구역 내 침입 감지
- **자동 데이터셋 수집**: YOLO 포맷으로 학습 데이터 자동 저장
- **서버 연동**: 감지된 이벤트 자동 전송
- **메모리 관리**: 24시간 이상 된 이벤트 자동 정리
- **재연결 메커니즘**: RTSP 연결 끊김 시 exponential backoff로 재시도

## 시스템 요구사항

### 하드웨어
- CPU: Intel Core i5 이상 (또는 동급)
- RAM: 8GB 이상 (16GB 권장)
- GPU: NVIDIA CUDA 지원 GPU (선택사항, 성능 향상)
- 저장 공간: 10GB 이상

### 소프트웨어
- Python 3.8 이상
- Windows 10/11 또는 Linux
- CUDA Toolkit 11.x 이상 (GPU 사용 시)

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd Project
```

### 2. 가상 환경 생성 (권장)
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
다음 위치 중 하나에 모델 파일 배치 (자동 탐지됨):
- `runs/detect/train/weights/test.pt` - 헬멧 모델
- `runs/detect/train/weights/best.pt` - 헬멧 모델 (대체)
- `helmet_model.pt` - 헬멧 모델 (프로젝트 루트)
- `yolov8n-pose.pt` - Pose 모델 (자동 다운로드 - 사람 + 관절 감지)

## 설정 방법

### 1. 환경 변수 설정 (권장)
```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 수정
HELMET_MODEL_PATH=runs/detect/train/weights/best.pt
SERVER_URL=http://your-server.com/api/events
DEVICE=cuda  # GPU 사용 시
```

### 2. 카메라 설정
```bash
# cameras.json.example을 참고하여 cameras.json 생성
cp cameras.json.example cameras.json

# cameras.json 수정
[
  {
    "id": "camera_1",
    "source": "rtsp://username:password@192.168.1.100:554/stream",
    "name": "Camera 1",
    "location": "Building A"
  }
]
```

### 3. 코드에서 직접 설정 (config.py)
```python
# config.py 파일 수정
DETECTION_CONFIG = {
    'helmet_confidence': 0.5,    # 헬멧 감지 신뢰도 (0.0-1.0)
    'person_confidence': 0.25,   # 사람 감지 신뢰도
    'fall_confidence': 0.7,      # 낙상 감지 신뢰도
    'imgsz': 1280,               # YOLO 입력 이미지 크기
    'iou_threshold': 0.3         # NMS IoU 임계값
}
```

### 4. 위험 구역 설정 (zones_config.json)
```json
{
  "camera_1": [
    {
      "name": "위험구역 1",
      "polygon": [[100, 100], [500, 100], [500, 400], [100, 400]],
      "color": [255, 0, 0]
    }
  ]
}
```

## 사용 방법

### 기본 실행 (웹캠)
```bash
python main.py --display
```

### 비디오 파일 테스트
```bash
python test_video.py
# 또는
python main.py --video test_video/sample.mp4 --display
```

### RTSP 카메라로 실행
```bash
python main.py --cameras cameras.json --server http://server.com/api/events
```

### GPU 사용
```bash
python main.py --device cuda --display
```

### 데이터셋 수집 모드
```bash
python main.py --collect-dataset --dataset-dir ./my_data --display
```

### 전체 옵션 확인
```bash
python main.py --help
```

### 주요 명령줄 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video` | 비디오 파일 경로 | 웹캠(0) |
| `--cameras` | 카메라 JSON 파일 | None |
| `--device` | cpu 또는 cuda | cpu |
| `--confidence` | 감지 신뢰도 (0.0-1.0) | 0.5 |
| `--display` | 화면 표시 활성화 | False |
| `--server` | 서버 URL | localhost:8000 |
| `--collect-dataset` | 데이터 수집 | False |
| `--zone-detection` | 위험 구역 감지 | False |

## 프로젝트 구조

```
Project/
├── main.py                  # 메인 실행 파일
├── config.py                # 중앙 설정 관리
├── processor.py             # 비디오 처리 및 워커 관리
├── ai_analysis.py           # AI 모델 추론
├── camera_input.py          # RTSP 카메라 관리
├── visualizer.py            # 시각화 및 UI
├── tracker.py               # 객체 추적
├── events.py                # 이벤트 타입 정의
├── server_comm.py           # 서버 통신
├── zone_detection.py        # 구역 감지
├── dataset_collector.py     # 데이터셋 자동 수집
├── test_video.py            # 비디오 파일 테스트
├── zones_config.json        # 위험 구역 설정
├── data.yaml                # YOLO 데이터셋 설정
├── utils/
│   ├── geometry.py          # 기하학 연산 유틸리티
│   └── logger.py            # 로깅 유틸리티
├── test_video/              # 테스트 비디오 파일
├── recorded_video/          # 녹화 파일 저장
├── dataset/                 # 자동 수집 데이터셋
└── README.md                # 프로젝트 문서
```

## 주요 클래스 및 모듈

### VideoProcessor (processor.py)
비디오 처리의 핵심 클래스로 다음 워커를 관리:
- **카메라 처리 워커**: 각 카메라별 독립 처리
- **이벤트 전송 워커**: 서버로 이벤트 비동기 전송
- **메모리 정리 워커**: 24시간 이상 된 이벤트 자동 삭제

### AIAnalyzer (ai_analysis.py)
다중 모델 AI 추론:
- `run_inference()`: 헬멧, 사람, 낙상 모델 동시 실행
- `_remove_duplicate_helmets()`: IoU 기반 중복 박스 제거
- 모델별 독립적인 confidence threshold 적용

### CameraInput (camera_input.py)
RTSP 카메라 관리:
- `get_frame()`: 프레임 획득 및 재연결
- Exponential backoff: 5s → 10s → 20s → 40s → 60s (max)

### SimpleTracker (tracker.py)
객체 추적:
- 프레임 간 객체 ID 유지
- IoU 기반 매칭

## 설정 파라미터 상세

### 감지 설정
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `helmet_confidence` | 0.5 | 헬멧 감지 최소 신뢰도 (높이면 false positive 감소) |
| `person_confidence` | 0.25 | 사람 감지 최소 신뢰도 |
| `fall_confidence` | 0.7 | 낙상 감지 최소 신뢰도 |
| `imgsz` | 1280 | YOLO 입력 크기 (작은 객체 감지 시 1280 권장) |
| `iou_threshold` | 0.3 | NMS IoU 임계값 (낮추면 중복 감소) |

### 시스템 설정
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `FALL_INFERENCE_INTERVAL` | 7 | 낙상 추론 프레임 간격 |
| `EVENT_RETENTION_HOURS` | 24 | 이벤트 보관 시간 |
| `CLEANUP_INTERVAL` | 3600 | 메모리 정리 간격 (초) |
| `CONSECUTIVE_FAILURE_THRESHOLD` | 5 | 서버 전송 실패 허용 횟수 |

## 성능 최적화

### GPU 사용
```python
# ai_analysis.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 처리 속도 향상
1. **이미지 크기 조정**: `imgsz=960` (1280에서 감소)
2. **낙상 감지 간격 증가**: `FALL_INFERENCE_INTERVAL=10`
3. **트래킹 최적화**: IoU threshold 조정
4. **멀티스레딩**: 카메라별 독립 처리

### 메모리 사용 최적화
- 24시간 이상 된 이벤트 자동 삭제
- GPU 메모리 정리: `torch.cuda.empty_cache()`
- 프레임 버퍼 크기 제한

## 문제 해결

### 헬멧이 감지되지 않는 경우
1. `helmet_confidence` 값을 0.3으로 낮추기
2. `imgsz`를 1280으로 유지 (작은 객체 감지)
3. 헬멧 크기 필터 확인: `h.height < 500 and h.width < 500`

### 중복 박스가 표시되는 경우
1. `iou_threshold`를 0.4로 증가
2. `_remove_duplicate_helmets()` 로직 확인

### 카메라 연결 실패
1. RTSP URL 및 인증 정보 확인
2. 네트워크 연결 상태 확인
3. 로그에서 exponential backoff 동작 확인

### False Positive가 많은 경우
1. `helmet_confidence`를 0.6으로 증가
2. 헬멧 모델 재학습 필요 (데이터셋 균형)
3. 현재 데이터셋: helmet_wearing 80%, helmet_missing 20%

## 데이터셋 수집

### 자동 수집 활성화
```python
processor = VideoProcessor(
    cameras=cameras,
    collect_dataset=True
)
```

### 수집된 데이터 위치
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### YOLO 포맷 라벨
```
class_id center_x center_y width height
```

## 모델 재학습

### 1. 데이터셋 준비
```bash
# data.yaml 확인
python dataset_collector.py
```

### 2. YOLOv8 학습
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

### 3. 모델 교체
```bash
cp runs/detect/train/weights/best.pt helmet_model.pt
```

## 운영 환경 체크리스트

### 배포 전 확인사항
- [ ] 모델 파일 준비 (helmet_model.pt, fall_model.pt)
- [ ] 카메라 URL 및 인증 정보 설정
- [ ] 서버 API 엔드포인트 설정
- [ ] 위험 구역 폴리곤 설정
- [ ] GPU 드라이버 및 CUDA 설치 (GPU 사용 시)
- [ ] 로그 파일 저장 위치 확인
- [ ] 24시간 안정성 테스트 완료

### 모니터링
- FPS 및 처리 속도
- 메모리 사용량
- 서버 전송 성공률
- 카메라 연결 상태
- 이벤트 큐 크기

## 알려진 이슈

1. **헬멧 모델 False Positive**: 
   - 원인: 학습 데이터 불균형 (helmet_wearing 80%, helmet_missing 20%)
   - 해결: 1500개 이상의 helmet_missing 이미지 추가 학습 필요

2. **작은 헬멧 감지 한계**:
   - 18-26px 크기의 헬멧은 낮은 신뢰도 (0.11-0.67)
   - `imgsz=1280` 유지 필요

3. **RTSP 재연결 지연**:
   - Exponential backoff로 최대 60초 지연 발생 가능
   - 네트워크 안정성 확인 필요

## 라이선스

이 프로젝트는 [라이선스 종류]에 따라 배포됩니다.

## 기여

버그 리포트 및 개선 제안은 이슈 트래커를 통해 제출해주세요.

## 연락처

- 프로젝트 관리자: [이름]
- 이메일: [이메일]
- 이슈 트래커: [URL]

## 변경 이력

### v1.0.0 (2025-12-22)
- 초기 릴리스
- 헬멧 착용/미착용 감지
- 낙상 감지
- 다중 카메라 지원
- 위험 구역 감지
- 자동 데이터셋 수집
- 메모리 관리 및 exponential backoff 구현
- 전체 코드 리팩토링 (constants, type hints, docstrings)
