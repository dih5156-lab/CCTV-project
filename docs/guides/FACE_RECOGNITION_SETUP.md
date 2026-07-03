# Face Recognition Setup

이 프로젝트의 얼굴 인식은 환경별 역할을 나눠서 운영하는 것을 권장합니다.

## 권장 역할 분리

- Windows: 개발/디버깅/웹 UI/API 테스트
- Jetson: 실제 얼굴 인식 운영

## 현재 코드 구조

- `src/utils/face_recognition.py`
  - `InsightFace + ONNX Runtime`가 설치되어 있으면 실사용 얼굴 인식 사용
  - 설치되어 있지 않으면 OpenCV 기반 폴백 사용

즉, 같은 코드베이스를 유지하면서도 환경에 따라 동작 수준이 달라집니다.

## Windows 개발 환경

Windows에서는 기본적으로 공통 의존성만 설치합니다.

```bash
pip install -r requirements/ai.txt
```

이 환경에서는:

- 카메라/구역/이벤트/API 흐름 점검 가능
- 얼굴 등록/삭제 API 테스트 가능
- 이름 라벨 표시 흐름 점검 가능
- 다만 얼굴 인식 정확도는 운영 수준이 아닐 수 있음

## Jetson 운영 환경

Jetson에서는 Python 3.10 가상환경에서 아래 순서로 설치를 권장합니다.

```bash
pip install -r requirements/ai.txt
pip install -r requirements/jetson.txt
```

`requirements/jetson.txt`에는 InsightFace가 포함되어 있지만, 현재 `.env.jetson.example` 기본값은 안정적인 폴백 경로인 `FACE_RECOGNITION_BACKEND=opencv`입니다. InsightFace를 사용하려면 설치 후 명시적으로 다음 값을 설정하고 Jetson에서 초기화 로그를 확인합니다.

```bash
FACE_RECOGNITION_BACKEND=insightface
FACE_DEVICE=cpu
```

`auto` 또는 `insightface`에서 InsightFace import·초기화가 실패하면 OpenCV Haar 기반으로 폴백합니다. `FACE_DEVICE`가 `cuda` 또는 `cuda:N`으로 시작하면 GPU context를, 그 외에는 CPU context를 선택합니다. GPU 사용 여부는 실제 시작 로그와 Jetson 부하로 확인합니다.

## 등록 얼굴 데이터

- 메타데이터: `known_faces.json`
- 이미지 폴더: `known_faces/`

예시:

```json
[
  {
    "id": "hanface01",
    "name": "한기범",
    "image": "known_faces/han_gibeom_20260327.jpg"
  }
]
```

## 얼굴 등록 API

- `GET /faces`
- `POST /faces`
- `DELETE /faces/{face_id}`

`POST /faces` 예시:

```json
{
  "name": "한기범",
  "filename": "profile.jpg",
  "image_base64": "data:image/jpeg;base64,..."
}
```

## 카메라 설정

얼굴 인식을 켜려면 해당 카메라에 `face` 또는 `use_face`가 필요합니다.

```json
{
  "detections": ["helmet", "fall", "intrusion", "face"],
  "model_settings": {
    "use_helmet": true,
    "use_pose": true,
    "use_person": false,
    "use_face": true
  }
}
```

## 운영 팁

- 등록 사진은 1장보다 여러 장이 유리합니다.
- 정면/측면/조명 다른 사진을 함께 등록하는 것이 좋습니다.
- Windows에서 `unknown`이 나와도 Jetson 실모델 환경에서는 결과가 크게 달라질 수 있습니다.
