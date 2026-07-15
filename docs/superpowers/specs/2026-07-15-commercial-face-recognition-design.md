# 상용 얼굴 등록·식별 파이프라인 설계

## 1. 목표

Jetson 한 대에서 실시간으로 얼굴을 검출·정렬·임베딩하고, 등록된 사람을 1:N 검색해 출입 기록과 관심 인물 이벤트를 생성한다. 얼굴 모델은 배포 파일에 명시된 상용 사용 가능 라이선스만 허용하며, InsightFace 기본 pretrained weight는 운영 경로에서 제외한다.

초기 성공 기준은 다음과 같다.

- 등록 사진은 1장부터 허용하고 한 사람당 여러 장을 추가할 수 있다.
- 카메라 1대, 동시 얼굴 1~3명 조건에서 1초 이내에 식별 결과를 생성한다.
- 같은 track을 매 프레임 재추론하지 않고 기본 0.75초 간격으로 제한한다.
- 얼굴 식별 실패가 DeepStream의 헬멧·낙상·외형 분석을 중단시키지 않는다.
- 운영 모델 artifact마다 출처 URL, 버전, SHA-256, 라이선스 파일을 기록한다.
- 실제 등록·카메라 샘플로 임계값을 검증하기 전에는 출퇴근이나 관심 인물 신원을 자동 확정하지 않는다.

## 2. 선택한 모델

### 얼굴 검출: OpenCV YuNet

- 공식 OpenCV 배포 파일의 라이선스: MIT
- 역할: 얼굴 bbox, 양쪽 눈, 코, 양쪽 입꼬리의 5개 landmark 출력
- 초기 실행 경로: OpenCV `FaceDetectorYN` CPU 비동기 worker
- 선택 이유: 가볍고 5-point landmark를 함께 제공하며 현재 프로젝트의 OpenCV 의존성을 그대로 사용한다.

### 얼굴 임베딩: OpenCV SFace

- 공식 OpenCV 배포 파일의 라이선스: Apache License 2.0
- 역할: 정렬 얼굴을 고정 길이 임베딩으로 변환
- 초기 실행 경로: ONNX를 Jetson에서 TensorRT FP16 engine으로 변환해 직접 실행
- 선택 이유: 5-point 정렬 규격과 공개 평가 자료가 있고, InsightFace pretrained weight보다 상용 배포 조건이 명확하다.

### InsightFace 처리

- `buffalo_l`, `det_10g`, `w600k_r50` pretrained weight와 생성 engine은 연구 비교용으로만 분류한다.
- 운영 Docker image, 모델 manifest, 배포 패키지에는 포함하지 않는다.
- InsightFace 코드나 정렬 개념을 참고할 때는 해당 코드 파일의 라이선스와 저작권 고지를 유지한다.
- 향후 별도 상용 라이선스 또는 권리가 확보된 자체 weight가 생기면 `FaceEmbeddingBackend` 구현체로 다시 추가할 수 있다.

## 3. 아키텍처

```text
RTSP camera
    |
    v
DeepStream person tracking
    |  person bbox + camera_id + track_id
    v
Bounded asynchronous face worker
    |-- YuNet face detection and five landmarks
    |-- five-point similarity-transform alignment
    |-- SFace TensorRT embedding
    `-- FaceGallery 1:N cosine search
             |
             v
name / person_id / category / similarity / decision
             |
             v
track cache -> face event -> attendance/watchlist policy
```

DeepStream 영상 처리 thread는 얼굴 결과를 기다리지 않는다. bounded queue와 track cache를 사용하고, 얼굴 worker가 unavailable이면 얼굴 필드만 생략한다.

## 4. 구성 요소

### 4.1 모델 artifact 관리자

- YuNet과 SFace는 OpenCV 공식 Hugging Face 저장소의 고정 revision URL에서 받는다.
- 각 artifact의 SHA-256을 모델 manifest에 고정한다.
- 모델 파일과 함께 원본 LICENSE 파일을 저장한다.
- hash 또는 license 파일이 맞지 않으면 빌드와 배포 readiness 검사를 실패시킨다.
- 개발자가 임의 URL이나 `latest` 별칭으로 모델을 교체하지 못하게 한다.

### 4.2 YuNet 얼굴 검출기

- 사람 bbox의 상단 ROI만 입력해 전체 frame 반복 검출을 피한다.
- 결과 bbox와 5개 landmark를 원본 frame 좌표로 복원한다.
- 너무 작거나 흐리거나 frame 경계를 벗어난 얼굴은 임베딩 단계로 보내지 않는다.
- 초기 최소 얼굴 크기는 40px로 두고 현장 데이터로 조정한다.
- confidence와 NMS threshold는 환경변수로 노출하되 검증된 기본값을 문서화한다.

### 4.3 5-point 얼굴 정렬

- YuNet의 두 눈, 코, 두 입꼬리를 SFace 기준 좌표로 similarity transform한다.
- 정렬 결과는 SFace가 요구하는 고정 입력 크기로 만든다.
- landmark가 비정상적이거나 transform 계산이 불가능하면 해당 얼굴을 `low_quality`로 제외한다.
- 정렬 crop은 기본적으로 디스크에 저장하지 않는다.

### 4.4 SFace TensorRT 임베딩

- OpenCV 공식 SFace ONNX 입력·전처리 규격을 그대로 사용한다.
- TensorRT engine은 배포 대상 Jetson에서 FP16, batch 1로 생성한다.
- 출력 벡터는 float32로 변환하고 L2 정규화한다.
- NaN, infinity, zero norm, 예상하지 않은 shape는 오류로 처리한다.
- runtime 오류는 worker 내부에 격리하고 DeepStream process를 종료하지 않는다.

### 4.5 FaceGallery

모델과 저장소를 분리하기 위해 다음 인터페이스를 사용한다.

```python
class FaceGallery:
    def enroll(self, person, embeddings): ...
    def search(self, embedding, top_k=5): ...
    def update(self, person_id, embeddings): ...
    def deactivate(self, person_id): ...
    def delete(self, person_id): ...
    def reload(self): ...
```

- 사람 metadata는 SQLite에 저장한다.
- 임베딩은 `(N, D)` float32 행렬로 메모리에 로드해 vectorized cosine search를 수행한다.
- 한 사람의 여러 등록 임베딩을 유지하고 최고 점수와 상위 샘플 통계를 함께 반환한다.
- 등록 인원이 커져도 호출부를 바꾸지 않도록 향후 pgvector 또는 ANN 구현체로 교체 가능하게 한다.
- 모델 ID와 전처리 버전이 다른 임베딩은 같은 gallery에서 비교하지 않는다.

### 4.6 등록 정책

- 최소 등록 사진은 1장이다.
- 1장만 등록된 사람은 `single_sample` 상태로 표시한다.
- 권장 사진은 정면·좌우·다른 조명 조건의 3~5장이다.
- 등록 시 사진별로 얼굴이 정확히 1개인지 검사한다.
- 얼굴이 없거나 여러 개이거나 품질 기준 미달이면 등록을 거부한다.
- 모델이 바뀔 때 재임베딩할 수 있도록 동의받은 등록 원본을 암호화 저장한다.
- 카메라에서 반복 인식된 얼굴을 자동 등록하지 않는다. 관리자 승인 후에만 샘플을 추가한다.

### 4.7 식별 판정

- 최고 cosine similarity가 검증된 threshold 미만이면 `unknown`이다.
- 최고 후보와 두 번째 후보의 점수 차이가 검증된 margin 미만이면 `ambiguous`이다.
- `single_sample` 등록자는 더 보수적인 threshold를 적용할 수 있게 한다.
- track 내 여러 프레임 결과를 누적해 단일 frame 오인식을 억제한다.
- 이벤트에는 모델 ID, similarity, margin, 품질 점수, 판정 근거를 포함한다.

### 4.8 출입 및 관심 인물 정책

- 사람 category는 `employee`, `watchlist`, `visitor`를 지원한다.
- employee는 cooldown과 카메라 역할을 기준으로 출근·퇴근 이벤트를 만든다.
- watchlist는 높은 threshold와 연속 frame 확인을 모두 만족할 때만 알림 후보를 만든다.
- 임계값이 애매한 결과는 자동 확정하지 않고 관리자 검토 대상으로 저장한다.
- liveness가 구현되기 전까지 출퇴근 기록은 사진 재생 공격 가능성이 있음을 UI와 운영 문서에 표시한다.

## 5. API 및 데이터

기존 얼굴 등록 API의 외부 형식은 최대한 유지한다. 내부 저장소에는 다음 필드를 추가한다.

```json
{
  "person_id": "employee-001",
  "name": "registered-person",
  "category": "employee",
  "active": true,
  "sample_count": 1,
  "enrollment_status": "single_sample",
  "embedding_model": "opencv-sface-tensorrt-v1",
  "preprocessing_version": "yunet-5point-v1",
  "gallery_version": 1
}
```

식별 결과는 다음 내부 계약을 사용한다.

```json
{
  "matched": true,
  "decision": "matched",
  "person_id": "employee-001",
  "name": "registered-person",
  "category": "employee",
  "similarity": 0.73,
  "second_best_similarity": 0.44,
  "margin": 0.29,
  "quality_score": 0.86,
  "model_id": "opencv-sface-tensorrt-v1"
}
```

## 6. 장애 처리

- 얼굴 queue가 가득 차면 오래된 중간 frame 요청을 버리고 최신 요청을 우선한다.
- YuNet 또는 SFace 로드 실패 시 얼굴 backend를 unavailable로 표시한다.
- TensorRT native 오류가 메인 DeepStream process에 전파되지 않도록 최종 운영에서는 별도 로컬 얼굴 서비스로 격리한다.
- 얼굴 서비스 timeout 시 기존 track cache가 TTL 안에 있으면 마지막 결과를 사용하고, 아니면 얼굴 metadata를 생략한다.
- gallery reload 실패 시 검증된 이전 snapshot을 계속 사용한다.
- 모델 artifact hash 불일치는 폴백하지 않고 배포 readiness 실패로 처리한다.

## 7. 보안·개인정보·라이선스

- 얼굴 이미지와 임베딩은 생체정보로 취급한다.
- 등록 목적, 보관 기간, 삭제 절차와 접근 권한을 명시하고 동의를 기록한다.
- 등록 원본과 gallery는 저장 시 암호화하고 API 접근을 감사 로그에 남긴다.
- 로그에 얼굴 image bytes와 전체 embedding을 출력하지 않는다.
- 탈퇴·퇴사·관심 인물 해제 시 원본, 임베딩, cache, 서버 복제본을 삭제할 수 있어야 한다.
- YuNet MIT 저작권 고지와 SFace Apache 2.0 LICENSE/NOTICE 의무를 배포 패키지에 포함한다.
- 이 설계의 라이선스 검토는 기술적 검토이며 최종 상용 출시 전 법률 전문가의 확인을 받는다.

## 8. 검증 계획

### 라이선스·artifact 검증

- 고정 revision URL과 SHA-256 검증
- YuNet MIT 및 SFace Apache 2.0 파일 존재 검증
- 운영 manifest와 Docker context에 InsightFace pretrained artifact가 없는지 검사

### 단위 검증

- YuNet 결과의 ROI→frame 좌표 복원
- 5-point 정렬 기준점과 잘못된 landmark 거부
- SFace 전처리, output shape, L2 정규화
- gallery 등록·수정·비활성화·삭제·top-k 검색
- threshold, margin, single-sample 정책
- track cooldown과 누적 판정

### 모델 검증

- OpenCV 공식 ONNX와 TensorRT 출력 cosine 일치도 비교
- 동일인·타인 pair의 score 분포 측정
- 사진 1장 등록과 3장 등록의 FRR 비교
- 정면·측면·안경·모자·조명 변화 샘플 평가
- 실제 카메라에서 FAR, FRR, p95 latency 측정

### 통합·운영 검증

- 등록 API로 1장 등록 후 카메라 식별
- 동일인 추가 사진 등록 후 gallery hot reload
- 얼굴 서비스 강제 종료 중 DeepStream 프레임·낙상·헬멧 이벤트 유지
- 최소 30분 안정성 감시에서 frame drop, restart, FD 증가 확인
- 모델 artifact 제거 또는 변조 시 readiness 실패 확인

## 9. 단계적 적용

1. 공식 YuNet/SFace artifact와 LICENSE를 고정 revision·hash로 가져온다.
2. YuNet 검출과 5-point 정렬을 독립 샘플에서 검증한다.
3. SFace ONNX를 TensorRT engine으로 변환하고 출력·지연시간을 검증한다.
4. SQLite metadata와 vectorized `FaceGallery`를 구현한다.
5. 기존 등록 API를 새 gallery와 연결한다.
6. DeepStream 비동기 얼굴 worker와 track cache를 연결한다.
7. 실제 등록자 데이터로 threshold·margin을 측정한다.
8. 장애 격리와 운영 검증을 통과한 뒤 기본 얼굴 backend를 전환한다.

각 단계에서 기존 OpenCV 폴백은 유지한다. 새 경로가 실제 카메라 검증을 통과하기 전에는 현재 운영 설정을 변경하지 않는다.

## 10. 영향 범위

- 얼굴 모델 artifact와 라이선스 manifest가 변경된다.
- 얼굴 검출·정렬·임베딩 backend가 추가된다.
- 얼굴 등록 저장소가 여러 임베딩과 모델 버전을 지원하도록 확장된다.
- DeepStream 얼굴 context가 동기 직접 호출에서 bounded 비동기 호출로 이동한다.
- 기존 얼굴 API의 기본 필드는 유지하며 category, sample count, model version 필드가 추가된다.
- 낙상·헬멧·외형·RTSP·MQTT 기존 경로는 변경하지 않는다.
