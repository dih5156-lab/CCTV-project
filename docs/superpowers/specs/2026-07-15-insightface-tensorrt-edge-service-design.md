# InsightFace TensorRT 엣지 서비스 설계

## 1. 목표

Jetson 한 대에서 영상과 얼굴 데이터를 외부로 전송하지 않고 얼굴 인식을 수행한다. 얼굴 인식 런타임 장애가 헬멧·낙상·외형 분석과 RTSP 출력에 영향을 주지 않도록 얼굴 인식을 별도 로컬 서비스로 격리한다.

초기 성공 기준은 다음과 같다.

- 카메라 1대에서 동시에 보이는 얼굴 1~3명을 처리한다.
- 얼굴 crop 제출부터 인식 결과 수신까지 정상 부하에서 1초 이내를 목표로 한다.
- 얼굴 서비스 중단 시 DeepStream 파이프라인과 다른 이벤트 발행은 계속 동작한다.
- 얼굴 이미지와 임베딩은 Jetson 외부로 전송하지 않는다.
- 운영 경로에서는 현재 네이티브 크래시가 재현된 ONNX Runtime을 사용하지 않는다.

## 2. 범위

### 포함

- 동일 Jetson 내부의 얼굴 인식 전용 컨테이너
- TensorRT 기반 얼굴 임베딩 추론
- 얼굴 crop 요청과 인식 결과 응답을 위한 로컬 API
- track ID 기반 요청 제한과 결과 캐시
- 등록 얼굴 갤러리의 임베딩 생성 및 재생성
- timeout, 장애 격리, health check, 기본 메트릭
- 기존 OpenCV 얼굴 인식 경로를 비상 폴백으로 유지

### 제외

- 외부 클라우드 얼굴 인식 API
- 여러 Jetson 간 분산 추론
- 얼굴 등록 UI의 대규모 개편
- DeepStream 전체 파이프라인의 구조 변경
- 학습 또는 파인튜닝

## 3. 선택한 접근법

InsightFace의 얼굴 정렬 및 임베딩 모델을 TensorRT로 실행하는 전용 로컬 서비스를 추가한다. DeepStream 메인 컨테이너는 얼굴 crop을 비동기 제출하고 결과를 track ID에 결합한다.

ONNX Runtime 전용 컨테이너는 메인 프로세스 장애를 막을 수 있지만 이 Jetson에서 ONNX Runtime 1.22.1과 GPU 1.24.0 모두 네이티브 abort가 재현되어 운영안에서 제외한다. 얼굴 모델을 DeepStream SGIE로 직접 통합하는 방식은 장기 성능 최적화 후보지만 POC 범위가 커서 첫 구현에서는 제외한다.

## 4. 아키텍처

```text
RTSP camera
    |
    v
DeepStream AI engine
    |-- person / helmet / fall / appearance
    |-- face crop + camera_id + track_id
    v
Local face inference API (same Jetson)
    |-- validation and alignment
    |-- TensorRT embedding inference
    |-- cosine similarity search
    v
name + similarity + matched + model_version
    |
    v
Track context cache -> MQTT/event metadata
```

두 컨테이너는 외부 포트를 공개하지 않고 Compose 내부 네트워크로만 통신한다. 얼굴 등록 API가 필요하면 기존 AI 엔진 API가 프록시 역할을 하며 얼굴 추론 서비스 자체는 호스트에 노출하지 않는다.

## 5. 구성 요소

### 5.1 DeepStream 얼굴 요청 클라이언트

- 기존 사람 bbox와 track ID를 사용해 얼굴 후보 crop을 만든다.
- 같은 `(camera_id, track_id)` 요청은 기본 0.75초 간격으로 제한한다.
- 추론 요청은 bounded queue에 넣고 영상 처리 스레드를 차단하지 않는다.
- queue가 가득 차면 가장 오래된 불필요 요청을 버리고 드롭 카운터를 증가시킨다.
- 결과는 track ID별 TTL 캐시에 저장한다. 초기 TTL은 3초로 둔다.
- API timeout이나 오류가 발생하면 얼굴 필드만 생략하고 다른 이벤트는 그대로 처리한다.

### 5.2 얼굴 추론 서비스

- 입력 이미지의 크기와 형식을 검증한다.
- 얼굴 정렬 후 TensorRT 엔진으로 정규화된 임베딩을 생성한다.
- 갤러리 임베딩과 cosine similarity를 계산한다.
- 임계값 이상인 최상위 결과만 matched로 반환한다.
- 엔진과 갤러리는 프로세스 시작 시 한 번 로드한다.
- 배치 크기는 POC에서 1로 시작하고 실제 부하 측정 후 소규모 동적 배치를 검토한다.

### 5.3 모델과 갤러리

- 첫 후보는 현재 InsightFace `buffalo_l`에서 사용하는 검출/정렬/인식 조합이다.
- Jetson 지연시간이나 메모리 목표를 넘으면 `buffalo_s` 또는 경량 ArcFace 계열을 비교한다.
- 모델 ID와 전처리 버전을 갤러리 메타데이터에 기록한다.
- 모델 또는 전처리가 바뀌면 등록 원본 이미지로 전체 임베딩을 재생성한다. 서로 다른 모델의 임베딩은 혼합하지 않는다.

## 6. API 계약

내부 API의 최소 요청은 다음과 같다.

```json
{
  "camera_id": "sample_eval",
  "track_id": 4,
  "captured_at": "2026-07-15T13:00:00+09:00",
  "image_jpeg_base64": "..."
}
```

최소 응답은 다음과 같다.

```json
{
  "matched": true,
  "name": "registered-person",
  "similarity": 0.73,
  "model_version": "arcface-tensorrt-v1",
  "latency_ms": 42.1
}
```

POC에서는 구현 단순성을 위해 HTTP/JSON을 사용한다. localhost가 아니라 Compose 내부 네트워크에서만 접근시키고, 요청 크기 제한을 둔다. 성능 측정에서 직렬화 비용이 문제가 될 때만 Unix socket 또는 바이너리 전송을 검토한다.

## 7. 장애 처리

- 연결 timeout은 100ms, 전체 응답 timeout은 초기 800ms로 둔다.
- 연속 실패가 임계값을 넘으면 짧은 circuit breaker를 열어 불필요한 요청을 막는다.
- 얼굴 서비스가 unavailable이면 `face_recognition_status=unavailable`만 기록한다.
- DeepStream 프로세스는 얼굴 서비스 실패 때문에 종료하거나 재시작하지 않는다.
- 얼굴 서비스는 Docker health check와 restart policy를 사용한다.
- TensorRT 엔진 로드 실패는 서비스 health를 unhealthy로 표시하고 명확한 오류를 남긴다.
- OpenCV 폴백은 명시적 운영 설정으로만 활성화하며 TensorRT 결과와 같은 정확도로 간주하지 않는다.

## 8. 보안과 개인정보

- 얼굴 API 포트는 호스트에 publish하지 않는다.
- 얼굴 crop은 기본적으로 디스크에 저장하지 않는다.
- 등록 원본과 갤러리 파일은 기존 보호된 데이터 볼륨을 사용한다.
- 로그에는 base64 이미지와 전체 임베딩을 남기지 않는다.
- 내부 요청은 컨테이너 네트워크 제한을 기본 경계로 사용하고, 향후 다른 프로세스가 같은 네트워크에 참여하면 서비스 토큰을 추가한다.

## 9. 관측성

최소한 다음 값을 health 또는 메트릭에서 확인할 수 있어야 한다.

- 요청 수, 성공 수, 실패 수, timeout 수
- queue depth와 queue drop 수
- 평균 및 p95 추론 지연시간
- matched/unmatched 수
- 로드된 모델 버전과 갤러리 엔트리 수
- DeepStream의 얼굴 서비스 연결 상태와 circuit breaker 상태

로그는 camera ID와 track ID를 포함하되 얼굴 crop이나 임베딩은 포함하지 않는다.

## 10. 검증 계획

### 단위 검증

- similarity 임계값 경계
- 잘못된 이미지와 과도한 요청 크기 거부
- track ID 요청 간격 제한 및 TTL 캐시
- timeout 시 다른 이벤트가 유지되는지 확인
- 모델 버전 불일치 갤러리 거부

### 통합 검증

- TensorRT 엔진 로드 및 알려진 얼굴/미등록 얼굴 샘플 추론
- 얼굴 서비스 강제 종료 중 DeepStream 프레임과 MQTT 이벤트 유지
- 얼굴 서비스 재기동 후 자동 복구
- 카메라 1대, 얼굴 1~3명 조건에서 1초 이내 응답 확인

### 품질 검증

- 기존 등록 인원 데이터로 OpenCV 대비 InsightFace/TensorRT 정확도 비교
- FAR(타인을 등록자로 잘못 인식)과 FRR(등록자를 놓침)을 함께 측정
- 운영 threshold는 고정값을 추측하지 않고 검증 데이터 결과로 결정

### 운영 검증

- 기존 운영 점검 스크립트에 얼굴 서비스 health를 추가
- 최소 30분 DeepStream 안정성 감시에서 컨테이너 재시작, 프레임 드롭, FD 증가 확인
- Jetson GPU/CPU/메모리 사용량과 온도 확인

## 11. 단계적 적용

1. 현재 모델 artifact와 전처리 규격을 식별하고 TensorRT 변환 가능성을 검증한다.
2. 독립적인 TensorRT 얼굴 추론 POC를 만들고 정확도와 지연시간을 측정한다.
3. 로컬 API와 갤러리 로딩을 추가한다.
4. DeepStream에 bounded 비동기 클라이언트와 track 캐시를 연결한다.
5. 장애 격리 및 운영 점검을 검증한다.
6. 검증을 통과한 뒤 기본 백엔드를 OpenCV에서 TensorRT 서비스로 전환한다.

각 단계는 기존 OpenCV 운영 경로를 유지하며 진행한다. TensorRT 경로가 검증되기 전에는 현재 정상 운영 중인 DeepStream 설정을 바꾸지 않는다.

## 12. 영향 범위

- 새 얼굴 추론 서비스와 Docker 이미지가 추가된다.
- `docker-compose.jetson.yml`에 내부 서비스, health check, 데이터 볼륨 설정이 추가된다.
- DeepStream 얼굴 컨텍스트 처리는 직접 InsightFace 호출 대신 비동기 로컬 클라이언트를 사용하게 된다.
- 얼굴 갤러리에 모델 버전 메타데이터가 추가된다.
- 외부 공개 API와 MQTT 이벤트의 기존 필드는 유지하고, 얼굴 서비스 상태와 모델 버전 필드만 선택적으로 추가한다.

