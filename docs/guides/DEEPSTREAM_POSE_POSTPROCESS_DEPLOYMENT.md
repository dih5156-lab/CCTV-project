# DeepStream Pose 후처리 배포 기준

## 현재 운영 경로

`primary nvinfer`의 YOLOv8-pose raw tensor를 Python pad-probe에서 읽고 다음 순서로 처리한다.

1. NumPy confidence/class 일괄 필터
2. letterbox 좌표 복원
3. keypoint 기반 낙상 판정
4. class별 NMS
5. OSD 및 이벤트 생성

운영 기본값은 `DS_YOLO_POSTPROCESS_MODE=vectorized`다. 장애 시 `legacy`로 변경하고
`cctv-ai-engine`만 재시작하면 기존 Python row 반복 경로로 롤백된다.

10초 주기 `DeepStream stats` 로그에서 다음 필드를 확인한다.

- `yolo_postprocess`: 활성 모드
- `avg_ms`, `max_ms`, `calls`: 후처리 지연과 호출 수
- `frame_dropped`, `failed`: 파이프라인 안정성

## C++ 전환 구조

표준 `NvDsInferParseCustomFunc`만으로 전환하지 않는다. 해당 인터페이스의 반환 계약은
`NvDsInferObjectDetectionInfo` 목록이므로 현재 낙상 판정에 필요한 17개 keypoint와
부가 결과를 완전하게 전달하지 못한다.

배포용 네이티브 경로는 primary `nvinfer` 직후에 C++ tensor postprocess element 또는
pad-probe를 배치하고 다음 메타를 생성해야 한다.

- `NvDsObjectMeta`: person bbox, confidence, class id
- 전용 `NvDsUserMeta`: 17x3 keypoint, pose tensor schema version
- copy/release callback: tee, queue, tracker를 통과할 때 메타 수명 보장

Python은 raw tensor를 다시 파싱하지 않고 위 메타를 이벤트/낙상 규칙 입력으로 변환한다.
이 구조여야 tracker와 PP-Human SGIE도 primary person object meta를 정상 소비할 수 있다.

## 전환 승인 조건

네이티브 경로는 아래 조건을 모두 통과하기 전 운영 기본값으로 바꾸지 않는다.

1. 고정 tensor fixture에서 Python과 bbox/class/confidence/keypoint가 허용 오차 내 일치
2. NMS 결과의 개수와 정렬 순서 일치
3. 낙상/비낙상 골든 영상 recall 및 false-positive가 기존 기준 이상
4. 1/2/4 카메라 soak test에서 frame drop과 queue drop이 허용 범위 이내
5. ASan/UBSan 테스트와 DeepStream meta copy/release 반복 테스트 통과
6. `DS_YOLO_POSTPROCESS_MODE=vectorized` Python 경로로 즉시 롤백 가능

## 성능 예산

- Python vectorized 후처리 평균: 5ms 이하
- C++ 후처리 목표 평균: 1ms 이하
- p99/max 지연은 별도 기록하며 장시간 증가 추세가 없어야 함
- 운영 GPU/CPU 평가는 모델 warm-up 이후 최소 5분 평균으로 비교

