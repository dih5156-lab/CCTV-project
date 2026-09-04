# Pose 모델 학습 기준선

## 현재 운영 기준

| 모델 | Precision | Recall | 평균 지연시간 |
|---|---:|---:|---:|
| YOLOv8n-Pose TensorRT | 0.8264 | 0.3937 | 44.5ms |
| YOLOv8m-Pose TensorRT | 0.8837 | 0.4488 | 43.2ms |

위 수치는 61개 평가 이미지의 사람 검출 단계 기준이며, 관절별 OKS/AP 지표와는 구분합니다.

## 공개 데이터 학습 목표

- COCO Keypoints로 17개 관절 형식과 기본 Pose 성능을 확보합니다.
- 공개 데이터 검증 Recall이 기존 기준보다 낮으면 운영 후보로 승격하지 않습니다.
- 이후 CCTV 가림·저해상도·어두운 장면을 추가해 fine-tuning합니다.
- TensorRT 변환은 학습과 검증을 통과한 모델에만 적용합니다.

## 운영 승격 기준

1. 사람 검출 Recall: 기존 후보 대비 개선
2. Pose 누락률: CCTV 검증 영상에서 측정
3. 낙상 temporal Recall: 기존 0.95 수준 이상 유지
4. Jetson 추론 지연시간: 운영 허용치 이내
5. TensorRT 변환 후 정확도 하락 여부 확인
