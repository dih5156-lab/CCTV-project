# 외형 통합 모델 결정 및 보강 계획 (2026-08-28)

## 결론

최종 외형 모델 후보는 `appearance_mobilenet_v3_crop_weighted/protected_multitask`로 고정한다.
현재 PA100K TensorRT와 HSV/LAB 색상 경로는 후보 모델의 운영 승격 전까지 유지한다.

## 현재 검증 결과

| 속성 그룹 | Macro F1 | 판단 |
|---|---:|---|
| 성별 | 0.8731 | 운영 후보 수준 |
| 상의 형태 | 0.8704 | 운영 후보 수준 |
| 하의 형태 | 0.6628 | 치마 계열 보강 필요 |
| 상의 색상 | 0.6852 | 목표 0.70에 근접 |
| 하의 색상 | 0.5261 | 운영 승격 불가, 우선 보강 필요 |
| 소지품 | 0.5147 | 모자·가방 Recall 보강 필요 |

전체 학습 로그 기준 색상 F1은 0.6062, 비색상 F1은 0.7231이다.

## 주요 병목

### 하의 색상

| 클래스 | 검증 샘플 | F1 | 조치 |
|---|---:|---:|---|
| brown | 90 | 0.0336 | 최우선 추가 수집 |
| navy | 654 | 0.2160 | blue/black 혼동 샘플 보강 |
| white | 295 | 0.5594 | 조명 변화 샘플 보강 |
| gray | 706 | 0.6126 | black/white 혼동 검수 |
| red/green/yellow/purple/orange | 0 | 0 | 학습·검증 샘플 신규 확보 |

하의가 보이지 않는 사람 crop은 색상 라벨을 강제로 지정하지 않고 `unknown/exclude`로 유지한다.

### 상의 색상

- `blue`: Recall 0.369, F1 0.506으로 최우선 개선 대상이다.
- `navy`: F1 0.556으로 blue/black과의 혼동을 점검한다.
- black, white, yellow, red는 상대적으로 안정적이다.

### 비색상 속성

- 성별: female F1 0.868, male F1 0.878로 양호하다.
- 모자: Recall 0.087, F1 0.152로 운영 검색 조건에 사용하기 어렵다.
- backpack/bag: Recall이 각각 0.413/0.466이므로 추가 보강이 필요하다.
- long_skirt: Recall 0.117로 치마 데이터 보강이 필요하다.

## 운영 승격 기준

다음 조건을 모두 만족한 뒤 ONNX/TensorRT 변환과 Jetson 배포 평가를 진행한다.

1. 상의 색상 Macro F1 0.70 이상
2. 하의 색상 Macro F1 0.65 이상
3. 비색상 종합 F1 0.72 이상 유지
4. 성별 Macro F1 0.85 이상 유지
5. TensorRT FP16 단일 crop 추론 평균 20ms 이하
6. 실제 CCTV 대표 crop 검수에서 심각한 색상 혼동이 없을 것

## 다음 작업

1. 하의 brown/navy와 누락된 색상 데이터를 우선 확보한다.
2. 상의 blue/navy/black 혼동 샘플을 별도 검수 세트로 만든다.
3. 모자·가방·치마 샘플에 class weight 또는 balanced sampler를 적용한다.
4. 동일 검증 split으로 재학습해 현재 checkpoint와 비교한다.
5. 품질 기준 통과 후 ONNX 및 TensorRT FP16 엔진을 생성한다.
