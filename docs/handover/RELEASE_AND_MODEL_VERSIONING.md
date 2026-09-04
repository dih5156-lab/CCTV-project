# 릴리스·모델 버전 관리

## 릴리스 단위

코드만의 버전이 아니라 다음 조합을 하나의 배포 버전으로 기록한다.

```text
application commit
+ docker-compose / env schema
+ model artifact + checksum
+ TensorRT/CUDA/JetPack
+ dataset/evaluation report
+ deployment date and operator
```

## 릴리스 기록 템플릿

| 항목 | 값 |
|---|---|
| 릴리스 ID | `<예: 2026.09.03-r01>` |
| Git commit | `<입력>` |
| 모델 manifest | `<입력>` |
| 주요 모델 | `<입력>` |
| 평가 리포트 | `<경로>` |
| JetPack/CUDA/TensorRT | `<입력>` |
| Compose 파일 | `<입력>` |
| 변경 요약 | `<입력>` |
| rollback 대상 | `<입력>` |
| 승인자 | `<입력>` |

## 배포 순서

1. 변경 diff와 평가 리포트를 검토한다.
2. Compose config와 unit/integration test를 실행한다.
3. 이전 모델·환경변수·이미지를 보존한다.
4. 가능하면 shadow 또는 단일 카메라로 먼저 배포한다.
5. health, 이벤트, 장치, latency를 확인한다.
6. 현장 승인 후 전체 카메라로 확대한다.

모델 파일을 단순히 덮어쓰지 말고 manifest, checksum, 평가 결과를 함께 변경한다.

