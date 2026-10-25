# SS25 Anomaly Detection 저장소 적용 검토

확인일: 2026-10-13. 사용자 지정 [저장소](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project), 확인 커밋 `8367d372c445722fa2c247a8420cc2570cbfce6e`. 코드·설정·보고서를 읽기 전용으로 검토했으며 설치·빌드·실행·제품 코드 도입은 하지 않았다.

**결론: 다중 입력/배치 구성의 비교 자료로 활용한다. 낙상 모델이나 운영 시스템 전체의 대체 후보로 채택할 근거는 부족하다.** NVIDIA 참조 앱을 확장한 교육 프로젝트이며 우리 프로젝트도 이미 DeepStream·TensorRT·nvstreammux를 사용한다. 공통 구조를 새로 도입하기보다 현재 구성과 차이를 대조하는 편이 작다.

| 확인 항목 | 실제 근거 | 우리 프로젝트 판단 |
|---|---|---|
| 탐지 방식 | ResNet18 TrafficCamNet 객체 탐지 + `nvof` 광학 흐름 + bbox 내부 평균 이동 방향 8방향 표시 | 확인한 경로에 낙상 전용 학습 모델·지속 판정·정답 기반 낙상 성적은 없음. Pose 낙상 판정부 대체 근거 부족 |
| 입력 | C 앱은 `/dev/video*`의 v4l2src와 URI의 uridecodebin 분기. 실행 예시는 D455 1280×720/30FPS, USB·파일 혼합 | RTSP URI 분기는 있으나 실시간 CCTV 재접속/장시간 운용 성적은 별도. RealSense 장비 추가 불필요 |
| 배치 | 소스 수를 mux/추론 batch-size에 반영, 타임아웃33000µs | P5.1/P6.2에서 우리 mux·TensorRT 엔진 profile·실제 tensor 출력 수 대조에 활용. batch 숫자만 바꿔 성능 해결로 보지 않음 |
| 표시 분기 | tee 뒤 queue, 추론 표시·광학 흐름 표시를 분리 | 큐 포화 시 전체 지연 여부를 점검하는 비교 자료. queue 추가만으로 느린 카메라 격리가 보장되지 않음 |
| 동기화/오류 | `sync-inputs=TRUE`, 표시 sink `sync=1`; bus ERROR는 전체 main loop 종료 | 우리 운영에 그대로 복사하지 않는다. 서로 다른 FPS·단절·느린 입력에서 다른 카메라 처리 지속 여부를 측정 |
| 확장성 | 보고서 인쇄34쪽은 3개 영상 입력에서 큰 지연과 충돌을 기록 | 실물4카메라 또는 장시간 운영 합격의 증거로 사용할 수 없음 |
| 권리 표시 | 포함된 `deepstream_reference_apps/LICENSE`와 검토 C/C++ 파일은 Apache-2.0 명시 | 선택 도입 시 해당 파일의 저작권/라이선스 보존. 모델·엔진·다른 자료까지 같은 조건이라고 확대 해석하지 않음 |

코드 근거: [입력·배치·분기·오류 처리](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project/blob/8367d372c445722fa2c247a8420cc2570cbfce6e/Anomaly_Detection_Camera/deepstream_reference_apps/anomaly/apps/deepstream-anomaly-detection-test/deepstream_anomaly_detection_test.c), [모델 설정](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project/blob/8367d372c445722fa2c247a8420cc2570cbfce6e/Anomaly_Detection_Camera/deepstream_reference_apps/anomaly/apps/deepstream-anomaly-detection-test/dsanomaly_pgie_config.txt), [방향 계산](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project/blob/8367d372c445722fa2c247a8420cc2570cbfce6e/Anomaly_Detection_Camera/deepstream_reference_apps/anomaly/plugins/gst-dsdirection/dsdirection_lib/dsdirection_lib.cpp), [실행 명령](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project/blob/8367d372c445722fa2c247a8420cc2570cbfce6e/Commands.txt), [보고서](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project/blob/8367d372c445722fa2c247a8420cc2570cbfce6e/Report_Nvidia_Orin_Nano_ss25.pdf), [라이선스](https://github.com/Prajwal-Mangaluru/SS25-Anomaly_Detection_Project/blob/8367d372c445722fa2c247a8420cc2570cbfce6e/Anomaly_Detection_Camera/deepstream_reference_apps/LICENSE).

## 선택 도입 전 확인할 코드 제약

`DsDirectionProcess`에는 bbox→flow 배열 접근의 명시적 범위 제한과 유효 움직임 픽셀0개일 때 나눗셈 방어가 보이지 않는다. 해당 함수만의 정적 관찰이며 실제 충돌을 재현한 결과는 아니다. 또한 코드의 흐름값 환산은 `/4`인 반면 [DeepStream 7.1 nvof 문서](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvof.html)는 S10.5 값을 `/32`로 환산한다. 방향각과 별개로 크기 임계값의 의미가 달라질 수 있으므로 그대로 적용하지 않는다. 광학 흐름은 움직임 보조 정보이며 낙상 판단 근거가 추가로 필요하다.

## 현재 WBS에 연결할 범위

1. **P5.2:** 자세 보완31회 중 단일 영상30회 완료. 보완10영상 낙상4/5→5/5·기존 긴 지연6.190→3.587초, 추가10영상2/5→3/5·정상10개 무경보다. 추가 미탐2개·지연5.322초와 미사용80영상은 잔여다. 외부 방향 플러그인을 도입하지 않고 기존 공통 Pose 판정부를 보완했다.
2. **P5.1/P6.2 사전검증:** 다중카메라의 배치·큐 대조 자료로 활용하는 방향은 적절하다. 현재 `configure_streammux`/`configure_infer_elements`도 소스 수를 batch에 반영하고 `configure_output_queue`는 leaky=2·max-size-buffers=2를 사용하므로 중복 도입하지 않는다. 이번2입력 시험은 단일 비교용 엔진maxBatch1/앱 요청2 불일치로 기동 실패·처리0이다. 파일/Compose 사전점검에 engine profile 검증이 빠졌으며, 다음에는 AI 정지 전에 해시와 실제profile/요청batch를 대조하고 기존 batch1~2 후보를 명시한다. 해당 후보는 기존 엔진과 수치 동등성 미달 이력이 있어 정확도·속도도 별도로 확인한다. 이후 느린 입력/단절 시 다른 채널 처리율·큐 대기·PTS/세대 유지와 출력 분리를 검증한다. 실물4카메라 최종 조건은 유지한다.
3. **운영 준비:** 기존 경보 만료·영속 저장·장치 분리·복귀 설계를 유지한다. 이 예제의 화면 출력 성공을 우리 장치 경보 전달 성공으로 치환하지 않는다.

원저장소인 [NVIDIA DeepStream reference apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/anomaly)도 비교 자료로 확인했다. 현재 master 안내는 DeepStream9.0이므로 7.1 프로젝트에 최신 파일/설정을 일괄 복사하지 않는다. 이번 검토로 새로운 라이브러리나 광학 흐름 플러그인을 설치할 필요는 없다.

원문 사본·커밋 응답·PDF 텍스트는 `data/runtime/p52_expanded_20261013/reference_review/`에 보존했다. 모델 실행 성적은 [우리 GPU 비교 근거](../planning/evidence/fall_expanded_evaluation_20261013.json)와 구분한다. 외부 저장소 성능 수치는 우리 Jetson에서 재현하지 않았다.

후속 근거: [자세 보완·2입력 실패·운영 복구](../planning/evidence/fall_projection_20261013.json). 배치 설정을 맞춘다는 설계 방향과 실제2채널 운용 합격은 구분한다.
