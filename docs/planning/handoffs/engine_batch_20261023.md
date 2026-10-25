# 다중 카메라 엔진 배치 점검 인계 — 2026-10-23

**10-26 통합 마감 보완:** 최신 GPU 결과는 10-23 attempt3의 1 INVALID/8 NOT_RUN(카메라별1 INVALID/11 NOT_RUN), 유효결과0·실제GPU추론 미확인이다. 16:13:55.580339 정리/16:15:59.591102 반환 완료. 아래13:29·14:25 및 링크 진단은 당시 이력이며 현재 실행 상태로 외삽하지 않는다. 사전점검/변환 도구는 코드 커밋 `59624ed`에 포함됐지만 정확도/공유4입력 미평가·운영 엔진 미변경은 유지한다.

상태: **사전점검 원본77 PASS·max4 raw batch1/2/4 PASS 유지.13:29와14:25 후속 사전점검 모두 보호 중단·GPU0/9, 정확도/영상4입력 미평가.14:27:02 반환 확인·새 배정 없음·운영 미반영.**

## 링크 단절 원인·자동 복구 관측 — 최신 인계

**RTSP 입력은 운영 자체 복구가 관측됐으며 현재 실패 지속으로 표시하지 않는다.** 사용자 확인으로 Jetson A(Wi-Fi 식별IP192.168.2.141)·CCTV1대·미니PC를 PoE 스위치에 연결했다. A eno1은14:34:25~14:37:44에199초 단절됐고 EOS14:34:51·SDP timeout14:35:32/14:36:52가 모두 단절 구간 안이다. 이 입력 실패의 직접 네트워크 원인은 링크 단절로 설명된다.14:38부터 처리 재개,14:53~14:58 저장5분 평균29.306667FPS/14.650000Hz·마지막30초27.2FPS/13.566667Hz를 재계산했다. 저장 평가 차분이며 중간25~26FPS 구간이 있어 새 사전점검/장기 안정 PASS가 아니다.14:22~14:28 저하는 단절보다 앞서 회복했으므로 별도 원인 미확정이다. 수동 재시작0·운영 컨테이너 동일, A/B 조회 종료·새 GPU/부하 배정 없음.

아래14:42 절은 복구 확인 전 관측 이력이다. [공통 근거](../evidence/engine_batch_gpu_20261023.json)의`link_interruption_and_recovery_1500` 참조.

## 반환 후 RTSP 입력 실패 진단 — 14:42

후속14:36:39~49의10.03초에는 frames127556→127556으로0FPS·인덱서0·CPU약17%가 관측됐다.14:34:51 EOS 뒤 운영 자체 복구가RTSP SDP 응답 시간초과로 실패했다. 현재 입력을 열지 못하는 상태와14:25 저하의 동일 원인 여부는 미확정이다. Primary 카운터63778→0 감소는RESET으로 구분하며 음수 처리율로 사용하지 않는다. 카메라 링크/TCP 연결 존재만으로RTSP 정상 판정하지 않는다. 운영 재시작/설정 변경·새 카메라 접속 없이 원인 확인을 이어가며 새 시험은 보류한다.

통합 측은 원시10.030267초의 프레임/추론 차분0과SDP/timeout 로그 존재를 대조했다. CPU/네트워크 세부는 담당 보존 자료 기준이며 새 장비 조회를 하지 않았다. [공통 근거](../evidence/engine_batch_gpu_20261023.json)의`post_return_rtsp_diagnosis_1442` 참조.

## 개발도구 조치 완료와 attempt2 조기 반환 — 14:33 통합

**P6.2 개발도구 중복 인덱싱 조치 완료·엔진 재사전점검 차단**: 자동감시/자동시작OFF·직접 갱신 도구 비활성화·단일 잠금 수동 갱신을 적용했다. 설정/해시 대조 및14:15~16의45.20초30FPS/15Hz 회복은 확인했으나, 이후worker0 상태에서4.77FPS/2.39Hz로 다시 낮아졌다(원인 미확정). 엔진14:25 사전점검은4.19FPS/1.99Hz로 GPU0/9·변환0 보호 종료,14:27:02 반환·새 배정 없음. 정확도/영상4입력은 미평가다. ROI guard 보완31 PASS/Ruff PASS는 담당의 모의시험 결과이며 기존 실패 수치는 미기록 유지. camera-offload는 A/B 완성본을 보존한 카메라 서비스 선택3 hunk만 조율해 소스 개발 중이다. 운영 변경0·**P6.2 PLANNED/I56 OPEN/DONE9/37 유지**.

14:25:33.425312~14:25:38.439259의5.013944396초에 처리116011→116032(+21), Primary58006→58016(+10)으로4.188319FPS/1.994438Hz를 재계산했다. 가용38.753738GiB·최대49.875°C·age0.096685초, 오류/드롭/재시작 증가0. 자체 정리14:25:38.504054, 반환14:27:02.015114를 구분한다.9회·카메라별12결과 NOT_RUN, 운영 컨테이너 ID/이미지/시작/재시작 및177고정파일 보존.13:48의30초29.98FPS/14.99Hz와14:15~16의45초30/15 회복은 과거 관측이다.

A/B의reload params={}는 디스크 MCP 설정을 읽어loaded threads refresh를 예약했다. 설정 수정은codebase-memory 항목에 한정했지만 API 호출은 서버명을 지정하지 않았다. 다른 MCP별 재연결/일시 지연은 미계측이며 영향0으로 판정하지 않는다. 명시적 interrupt/kill/stop/앱·운영 재시작 요청0과 구분한다. 자동 인덱싱은OFF이며 검색 결과는 실제 파일과 대조한다. 수동 갱신은 통합 조율 후 단일flock helper만 사용한다.

[공통 근거](../evidence/engine_batch_gpu_20261023.json)의`indexing_fix_1418`·`quality_four_input_attempt2_1425`·`operating_drop_1425_after_indexing_fix`에 원시 사본과 재계산 결과를 기록했다. 아래13:45/13:29 절은 당시 인계 이력이다.

## 조치 전 중복 인덱싱 재발 — 14:05 지연 인계

**14:05 관측의 지연 인계 추가(조치 전 이력):** 30.323360초 전체28.426929FPS/14.246442Hz였지만 마지막5.188020초는 처리84630→84747·Primary42316→42376, **22.551955FPS/11.565105Hz로 보호 기준 미달**이었다.7표본 모두 인덱서13개·CPU99.760~99.904%, 오류/드롭/재시작 증가0·운영 식별 보존·GPU/변환0. 통합 측이 전체/실패창 차분과CPU6구간을 재계산했다. 전체 평균만으로 회복 PASS로 판정하지 않는다. 이는14:13:56 reload 이전 이력이며, 이후 조치 완료·14:27 엔진 반환·14:36 RTSP 입력 실패와 구분한다. 현재 새 배정 없음·자동 재시도0을 유지하고 단독 원인/채팅 책임/재부팅 원인은 확정하지 않는다.

[공통 원시 근거](../evidence/engine_batch_gpu_20261023.json)의`historical_indexing_recurrence_1405` 참조.

## 운영 FPS 진단·개발도구 담당 조율 — 13:45 이후

A/B 담당의 읽기 전용 진단 원시 자료를 통합 검토했다.13:40~13:41 운영16.57~17.60FPS와 CPU99.84~99.99%, codebase-memory-mcp 인덱싱7개 합계3.94~5.11코어(온라인8)가 함께 관측됐다. 변경 없이13:44:51~13:45:22 인덱싱0·CPU46.97~54.27%·30.29초 평균30.01FPS로 회복했다. **CPU 경합이 주된 원인으로 판단되지만 단독 원인·기여율·특정 채팅 책임은 미확정**이다. 통합 담당이 FPS/CPU를 재계산했고 오류/재시작/드롭 증가0을 대조했다.

가용33.82~37.04GiB·CPU약55°C·swap/OOM0으로 메모리 부족/과열 근거는 없다. 네트워크 오류/드롭 증가0이나 새 RTSP/패킷 확인은 하지 않아 네트워크 원인을 완전히 배제하지 않는다. 호스트13:03:28 부팅·운영AI13:04:48 시작으로12시 시험의 동일 실행과 구분하며 재부팅 원인은 미확인이다. **이30초 회복에는 PrimaryHz가 없고 장기 안정·추가 부하 여유·새 GPU 사전점검 합격을 뜻하지 않는다.**

사용자의 후속 요청으로 **A/B 담당이 공유 개발도구 실행기/자동감시/잠금·동시성과 중복 인덱싱 방지 수정**, 엔진 담당이 기존 자료 재사용 및 현재 회복의 저부하 읽기 전용 확인을 맡는다. 실제 공유 수정 경로·구체 패치·기존 검색/채팅을 유지하는 적용/복구 방법은 A/B 인계 대기다. ROI는 전용 계측 코드/모의시험, 통합은 공통 기록을 담당한다. 운영/타 작업·인덱싱 중지·재시작·설정 변경을 통합 측에서 실행하지 않았다.

기존13:29 배정 종료·GPU0/9회 NOT_RUN·정확도 미평가를 유지한다. **개발도구 조치/검증 영향 정리 → 현재 운영 회복 근거 → 새 명시 배정 → 배정 구간30초 사전점검 PASS →9회 비교** 순서다. 자동 GPU 재시도는 없다. [공통 근거](../evidence/engine_batch_gpu_20261023.json)의 `operating_fps_diagnosis_1345`와 `development_tool_coordination`에 원본 사본/해시와 담당·전달 접수를 기록했다.

## 후속 정확도·영상4입력 사전점검 중단 — 13:29

13:29~13:44 KST 배정에서 실행은13:29:29.251~13:29:34.705였다. 첫5.02289776초의 운영 프레임27480→27606(+126)은 **25.085121FPS**, Primary13741→13804(+63)은 **12.542561Hz**로27FPS/13.5Hz 기준 모두 미달했다.30초 사전점검 PASS가 아니며 GPU·변환·9회 비교시험은 모두0이다. 원시 카운터/monotonic/KST·중단 수치를 예외 전에 저장했고 통합 담당이 재계산했다.

- 가용36.701GiB·최고55.062°C·프레임 age0.0752초, 오류/누락/재시작 증가0. 처리율 저하 원인과 프로세스별 기여는 미확정이다.
- 기존batch1 네클립→후보batch1 네클립→후보 공유batch4 한회 계획의9회·카메라별12결과는 모두 **NOT_RUN**이다. 정확도·TP/FN/TN/FP·성능을 성공/실패 수치로 집계하지 않는다. 실물4카메라는 미연결이다.
- 자체 시험 컨테이너0·실행기 종료1 보고, 이번 실행 전후 운영 컨테이너 ID/이미지/시작시각/재시작 및 고정파일 SHA 동일. 이전 시험 이후 연속 동일 상태였다는 의미는 아니다. 운영/타 프로세스·인덱싱 변경과 자동 재시도는 없다.
- 담당의 A GPU·남은 구간 반환을 수신해 배정을 종료했다. 정확한 반환 시각은 미기재이며 실행 종료 시각과 구분한다. **운영 회복 확인 및 새 명시 배정 전 추가 부하 보류**, 기존 max4 후보와 미커밋 작업은 전용 worktree에 보존한다. 동일 구간 재확인: 종료·자체 자원 정리13:29:34.705085, 반환 통보 접수 `01a16707-586c-7cd1-8c94-81232d248360`는 담당 보고다. 새 회차·재시도0.
- CPU12 PASS·Ruff PASS·격리 Compose9개는 준비 검증이다. 원본 `data/runtime/engine_quality_four_input_20261023/{run,operating_samples,plan,preparation_checks,comparison}.json`과 담당 인계를 대조했다. [공통 근거](../evidence/engine_batch_gpu_20261023.json)의 `quality_four_input_followup_1329`에 원시 사본·해시를 기록했다. 제품 소스 추가 통합은 없다.

## max4 생성·raw 배치 GPU 결과 통합 — 2026-10-23

현재 운영 nano 엔진의 profile 0은 **1/1/1**로 batch2·4를 지원하지 않는다. 새 후보의 profile은 **1/4/4**이며 실제 batch1·2·4 추론과 후보 내부 출력 일치 검증을 통과했다. 반환 시각은 **11:51:15.128141 KST**다. 통합 담당은 결과 JSON·최종 manifest를 대조하고 후보 파일의 SHA/크기 및 원본 engine/ONNX/config 해시를 재확인했다. 제품 코드 추가 수정이나 기존 패치 재적용은 없다. **P6.2 PLANNED·I56 OPEN·DONE9/37 유지**, 정답 기반 정확도·공유2/4입력 앱·실물4카메라 성능은 잔여다.

## max4 후보와 실검증

- 후보: `data/runtime/engine_batch_20261023/yolov8n-pose-b1-4.engine`
- 크기: **12,850,340 bytes**
- SHA256: `a9da4a16e1f9c15c8c905b2aefc05b611aae79fb9cb0ad52d2d01bd0bd975e38`
- ONNX SHA256: `67ae9087ca8ed813f934660f5323842a96dc2e884e37547862061217b398ee9d`
- 대상: Jetson A / TensorRT **10.3.0**, FP16, 입력 `input=N×3×640×640`, profile 0 **min1/opt4/max4**.
- 기존 변환 도구의 workspace 1GiB·builderOptimizationLevel 0을 사용했다. 성능 최적화 완료를 뜻하지 않는다.
- 실제 생성 **52.14초**, 최종 검증 컨테이너 실행 **13.28초**. 최종 비교 스크립트 내부 시간은 8.21초이며 FPS 측정값으로 해석하지 않는다.

| 검사 | 기존 운영 엔진 | max4 후보 |
| --- | --- | --- |
| batch1 profile | PASS | PASS |
| batch2 profile | FAIL: 최대1 | PASS |
| batch4 profile | FAIL: 최대1 | PASS |
| batch5 범위 초과 | 미실행 | 정상 거부 |
| 실제 batch1·2·4 추론 | batch1 참조만 실행 | 모두 PASS |
| 후보 단건 대비 batch2·4 출력 | 해당 없음 | 최대 절대차 0, 정확히 일치 |

기존 영상 `00610_H_D_SY_C5`의 1·3·5·8초에서 서로 다른 입력 4개를 사용했다. 원본 영상 SHA를 기존 ROI 시험 manifest와 대조했고, 전처리 후 입력 SHA 4개도 모두 달랐다. 기존 `letterbox_pose`, `load_runtime`, `infer_pose`, `pose_tensor_difference`를 재사용했다. 각 배치의 출력 `[batch,56,8400]`과 유한값을 검증했다. 네트워크가 차단된 임시 컨테이너에서 실행했고 원본 엔진·영상·소스·가상환경은 읽기 전용으로 연결했다.

### 수치 비교의 한계

기존 엔진과 후보의 batch1 출력은 `atol=rtol=0.001` 기준으로 **동일하지 않다**. 원시 출력 최대 절대차 5.75, 평균 절대차 0.04379, 사람 점수 최대 절대차 0.00868, 전경 제안 최대 절대차 0.25였다. 여러 종류의 값이 섞인 원시 텐서 차이를 그대로 정확도 저하나 허용 오차로 단정하지 않는다.

후보 내부의 배치 일관성은 확인했지만, 낙상·헬멧의 정답 기반 정확도는 **NOT_EVALUATED**다. 운영 엔진 로드 시 TensorRT의 장치 모델 관련 경고도 관측했으며 기존 엔진 제작 장비는 이번에 확인하지 않았다. 후보 생성·배치 PASS를 운영 교체 승인으로 사용하지 않는다.

## GPU 순서·운영 보존·정리

- ROI 담당은 시험 프로세스 0개와 GPU 보류를 확인했다. 통합 담당이 A/B 시험 다음 최대15분을 예약했다.
- A/B 담당은 **11:38:24 KST**에 시험 AI와 CPU 부하 종료·GPU 반환을 확인했다. 예약 구간은 **11:38:24~11:53:24 KST**다.
- 실제 빌드 시작은 11:39:40, 최종 추론 검증 종료는 **11:48:18 KST**다. 검증 후 전용 시험 컨테이너와 trtexec/검증 프로세스 0개를 확인하고 통합·A/B·ROI 담당에게 반환을 전달했다.
- 진행 중인 A/B 채팅에는 공식 로컬 app-server WebSocket `turn/steer` 수신 처리를 확인했다. 통합·ROI 채팅에는 CLI queue로 결과를 전달했다. 대기열 등록과 담당자의 최종 기록 반영은 구분한다.
- 운영 engine·ONNX·Primary config의 SHA256은 시작 전과 검증 후 동일했다. 운영 컨테이너 ID와 restart_count=0도 유지됐다.
- 빌드·검증 구간의 운영 dropped/source-error/pipeline-restart 지표 증가0, 프레임 처리 계속 증가를 확인했다.
- 운영 엔진 SHA256은 `cc6cd0ee490e4f2ae46acf853989cf8c8806aadab47853e39c1e97cdf6732a74` 그대로다.

## 재시도와 검증 근거

모든 근거는 전용 worktree `/home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project`의 `data/runtime/engine_batch_20261023/`에 보존했다. 공통 [GPU 검증 근거](../evidence/engine_batch_gpu_20261023.json)에 결과 범위·SHA·재시도·운영 보존을 요약했다.

- `batch4_candidate.json`: 최종 상태·profile·배정·검증 요약.
- `gpu_validation_baseline.json`: 운영 파일 해시와 컨테이너 기준 상태.
- `batch4_build_result.json`, `batch4_build_observation.json`, `batch4_build.log`: 생성 명령·엔진 해시·운영 지표.
- `batch4_profile_checks.json`: 기존1/후보1·2·4 및 후보5 거부 결과.
- `batch4_validation_input.json`, `batch4_distinct_tensor_comparison.json`: 원본 영상·네 입력 SHA·최종 배치 비교 PASS.
- `batch4_validation_attempt3_run.json`, `batch4_validation_attempt3.log`: 최종 실행과 컨테이너 정리·운영 원본 불변 근거.

첫 검증은 읽기 전용 소스 폴더 아래 가상환경 마운트 지점을 생성하지 못해 컨테이너 시작 전에 종료125였다. 가상환경 연결 위치만 `/verification-venv`로 바꾸었다. 두 번째는 실제 배치 추론과 출력 일치까지 통과했으나 반복 재생 영상의 입력이 두 종류뿐이어서 네 입력 다양성 assert가 실패했다. 정답 영상 한 개의 서로 다른 네 프레임으로 최종 재검증을 통과했다. 최초 실패·중간 출력 기록도 삭제하지 않았다.

## 사전점검 코드 통합 이력 — GPU 작업 전

> 아래 사전점검·후보 준비 기록은 GPU 작업 전의 이력이다. 현재 결과는 위 GPU 실검증 절과 공통 근거 JSON을 기준으로 확인한다.

- 통합 경로는 `/media/sawwave/Learning11/CCTV-project`, 기준 HEAD는 `fefa4ab04ee2ca26bbc0ee433381e4b7648265e7`다. 작업자가 복사 전 기록한 소스·시험·사용법 5파일의 해시를 확인한 뒤 이번 작업 4파일 패치만 적용했다. AGENTS.md는 앞선 병렬 규칙 변경을 그대로 유지했다.
- 실제 호출부·설정 전달·profile 0 범위·실패 시 엔진 해제·CLI 오류 처리를 검토했다. 검토 범위에서 통합 차단 결함은 발견하지 않았다. 설치된 `NvInferRuntime.h`의 profile 조회 계약과 대조했으며 실제 엔진의 배치 지원을 확인한 결과는 아니다.
- 테스트 검토에서 추론 컨텍스트 미생성 검사를 `runtime` 대신 실제 `engine` 객체로 바로잡고, 플러그인 생략 시험의 SDK 실행 경계를 모의 처리했다. 제품 구현은 작업자 패치와 동일하다.
- 원본 통합 후 **pytest77 PASS(환경 검사50 + 변환27), 변경 Python2파일 Ruff PASS**. CPU1코어·nice19·스레드1로 직렬 검증했으며 테스트 실행126.49초다. GPU/카메라/Docker 접근·운영 설정 변경·엔진 생성/교체는 없다. 오케스트레이션 시험은 별도 worktree에서 계속한다.
- 통합 직전 파일·패치·Git index 백업과 실행 시간/로그는 `data/runtime/engine_batch_integration_20261023_os2za4gp/`에 보존했다. index SHA256 `805b110d7fe793abd2732c104080488b798cc73eda2404066ffd4f10a411fd6b`를 유지했다. 기존 미커밋 변경을 커밋·초기화·정리하지 않았다.
- 공통 WBS/PROGRESS/STATUS·진행판 JSON/HTML과 배정표에 결과를 반영한다. **P6.2 PLANNED·I56 OPEN·DONE9/37 유지**. max4 후보/설정은 작업자 worktree에 보존하며 장비 순서 배정 전 GPU 빌드·추론을 진행하지 않는다.

## max4 후보 준비 이력 — GPU 작업 전

사용자가 엔진 배치를 max4로 진행하도록 지시했다. 이후 생성·검증 대상은 **min1/opt4/max4, input N×3×640×640, FP16**이며, 아래 이전 max2 dry-run 기록은 중간 준비 이력이다. 실제 실행 배치는 등록 카메라 수 1→2→4를 따르고 보조 추론 배치는 별도로 유지한다.

- 기존 nano ONNX의 SHA·동적 입력·min/opt/max 검증 및 변환 **dry-run PASS**. 별도 출력은 `data/runtime/engine_batch_20261023/yolov8n-pose-b1-4.engine`이며 실제 엔진은 아직 생성하지 않았다.
- `data/runtime/engine_batch_20261023/batch4_candidate.json`에 검증한 생성 명령·범위·상태를 저장했다. `config_infer_primary_batch4.txt`는 같은 디렉터리의 후보 전용 nvinfer 설정이다. 기존 설정에서 엔진/라벨 경로와 batch만 변경했으며 운영 config는 보존했다.
- 기존 `scripts/ops/compare_tensorrt_pose_candidate.py`가 이미 batch1 대비 batch2/4의 원시 Pose 출력을 비교한다. 새 검증기를 추가하지 않고 이를 재사용한다. 전체 경보 품질·성능 검증과는 구분한다.
- 후보 설정의 오프라인 검사는 예상한 미생성 엔진 1건만 실패했고 라벨·tracker는 통과했다. 원본 설정 해시 동일·GPU 접근0을 확인했다. 실제 엔진 파일이 없으므로 준비 상태를 엔진 생성 완료로 해석하지 않는다.
- 후속 확인에서 오케스트레이션 채팅이 A/B 자동 인계 실장비 시험 준비를 진행 중이다. 같은 장비 부하를 겹치지 않도록 GPU 빌드/추론은 시작하지 않았다. 통합 담당의 장비 순서 배정 후 **현재 엔진 profile 확인 → 별도 max4 후보 빌드 → batch1/2/4 사전점검·출력 비교 → 공유2/4입력 앱 검증**으로 이어간다. 운영 중단·엔진 교체는 하지 않는다.
- 통합 담당 기록 요청: P6.2/I56의 후속 후보를 **max4**로 변경하고 GPU 빌드 대기·기존77 PASS·새max4 dry-run PASS·운영 미반영을 구분한다. 기존 완료 수는 유지한다.

## 담당·격리

- 채팅: 다중 카메라 엔진 배치 문제 확인 (`01a1664c-5d6a-7ea2-a22e-04e379c47d30`).
- WBS: P6.2 사전검증 / 기존 I56 연관. 완료 수와 기존 이슈 상태를 변경하지 않는다.
- 경로: `/home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project`.
- 브랜치: `codex/multicamera-engine-batch`; 기준 HEAD `fefa4ab04ee2ca26bbc0ee433381e4b7648265e7`; 결과는 미커밋.
- 통합 창구: **Codex 병렬 작업 충돌 검토**. 배정표·공통 WBS/PROGRESS/STATUS/진행판은 수정하지 않았다. 이 세션에는 채팅 전송 도구가 제공되지 않아 자동 전달 완료로 표시하지 않는다.
- **다중 카메라 ROI·트래픽 수정 계획**은 `two-camera-baseline`, **오케스트레이션으로 프로젝트 진행하기**는 `edge-auto-failback` worktree에서 진행 중임을 앱 조회로 확인했다. 본 작업의 담당은 health 검사·관련 시험·사용법이며 GPU/카메라/컨테이너 생성·운영 설정 변경은 0이다.
- 현재 소스에서 필요한 파일만 복사하고 복사 전후/종료 시 원본 해시를 대조했다. 운영 .env·DB·모델·가상환경은 복사하지 않았다. 자세한 복사 목록은 `data/runtime/engine_batch_20261023/baseline.json`이다.

## GPU 실행 전 문제 확인 이력

1. `src/core/_deepstream_element_config.py:65`는 Primary batch를 카메라 수로 덮어쓴다. 설정 파일의 참고값 `batch-size=8`이 실제 실행 배치라는 뜻은 아니다. 이는 streammux/Primary 배치를 입력 수에 맞추는 [NVIDIA 기준](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html)과 맞으며 임의로 1에 고정하지 않았다.
2. 기존 medium 엔진의 batch1/요청2 불일치와 기동 실패는 `docs/planning/evidence/two_channel_preflight_20261007.json`, `VIDEO_INFERENCE_CONTRACT.md#two-channel-v136`에 기록되어 있다. 당시 기동 실패를 FPS 저하로 바꾸어 해석하지 않는다.
3. 현재 사전점검의 `_check_engine_load`는 역직렬화 성공만 확인했다. GPU를 모의한 회귀에서 고정 profile1 엔진/요청2도 통과하는 누락을 재현했다. 실제 SDK 소스 `nvdsinfer_context_impl.cpp:2020`은 엔진 최대 배치가 요청보다 작으면 `NVDSINFER_CONFIG_FAILED`를 반환하며, `nvdsinfer_backend.cpp:28`의 기본 profile은 0이다.
4. 현재 nano 엔진 `models/fall/yolov8n-pose.engine`은 8,956,700 bytes, SHA256 `cc6cd0ee490e4f2ae46acf853989cf8c8806aadab47853e39c1e97cdf6732a74`로 10-22 로드 기록과 같다. 그 기록은 로드 성공만 포함하므로 **현재 nano 엔진의 batch2 지원 여부는 확정하지 않았다**. 이번에 실제 엔진을 새로 GPU 로드하지 않았다.
5. nano ONNX `models/fall/yolov8n-pose.onnx`의 입력은 `input=[batch_size,3,height,width]`, SHA256 `67ae9087ca8ed813f934660f5323842a96dc2e884e37547862061217b398ee9d`다. 기존 변환 도구로 min1/opt2/max2·640 입력 dry-run이 통과했다. 원본은 동적 배치를 지원하지만 새 엔진의 빌드 성공·성능·정확도는 아직 검증하지 않았다.
6. 전체 앱의 단일 RTSP 출력과 batch2 미리보기 제한은 별도 잔여다. 이번 패치로 다중 카메라 전체 경로나 성능이 해결됐다고 판정하지 않는다.

## 변경 파일과 동작

- `scripts/health/check_deepstream_env.py`: `--load-engines`에서 설정 배치와 실제 profile 0의 min/opt/max 대조. 단일 4차원 영상 입력만 검사하고 지원 불명·잘못된 값·범위 불일치는 종료1. 실패 시에도 엔진 해제. `--batch-size`로 앱의 실제 override를 명시하며 양수/단일 `--config`/`--load-engines` 조합을 강제한다. 생략하면 설정값(누락 시 1)을 검사한다.
- `tests/test_check_deepstream_env.py`: 고정1/요청2, 동적 범위 경계, 잘못된 설정과 profile, CLI 전달/오용을 검사한다.
- `docs/guides/JETSON_EDGEX_FIELD_CHECKLIST.md`: 단일 입력 예시를 `--batch-size 1`로 명시하고 카메라 수와 SGIE ROI 배치의 차이, 검증 한계를 기록했다.

앱 추론/API/DB/운영 설정·엔진 교체는 없다. 기존 `--load-engines` 사용자는 설정의 참고 배치와 앱 override를 구분해야 한다. `--files-only`/`--native-only`는 기존처럼 GPU를 로드하지 않는다. 이 검사는 앱 시작 경로에 자동 연결된 기능이 아니라 명시적으로 실행하는 사전점검이다.

## 검증

- 수정 전 회귀 10건 실패: 배치 불일치/잘못된 값의 오통과를 확인했다.
- 수정 후 전용 worktree: `rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_check_deepstream_env.py tests/test_convert_onnx_to_engine.py -q -p no:cacheprovider --tb=short` → **77 PASS** (검사50 + 기존 변환27).
- 최초 worktree 실행은 sandbox 읽기 전용 제약으로 tmp fixture 64건 setup 오류/13 PASS였다. 제품 실패로 숨기지 않았으며 권한 허용 후 위 77건을 정상 실행했다.
- 변경 Python 2파일 Ruff PASS. 실제 nano ONNX의 SHA/입력 형상/범위 검증 및 변환 dry-run PASS, 엔진 파일 생성0.
- 현재 원본 checkout에 `git apply --check` PASS. 복사한 원본6파일 해시 동일. 운영 프로세스 조작0, 이번 작업이 남긴 실행 중 시험 프로세스0.
- 그래프를 우선 조회했으나 일부 최신 함수 누락 및 다른 변경을 섞은 리뷰 결과가 있어 필요한 실제 소스·SDK·호출부와 최소 diff로 대조했다. 그래프를 재생성하지 않았다.

## 최초 통합·GPU 준비 명령 이력

오래된 HEAD와 현재 작업판의 차이가 있어 worktree 전체 diff는 적용하지 않았다. 복사한 현재 소스를 기준으로 수정 3파일과 이 인계 문서를 포함한 패치만 통합했다. 아래는 최초 적용 전 확인 명령의 기록이며 이미 통합한 원본에 다시 적용하지 않는다.

```bash
rtk git apply --check /home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project/data/runtime/engine_batch_20261023/engine_batch_fix.patch
```

장비 사용 시간 조율 후 실제 SDK/GPU Python 환경에서 현재 nano 엔진부터 검사한다(아래 명령은 아직 미실행).

```bash
rtk proxy python3 /home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project/scripts/health/check_deepstream_env.py --root /media/sawwave/Learning11/CCTV-project --config /media/sawwave/Learning11/CCTV-project/config/deepstream/config_infer_primary.txt --load-engines --batch-size 2 --timeout 15 --json
```

후속 GPU 검증은 최종 목표인 max4 후보를 사용한다. 다음은 GPU 없는 준비 명령이며, 실제 빌드 시에만 장비 조율 후 `--build`를 추가한다.

```bash
rtk proxy python3 /home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project/scripts/convert/convert_onnx_to_engine.py --onnx /media/sawwave/Learning11/CCTV-project/models/fall/yolov8n-pose.onnx --engine /home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project/data/runtime/engine_batch_20261023/yolov8n-pose-b1-4.engine --sha256 67ae9087ca8ed813f934660f5323842a96dc2e884e37547862061217b398ee9d --input-name input --min-shape 1x3x640x640 --opt-shape 4x3x640x640 --max-shape 4x3x640x640
```

최적화0/workspace1GiB의 준비 후보이며 성능 최적화 완료가 아니다. 기존 원본은 보존하고 새 후보의 batch1/2/4·입력 순서 변경 출력, 실제 공유2/4입력 기동/메타 식별/누락/지연·추론율을 검증해야 한다. medium 후보의 과거 성능 미달도 있으므로 배치 지원만으로 운영 교체하지 않는다.

현재 판정: **P6.2 사전점검 통합77 PASS·후보 GPU batch1/2/4 PASS·정확도/공유 앱/성능 미검증·운영 미반영**. I56 OPEN 및 기존 완료 수를 유지한다. 다음은 기존 엔진 대비 정답 기반 품질과 공유2/4입력 앱·실물 다중 카메라 검증이다.
