# 기존 성과·실행 기준선·이슈 현황

확인일: 2026-10-02 KST · 계획 v0.101 · [WBS](WBS.md) · [진행 기록](PROGRESS.md)

v0.101: **실제 AI 후처리·버퍼 소유권5조건 확인, 종료 재진입 관측.** CPU 영상3565개 모두 독립 소유·인코더/디코더 중지20요청 반환, SIGTERM 이후2.18~14.65초 종료. flush 제외 대조에서 동일 스레드 stop() 중첩1건 확인·기존 native 멈춤과 인과 미확정. 제품206파일 불변·운영 정상, I40 OPEN·P5.1 IN_PROGRESS. [근거](evidence/buffer_ownership_20261002.json) · [분석](TRACKER_VALIDATION.md#buffer-ownership-v101).

v0.100: **실녹화·디코더·미리보기 종료6조건 확인.** 약89~90초씩 처리·인코딩15950개, 종료39~59ms·인코더/디코더 중지24요청 모두 반환. 미리보기 CPU 매핑5347개 확인. I40 원 실패는 미재현이며 해결 판정하지 않는다. 제품206파일 불변·운영 정상. [근거](evidence/encoder_decode_20261002.json) · [분석](TRACKER_VALIDATION.md#encoder-decode-v100).

v0.99: **Ponytail 검토·인코더 최소 재현6조건 확인.** 공통 생성/종료 함수와 기존 요청 계측을 재사용, 합성 입력·RTSP·단절·NVMM·mux6조건 종료21~35ms/중지 요청 모두 반환. 실제 AI 통합의90초 실패 조건과 달라 I40 해결 근거 아님. 진단 성공 판정에 실제 게시·단절 순서 확인 추가. 제품206파일 불변·운영 정상, I40 OPEN·P5.1 IN_PROGRESS. [근거](evidence/encoder_minimal_20261002.json) · [검토/시험](TRACKER_VALIDATION.md#encoder-minimal-v099).

v0.98: **종료 대기를 인코더 입력 STREAMOFF로 특정·flush 제외안 미채택.** 격리10회 중 정상 종료8회·30초 초과2회(서로 다른 조건, 실패율 아님). 실패 스택/동일 시점 주소와 SDK 바이너리 대조로 NvVideoEncAbortBuffer→NvOsSemaphoreWait 확인. 요청 계측에서도 인코더 입력 STREAMOFF만 반환되지 않음. 제품 소스·SDK·운영 변경 없음, I40 OPEN·P5.1 IN_PROGRESS. [실행 근거](evidence/native_shutdown_20261002.json) · [분석](TRACKER_VALIDATION.md#native-shutdown-v098).

v0.97: **첫 프레임 지연 구간 확인·새 후보 통합 검증, 종료 실패 잔여.** GPU OSD 첫 처리10.87초→CPU OSD17.64ms, 등록→CPU10.97→0.11초. 새 고정 후보 단절 복구8.16초·CPU1450개 정확 대응/누락0·TCN18회(유효17/Pose 부족1), 구성 회귀85 PASS. 추가 관찰 종료 실패2회(일반30초/진단60초 초과), 전체 NULL→NVIDIA 버퍼 풀/드라이버 대기 native 확보·I40 OPEN. 운영 미배포·P5.1 IN_PROGRESS. [실행 근거](evidence/startup_delay_20261002.json) · [분석](MODEL_INFERENCE_METRICS.md#startup-delay-v097).

v0.96: **매칭 누락 원인별 계측 구현·격리 검증.** 총계 유지·카메라별 최근256개 사유 기록, 실제 누락24건=무대상23+만료1로 프레임 기록/Prometheus 일치. 관련245 PASS/2 SKIP·전체2737 PASS/52 SKIP·Ruff PASS. 첫 프레임10.34초 중 메타 완료→미리보기 분기 전10.29초로 범위를 좁혔으며 원인은 미확정. 영상1333프레임/DB3개 검증·운영 미반영, I32/P5.1 진행 중. [실행 근거](evidence/match_metrics_20261002.json) · [분석](MODEL_INFERENCE_METRICS.md#match-reason-metrics-v096).

v0.95: **매칭 누락 사유 재현·배포 순서 조정.** 격리2회 정상 종료·영상3122프레임 디코드/DB3개ok. 두 번째 누락12건은 빈 사람 메타11·첫 프레임 만료1(도착7.95초)로 분류. 과거13건의 원인 확정은 아니며 I32/I38 OPEN. RF 재생성 설정2개를 현재 운영값에 맞춰 명시·서비스 재시작/제품 소스 변경 없음. 구성 회귀63 PASS·분류10조건 PASS, 기존 배포 구성 검사3실패/외형 모델 경로 불일치는 I46 OPEN. 별도 Jetson 시험은 기능 완성 후로 연기. [실행 근거](evidence/match_reason_20261002.json) · [분석](MODEL_INFERENCE_METRICS.md#match-reasons-v095).

v0.94: **운영 환경 복구·설치 검증 보완.** 호스트 누락2패키지 설치(I44 FIXED), 호스트/컨테이너 SDK·필수 플러그인 각6항목 PASS. X11 권한 복구와 동일 이미지 AI 재시작 후 프레임 진행 재개. 새 health는 첫 프레임 없음/30초 정지에503이며 운영 반영됨. X11 재시도·SDK 사전/사후 검사 서비스도 설치·reload 확인. 187초·7표본에서 처리 카운터6942→12535, 드롭/이벤트 실패0·video_ready=true. 재부팅·새 장비·장시간/다중 카메라 미검증, I43 VERIFY·P5.1 IN_PROGRESS 유지. [실행 근거](evidence/environment_20261002.json) · [설치·검증 절차](../guides/JETSON_EDGEX_FIELD_CHECKLIST.md#1-1-호스트와-컨테이너-의존성-준비). 최종 전체 회귀2717 PASS/52 SKIP/1 FAIL(DB 동시 쓰기 경합), 해당 단독 재시험1 PASS. 실패 경로4파일은 작업 전과 동일하나 원인 미확정으로 I45 OPEN을 등록했다.

v0.93: [복귀 매칭·TCN·CPU 대조](MODEL_INFERENCE_METRICS.md#rollback-match-timing-v093). 보존243표본 분석: 누락13건 중 TCN 시작/진행 구간5·반환 후8, CPU 제한 증가 없는 구간9. 후보/복귀 코드4파일 해시 일치. 개별 프레임 사유는 미확정·I32/I38 OPEN. 새 GPU 시험·제품 변경·운영 배포 없음.

v0.92: [I07 후보→이전 구성 전환](VIDEO_INFERENCE_CONTRACT.md#transition-rollback-v092). 동일 격리 DB에서 후보 실행·제거→이전 이미지 복귀. 후보 신규 이벤트31건/외형28건을 포함한705517행 보존·중복 재접수 시 행/내용 변경0. 복귀 후 이벤트32건/외형28건 추가·영상3009프레임 디코드. 후보/복귀 종료6.37/3.49초·복귀 매칭 누락13·TCN Pose 부족 잔여. 운영 미배포·I07 OPEN.

v0.91: [I07 복구본·이전 이미지 격리 기동](VIDEO_INFERENCE_CONTRACT.md#restore-bundle-v091). 49,917파일 해시 대조·설정12파일 암호화 복원·이전 이미지 내보내기/재적재. 기존705,398행 보존·대기32건/외형28건 추가·SIGTERM2.59초 종료. 첫 기동 timeout·TCN timeout1·매칭 누락17·최대 영상 간격1.936초는 잔여. 후보→이전 구성의 연속 전환은 후속·운영 미배포·I07 OPEN.

v0.90: [I07 후보 전체 기동·복구 의존성](VIDEO_INFERENCE_CONTRACT.md#profile-start-v090). 캐시 연결 디렉터리 누락 I42 수정·종료 유예60초 명시·46 PASS. 격리180초 입력/추론/송출·DB 저장, SIGINT25.12초/SIGTERM4.86초 정상 종료. 누적 이벤트82건 대기·외형73건/DB 무결성 정상. 외부 의존성 약7.69GB는 별도 복구본 준비 잔여·운영 미배포·I07 OPEN.

v0.89: [I07 후보 프로필·영속 저장 검사](VIDEO_INFERENCE_CONTRACT.md#candidate-profile-v089). 고정 이미지·읽기 전용 모델·영속 DB·필수 비밀값을 연결, 구성 회귀44 PASS. 격리 컨테이너 재생성 후 합성 대기2건 보존/중복 방지, 이미지359파일·모델8개 해시 일치. 전체 AI 통합 기동/이전 구성 복구는 후속이며 운영 미배포·I07 OPEN이다.

v0.88: [I07 구성 차이·DB 복구](VIDEO_INFERENCE_CONTRACT.md#rollback-preparation-v088). 재전송 DB 백업 누락 I41 보완·5 PASS, 실제 SQLite8개 복원·1717566행/스키마/상태 동일. 현재 디스크351파일·모델8개 보존, 비밀값/시작 당시 메모리·전체 구성 복구는 미검증. 다음은 운영용 후보의 영속 DB·비밀값·마운트 프로필 확인이며 AI 재배포 없음·I07 OPEN이다.

v0.87: [일반 실행 매칭·TCN 대조](MODEL_INFERENCE_METRICS.md#match-tcn-v087). 기준180초·단절/복구 시험 정상 종료, CPU 영상1497개 전부 등록 메타와 대응·후처리327/정상 제한1170. TCN18회 반환(유효17·Pose 부족1), timeout0·재연결8.23초. 과거 I32/I38 원인은 미확정·I34 OPEN·운영 미배포. 다음은 I07 고정 후보·이전 구성의 차이표와 격리 복구 준비다.

v0.86: [CUDA 할당 경계·수명 검사](TRACKER_VALIDATION.md#cuda-bounds-v086). padding32·비동기 할당 수명 검사를 추가해 원 보존본/고정 후보 각60프레임·정상 종료·메모리 오류0을 확인했다. 원 CUDA700은 미재현·I34 OPEN. 검사 중 TCN timeout과 매칭 누락은 별도 보존, 제품 코드·운영 배포 변경 없음. 다음은 일반 실행의 I32 매칭 사유·I38 상태 대조, 이후 I07 복구 준비다.

v0.85: [RTSP 종료 재활성화 수정·후보 이미지 검증](TRACKER_VALIDATION.md#rtsp-shutdown-fix-v085). 출력 bin이 종료 중 다시 열리는 결함 재현·상태 잠금 보완, 관련36 PASS. 공식 RTSP 보완을 포함한 조합5조건 종료1.62~5.27초·복구6.36초, 영상6파일 디코드/원본 DTS 정상. 후보 이미지 고정·I40 FIXED_LOCAL, I34 CUDA700/I32/I07 잔여·운영 미배포.

v0.84: [출력 장애 종료 대기 분리](TRACKER_VALIDATION.md#output-shutdown-v084). 진단 없는 정상 종료6.25초·출력 서버만 단절하면30초 초과 재현. RTSP caps 대기 native 스택 확보·공식1.20.2 패치 격리 빌드. 수정본 일반 실행의 전체 NULL 대기가 남아 I40 OPEN·배포 보류. CUDA700 해결 근거와 구분.

v0.83: [종료 시간 초과 재현·순서 보완](TRACKER_VALIDATION.md#shutdown-recovery-v083). 인코더 버퍼 풀 회수 대기 native 스택 확보·RTSP 출력 우선 종료로 최소 수정. 실제 후보 정상/입력 단절 종료7.23/2.49초·46 PASS. I40 추가 회귀·I34/I32/I07 잔여, 문제 해결·검증 전 배포 보류.

v0.82: [종료 단계·운영 배포 대조](TRACKER_VALIDATION.md#shutdown-stages-v082). 종료 상태 전환7.47초/보조 회수0.34초 분리·추가2회 정상이나 최초30초 초과 미재현. native 스택 미확보·I40 OPEN. 운영AI09-17 이후 재배포 없음·새 계측 미노출, bind mount와 실행 코드 차이 명시.

v0.81: [현재 후보 모델 실행 대조](MODEL_INFERENCE_METRICS.md#model-state-v081). 격리90초×3회·모델 실행/무결과/미호출 구분. 얼굴 Haar 검출 실행/결과0, RF 미호출·Python 속성 Null 경로와 SGIE 구분. 종료30초 초과1회·재시험2회 정상이나 I40 OPEN. 영상3파일 디코드·PTS/DTS 대조 통과·운영 미반영.

v0.80: [CUDA 최초 조건 대조·GPU 덤프 준비](TRACKER_VALIDATION.md#cuda-comparison-v080). 원 보존본 비동기/동기 각120프레임 통과·실제 적재 영상 라이브러리4개 호스트 해시 일치. 예외 덤프 선택 기능·최초719 기록 보완/12 PASS·수동 덤프719는 원700과 구분. I34 OPEN·운영 미반영.

v0.79: [단절 후 녹화 시간 대조](TRACKER_VALIDATION.md#match-timing-v079). 같은 출력의55/90초 수신기를 동시 대조·2회 확인. 약72초의 복구 프레임이55초 제한을 넘어 짧은 파일 생성;90초 파일에는 복구 후 영상 포함. 4파일 디코드 정상·매칭 누락 재현0. 원 매칭/CUDA 원인 미확정·I32/I34/I07 OPEN·운영 미반영.

v0.78: [현재 후보 종료 SIGABRT 수정](TRACKER_VALIDATION.md#candidate-shutdown-v078). 보조 영상 쓰기 중 2초 대기 후 종료하는 결함 재현·native/Python 스택 확인. 작업 완료 대기·모델 자식 회수, 256 PASS/2 SKIP. 인코더 포함/제외·재연결 중 종료·실제 RTSP 재시험 완료. I39 FIXED_LOCAL·I34 OPEN·운영 미반영.

v0.77: [입력 감시·NVDEC/NVENC·네이티브 대조](TRACKER_VALIDATION.md#source-boundary-v077). 무수신 경고 순간에도 디코더/추론은 직전0.05초까지 진행·큐21개 관측. 검사 전용 재연결60초는40프레임 목표/메모리 오류0. 네이티브 RTSP 대조는 EOS 개입으로 원인 판정 제외·파일 경로 별도 확인. 9 PASS·I34 OPEN·운영 미반영.

v0.76: [CUDA 최초 오류 보존·RTSP 메모리 검사](TRACKER_VALIDATION.md#cuda-capture-v076). 17개 입출력 지점·최근4096기록 보존, 합성 오류/과거 로그 수집 확인·7 PASS. 시작3조건 CUDA700 미재현. 메모리 검사2회 시간초과를 보존, 검사 전용 자동 재연결 해제 후40프레임·최종 memcheck 오류0. I34 OPEN·운영 미반영.

v0.75: [GPU/VIC 비교·P5.1 종료 점검](VIDEO_INFERENCE_CONTRACT.md#p51-closeout-v075). 각120프레임×2회 반복 결과 일치, 장치 간 관절 최대81px/유효점 경계119개 차이로 VIC 전환 보류. GPU 후보359파일 고정·단절 후8.07초 복구. 10-01 종료 검토 목표·I34/I07 등 미해결로 DONE 확약 불가.

v0.74: [CUDA 시작 경로 분리](TRACKER_VALIDATION.md#cuda-startup-v074). 보존본2회/현재 후보2회·Primary VIC1회 시작/35초 관측 통과. GPU 후보120프레임 메모리 검사 오류0·출력5파일 디코드 정상. Primary 전처리와 후단 복사 설정 구분·I34 근본 원인 미확정·운영 미반영.

v0.73: [TCN 재사용 동시 부하600초](MODEL_INFERENCE_METRICS.md#tcn-loaded-v073). 녹화1채널·TCN47회 완료, 첫12.92초/후속1.40~3.13초. timeout·후처리 큐 폐기/실패·CUDA 서명0. Pose SGIE/RF·얼굴 추론·관제 ACK 미검증, 출력 간격/I34 원인 확인 잔여·운영 미반영.

v0.72: [TCN 전용 모델 재사용](MODEL_INFERENCE_METRICS.md#tcn-worker-v072). 선택형 자식 프로세스 재사용·오류/timeout 종료·새 요청 복구. Jetson 동일 영상 단발11.26~12.42초→준비 후1.16~1.60초, 결과 일치. 영상 교체/자식 강제 종료 복구·194개 고유 테스트 통과. 기본false·운영 미반영·I38 원timeout/장시간 및 I37/I32/I34/I07 잔여.

v0.71: [RF 표본 선택 기준 보존](MODEL_INFERENCE_METRICS.md#rf-sampling-contract-v071). 캡처에 고유 균등 표본 정책/실제 최대 개수 기록·학습48개 고정 제거. 미상/혼합 기준 거부·205 PASS, 라벨 도구 포함 통합2건 재확인. 원본 탐색 범위에서 복원 못함·운영 미반영. I37 점수/원본·I38/I32/I34/I07 잔여.

v0.70: [RF 수집 시간창 보존](MODEL_INFERENCE_METRICS.md#rf-window-v070). 학습 시 수집 시간창 무시·3초 고정 결함 수정. 누락/비정상/혼합 시간창 거부, 수집→학습→재로딩→추론 합성 통합 포함155 PASS. 기존 후보 원본5개는 기록 경로에 없어 학습 조건 검증 보류. 운영 미반영·I37/I38/I32/I34/I07 잔여.

v0.69: [RF 학습 출처 검사](MODEL_INFERENCE_METRICS.md#rf-source-v069). 다른 출처/메타데이터 없는 RF의 inline 판정 차단·기존 fail-open 정책/무필터 특징 수집 유지.174 PASS(회귀142+연동32)·실제 불일치 모델 예측0회/호환 후보1회 확인. 기존 RF는 새 후보 적용 시 사용 불가·모델 교체/운영 배포 없음. I37 세부 기준/I38/I32/I34/I07 잔여.

v0.68: [RF 중복 표본 제거·TCN 단계 계측](MODEL_INFERENCE_METRICS.md#fall-sampling-v068). 부족한 Pose를48개로 반복하던 문제 로컬 수정·116 PASS. 실CCTV600초 TCN22회 완료/timeout0·큐 폐기/후처리 실패/재시작/CUDA 서명0·외형DB361건. I37 나머지 특징 계약/I38 기존 timeout 원인/I32/I34/I07 잔여·운영 미반영.

v0.67: [단일 CCTV30분·낙상 전처리 점검](MODEL_INFERENCE_METRICS.md#fall-contract-v067). 기존 후보30분 큐 폐기/후처리 실패/재시작/CUDA 서명0·외형DB847건. TCN 입력640 명시 전달 로컬 수정·170 PASS. TCN timeout24건/I37 특징 계약/I32 매칭122건 잔여·I34 원인 미확정·운영 미반영.

v0.66: [후처리 큐 병목 완화](TRACKER_VALIDATION.md#v066--후처리-큐-병목-완화). GPU 버퍼 재사용·전용 스트림, 인물별 최대2Hz·미완료 중복 방지. 실CCTV180초 큐 폐기/후처리 실패0·218접수/완료·대기 평균46.6ms/P95 363.1ms. 308 PASS/2 SKIP. I36 FIXED_LOCAL·I34 원인 미확정·운영 미반영.

09-22 [Archify 구조·데이터 흐름·시퀀스](../architecture/archify/index.html)를 추가했다.
현재 소스/기존 검증 기록을 설명하는 자료이며 장비 재점검·운영 전환·새 WBS 완료 근거는 아니다.
각 산출물 9/9 검사와 브라우저 검증 제한은 [사용법/근거](../architecture/archify/README.md)에 기록했다.

기존에 구현하고 시험한 기능을 아래 범위까지 성과로 인정한다. 새 WBS의 `DONE`은
해당 행의 완료 조건 전체를 만족했을 때 사용한다. 일부 시험 통과나 파일 존재를
전체 모듈·현장 POC 완료로 확대하지 않는다. 전체 프로젝트 완료율은 아직 산정하지 않는다.

## 2026-10-02 운영 환경 추가 점검

아래 최초 점검·부분 복구 기록 이후 v0.94에서 의존성 설치와 운영 프레임 복구, health/자동 시작 보완을 반영했다. 현재 판정은 상단 v0.94와 I43/I44 행을 따른다.

L4T36.5.2/Ubuntu22.04.5·CUDA12.6.68·TensorRT10.3.0.30·cuDNN9.3.0.75 확인. DeepStream7.1은 설치돼 있으나 호스트 직접 실행은 RTSP 서버 라이브러리 누락으로 실패한다. 컨테이너에서7.1 버전·실제 라이브러리 로드를 확인했다. jtop4.3.2에는36.5.2 매핑이 없다. 30W·CPU8/12·CPU최대1.728GHz/GPU최대612MHz이며 운영 컨테이너의 별도 CPU/메모리 상한은 없다.

운영 AI 시작은10-02 08:40 KST·healthy/restart0이다. 최초 읽기 전용 점검에서는 X11 인증·EGL 초기화 실패를 확인했다. 이후 사용자 승인으로 `xhost +SI:localuser:root`만 적용해09:43:42 KST 파이프라인 시작 로그를 확인했다. 그러나09:49:42까지 처리 통계 frames=0이며 영상 처리 정상 판정은 불가하다. 카메라 OPTIONS/DESCRIBE는 RTSP200·H.264를 반환했으나 RTP 수신/디코드 성공은 미검증이다. `/health`는 HTTP200/status=ok/camera_count=1로 응답하며 파이프라인 프레임 상태를 검사하지 않는다. 자동 시작 서비스는10-01 17:33:47 xhost의 display :1 열기 실패로 failed이며, 인증 파일 존재만 기다리고 실제 인증은 한 번 시도하는 구성이다. 세션 변경과 권한 소실의 정확한 경위는 미확정이다. 패키지·전력 변경과 서비스 재시작은 없었다. [원자료·후속 복구](evidence/environment_20261002.json).

10-02 사용자 방향에 따라 운영 안정화를 먼저 진행한다. 순서는 실제 프레임 진행 복구 → X11 자동 준비/재시도와 영상 상태 판정 보완 → 단일 카메라 연속 운전 및 재시작 복구 검증 → 기존 I32/I38 분석·다중 카메라 검증이다. 현재 지원 조합을 기준으로 고정하고 문제 근거가 있는 구성만 수정한다. 진행판 변경은 전체 SDK 교체·재부팅·일괄 재배포 승인이 아니며, 아직 안정화 합격/완료로 판정하지 않는다.

## 1. 기존에 확보한 성과

v0.65 당시 후보 실CCTV 회귀:60초 관측·최대1422처리프레임·재시작0·CUDA 오류0. 수신30.203초/693패킷의 전체 디코드 오류0. 후처리 큐 폐기307건은 I32 잔여이며 무손실 합격 아님. 운영 healthy/restart0·시작 시각 유지·배포 없음.

v0.65: [파일→RTSP 시작·종료 대기 수정](TRACKER_VALIDATION.md#v065--파일rtsp-시작종료-대기-수정). I35 FIXED_LOCAL: preview 첫 버퍼 대기 해제·NULL 전환 전 FLUSH_START. 실제 앱 전체30프레임 CUDA 메모리 오류0·RTSP 게시·정상 종료. 182 PASS/2 SKIP. 최초 CUDA700 원인(I34)은 미확정·운영 미반영.

v0.64: [실CCTV·CUDA 최소 경로 검사](TRACKER_VALIDATION.md#v064--실cctv와-cuda-최소-경로-검사). RC4 실CCTV VIC600초/GPU180초·CUDA 오류 미재현. 단독 Primary와 최소 경로 메모리 검사 결과 확보. VIC 후처리 큐 폐기1535건은 I32 잔여. 최초 CUDA 원인 미확정·I34 OPEN·운영 미반영.

v0.63: [CUDA 연속·대조와 치명적 오류 종료](TRACKER_VALIDATION.md#v063--cuda-연속-대조와-치명적-오류-종료). RC3 600초 관측·최대14389처리프레임, RC4 GPU/VIC 각2회 오류 미재현. 손상된 프로세스 재사용 차단·실제 앱 bus 오류 주입 종료1 확인·255 PASS/2 SKIP. GPU 대조도 통과하여 VIC 인과 효과 미입증·I34 OPEN·운영 미반영.

v0.62: [CUDA 메모리 오류 진단·VIC 복사 후보](TRACKER_VALIDATION.md#v062--cuda-메모리-오류-진단과-vic-복사-후보). 원 장애의 GPU MMU fault 확인, 공식 VIC 복사 우회 설정 추가. RC3 격리3회 CUDA700 미재현·단절 복구7.876초·수신4파일 디코드 오류0, 248 PASS/2 SKIP. 원인 확정/영구 해결 아님·I34 OPEN·운영 미반영·P5.1 IN_PROGRESS.

v0.61: [시작 반복·후처리 누락 계측](TRACKER_VALIDATION.md#v061--시작-반복과-후처리-누락-계측). 격리4회 각90초 완료·CUDA700 미재현. TCN을 꺼도 큐 폐기8건, 외형 분석 평균92ms. 후처리 Prometheus 지표6개 추가·실영상 폐기5건 노출 확인, 213 PASS/2 SKIP. 큐 포화·I34 해결은 아님·운영 미반영·P5.1 IN_PROGRESS.

v0.60: [RC2 장구간 관측·적용 범위](TRACKER_VALIDATION.md#v060--rc2-장구간-관측과-적용-범위). 첫 시작 CUDA 오류 I34 OPEN, 동일 후보 재실행은 330초 관측·300.333초/8194프레임 수신·디코드 오류0. 상대 적체 +0.304초/302.504초, 후처리 큐 폐기22건으로 무손실 합격 아님. 운영 변경 없음·P5.1 IN_PROGRESS.

v0.59: [RC2 영상 대응·DB 복사본 검증](TRACKER_VALIDATION.md#v059--rc2-영상-대응과-db-복사본-검증). 후처리327건 식별자/영상 해시 일치·native/preview11쌍 원본PTS 일치. 소스357개·모델8개 해시 확인, DB 기존485568건 보존·큰ID3종 재조회 통과. 가벼운29.97fps 관측의 상대 적체+0.638초/약20초는 잔여. 제품 코드/운영 변경 없음·P5.1 IN_PROGRESS.

v0.58: [CPU OSD 복구·출력 시간 정보 보완](TRACKER_VALIDATION.md#v058--cpu-osd-단절-복구와-vui-보완). 격리 후보3회에서 입력25초 단절 후7.794~10.984초에 처리 재개·전체 재시작0. VUI 기록 보완으로 전후6파일 정상 시간 헤더·디코드 오류0, 167 PASS/2 SKIP. I33 FIXED_LOCAL·I20/I32/I07 OPEN·운영 미반영·P5.1 IN_PROGRESS.

v0.57: [OSD 초기 정지 우회 검증](TRACKER_VALIDATION.md#v057--osd-초기-jit-정지와-cpu-우회-검증). native 스택에서 GPU OSD의 PTX JIT 대기를 확인하고 CPU 그리기 선택 설정을 추가했다. 실제 수정 소스 15/29.97fps 재생에서 최대 프레임 간격0.761/0.840초·디코드 오류0, 167 PASS/2 SKIP. I20/I33 OPEN·운영 미반영·P5.1 IN_PROGRESS.

v0.56: [초기 정지 원인 분리](TRACKER_VALIDATION.md#v056--초기-정지-원인-분리와-판정-정정). 출력 수정 전/출력 없는 과거 경로에도9~10초 초기 멈춤이 있어 v0.55의 출력 회귀 단정을 정정한다. 현재 후보의 재현·버퍼 복사/풀 확대·TCN 제외·파일 출력 등으로 분리했으나 확정 수정안은 없다. I20/I33 OPEN·운영 적용 보류·P5.1 IN_PROGRESS.

v0.55: [RF·TCN 실행 검증](MODEL_INFERENCE_METRICS.md#v055--rf·tcn-실제-호출-검증). 녹화 입력으로 RF 명시적 부품 호출 1회·TCN 유효 자세 결과 3건 확인. 자연 RF 경보 호출·정확도는 미검증. v0.54 출력 수정 후보가 1080p15 입력에서 정지하고 출력 제외/수정 전 후보는 진행하여 **I33 OPEN으로 재개**. 내부 원인은 추가 분리 필요, 운영 적용 보류·P5.1 IN_PROGRESS.

v0.54: [I33 출력 시간 정보 수정](TRACKER_VALIDATION.md#v054--i33-출력-시간-정보-수정). mux의0/1 FPS가 인코더에 전달되는 원인 확인, H.264 출력에 제한된 videorate 변환 적용. 실제 소스 격리 실행에서 수신386프레임/20.210초·정상 시간 헤더·디코드 경고0. 158 PASS/2 SKIP, I33 FIXED_LOCAL·운영 미반영·P5.1 IN_PROGRESS.

v0.53: [RC1 기능 통합 실행](TRACKER_VALIDATION.md#v053--rc1-기능-통합-실행). 격리 2회에서 Pose/헬멧/의상/색상·TCN 호출·RTSP 수신·DB 저장 확인. 얼굴 344호출 완료(결과 0), inline RF 실호출 미관측·TCN 자세 부족. H.264 시간 헤더 오류 I33 OPEN, POC 보정 해제로 해결 안 됨. 운영 변경 없음·P5.1 IN_PROGRESS.

v0.52: [운영 설정 대조·RC1 고정](TRACKER_VALIDATION.md#v052--운영-설정-대조와-rc1-고정). 환경 키 134개·설정 4개 대조, 소스/설정 357파일·모델 8개 별도 보존 및 해시 확인. 운영 Pose 엔진·얼굴·TCN·RTSP 송출이 기존 격리 시험과 달라 RC1 통합 실행은 미검증. 운영 변경 없음·P5.1 IN_PROGRESS.

v0.51: [관측 부하 대조·후보 복원 실행](TRACKER_VALIDATION.md#v051--관측-부하-대조와-후보-복원-실행). 복원 357파일/모델4개 해시 일치, 복원 후보 후처리 178건 영상/식별자 일치. 무거운 관측/CPU2 상대 적체 +1.214초, 가벼운 관측/CPU2 −0.128초·CPU4 +0.382초(각 마지막 약20초). CPU 제한 발생 0, 운영 CPU 확대 근거 없음. 제품/운영 변경 없음·P5.1 IN_PROGRESS.

v0.50: [실CCTV 수집 영상 검증·후보 보존](TRACKER_VALIDATION.md#v050--실cctv-수집-영상과-적용-준비). 1080p/29.97fps 약 25초를 수집해 격리 재생, 후처리 183건 영상/식별자 일치·변환/저장 오류 0. 마지막 약 20초 상대 적체 +2.684초로 지연 합격 아님. 후보 소스/설정 357파일 보존·해시 대조, 이전 운영 복구본과 구분. 전신 시험은 사용자 사정으로 미검증, P5.1 IN_PROGRESS·I32/I20/I07 OPEN.

v0.49: [동일 입력 재시작·단절 복구](TRACKER_VALIDATION.md#v049--반복-시작과-단절복구)를 격리 2회 검증했다. 송신 25초 중단 후 각각 9.287/9.308초에 처리 재개, 후처리 371건 식별자·영상 해시 일치·복구 후 새 세대만 관측. 파이프라인 재시작/서버 폐기 경고 0. 실CCTV·초기 정지 재발 방지·운영 계측은 잔여, I20/I32/I07 OPEN·P5.1 IN_PROGRESS.

v0.48: [입력 전송 대조](TRACKER_VALIDATION.md#v048--시험-rtsp-전송-손실-분리). 격리 6회 실행에서 FPS 단독 조정으로 손상이 해결되지 않음을 확인. 동일 4K15 입력의 시험 서버 큐 512→4096 확대 후 2회 각각 40초 실행·폐기 경고 0건·검토 표본 개선. 초기 준비 19.393~57.918초·실CCTV/절대 지연은 잔여. I32/I20/I07 OPEN, 제품/운영 변경 없음·P5.1 IN_PROGRESS.

v0.47: [I31 추적 ID 저장·조회 보정](APPEARANCE_TRACK_ID.md). 정확한 10진 TEXT 저장·API 문자열 필드 추가, 이전 DB 호환·저장 실패 후 재시도 보완. **67 PASS**, 실제 영상 의상 로그 13건 ID 일치·큰 ID 3종 별도 저장/재조회 통과. I31 FIXED_LOCAL, 운영 미반영·P5.1 IN_PROGRESS.

v0.46: [추적·입력 대조](TRACKER_VALIDATION.md). RTSP 지연 패킷 폐기 선택 설정 추가(기본 유지), **172 PASS / 2 SKIP**. 해제 시 영상 표본 개선이나 4K/59.94fps 처리 지연은 잔여(I32 OPEN). NvDCF 첫 시험 10프레임 정지·재시험 Primary 644프레임/35초 완료, 영상 추출 유지. I20/I31 OPEN·IOU 유지·운영 미반영·P5.1 IN_PROGRESS.

v0.45: [I30 좌표·ROI 보정](VIDEO_INFERENCE_CONTRACT.md#16-v045--mux-좌표roi-경계-보정). tensor·ROI·Pose·이벤트 크기를 mux 좌표로 통일하고 경계 밖 ROI를 제한했다. **230 PASS / 2 SKIP**, 실제 4K→1080p·4:3 padding 시험에서 변환 오류 0. I30 FIXED_LOCAL. **동영상 첫 ROI/preview 위치 불일치·고정 영상 화면 손상은 I32 OPEN**, 큰 추적 ID의 의상 DB 저장 실패는 I31 OPEN. 운영 미반영·P5.1 IN_PROGRESS.

v0.44: [모델별 실제 출력 대조](MODEL_INFERENCE_METRICS.md). 일반 Pose 경로의 SGIE 사람 영역 등록·탐지 완료 표시 누락을 보완했다(I29 FIXED_LOCAL). 원본/처리 해상도 일치 시험에서 Primary 325·의상 266·헬멧 266프레임 tensor 관측, 색상 229호출/228완료. **201 PASS / 2 SKIP**. 4K 원본→1080p 처리 좌표 오류는 I30 OPEN, 운영 미반영·P5.1 IN_PROGRESS.

v0.43: [동일 프레임 결합](VIDEO_INFERENCE_CONTRACT.md). native 프레임 토큰·세대·camera/frame/source PTS로 탐지와 CPU 영상을 연결하고 최신 이미지 조회를 제거했다. **235 PASS / 2 SKIP**. 실제 단일 입력·단절/복구에서 후처리 128건의 식별자·영상 SHA-256 일치, 새 세대 108건 처리. I27 FIXED_LOCAL, I28/I07 OPEN, 운영 미반영·P5.1 IN_PROGRESS.

v0.42: [프레임 식별·종료 점검](VIDEO_INFERENCE_CONTRACT.md). 다른 카메라 영상 fallback 제거·reset 시 기존 preview 삭제·native camera/frame/source PTS 보존. **172 PASS / 2 SKIP**. 실제 단일 입력 65개 preview 수신 확인. I27 동일 프레임 결합·I28 batch2 preview 변환 오류 OPEN, 운영 모델 완료 지표도 미노출(I07). P5.1 IN_PROGRESS.

v0.41: [얼굴·의상 비동기 세대 검증](VIDEO_INFERENCE_CONTRACT.md). 대기/추론 중 재연결 시 이전 세대의 캐시·DB·이벤트 반영을 차단하고 다른 카메라와 새 작업을 유지한다. 신규 9개 포함 **211 PASS / 2 SKIP**(이번 회귀 묶음). I26 카메라 간 의상 억제 상태 공유 FIXED_LOCAL. 운영 미반영, P5.1 IN_PROGRESS.

v0.40: [native 재연결 경계](VIDEO_INFERENCE_CONTRACT.md)에서 Pose 기록 잔류를 재현하고 카메라별 추적·판정 캐시 초기화와 reset 신호 계측을 보완했다. 격리 재시험에서 이전 기록 제거·세대 1→4, 정상 채널 세대 2 유지·전체 재시작 0. **258 PASS / 2 SKIP**. I20 OPEN, P5.1 IN_PROGRESS, 운영 미반영.

v0.39: [실제 CLI 2입력 단절·복구](VIDEO_INFERENCE_CONTRACT.md) 시험. RTSP 재연결 선택 설정 추가(기본 0 유지), 시험값 15초에서 재발행 후 약 14.9초에 프레임 재개·정상 입력 진행·전체 재시작 0. **140 PASS / 2 SKIP**. I20 OPEN, P5.1 IN_PROGRESS.

v0.38: [격리 영상 실행·보존본 재실행](VIDEO_INFERENCE_CONTRACT.md) 확인. 실제 Primary 출력 메타 190/215프레임(각 약 3초), 복원본 **225 PASS / 2 SKIP**. 운영 재시작 0회·healthy 유지. 전체 앱 복구·2입력 단절은 미검증, P5.1 IN_PROGRESS.

v0.37: [Pose 입력 수치 검증·적용 준비](VIDEO_INFERENCE_CONTRACT.md) 완료. 잘못된 인물 입력 격리·scope 오연결 차단, **197 PASS / 2 SKIP**. 현재 Python 소스 321개 해시/보존본 준비. 운영 이전 코드 복구 기준·실영상/단절 검증은 잔여, P5.1 IN_PROGRESS.

v0.36: [I20 입력 오류·재시작 계측](MODEL_INFERENCE_METRICS.md) 로컬 보완. ERROR/EOS 식별별 집계·명시적 재시작 결과·Prometheus 연결, **134 PASS / 2 SKIP**. I20 OPEN, P5.1 IN_PROGRESS; 실제 단절/프레임 복구는 미검증.

v0.35: [Copilot 변경 검토·TCN 보완](MODEL_INFERENCE_METRICS.md). 색상/의상 계측과 특징 캐시를 확인했고, 최소 구간 하향·실행 수 추정을 보완했다. **284 PASS / 2 SKIP**. 다음은 P5.1 입력 품질·단절/I20 검증이다.


v0.34: [로컬 GI 복구](MODEL_INFERENCE_METRICS.md) 완료. 기존 SKIP 5개 모두 PASS, 인접 회귀 **148 PASS / 2 SKIP**(미설치 전용). P5.1은 잔여 계측·입력 품질/단절·운영 활성화 검증 후 완료 판정한다.


Copilot 기존 기록(v0.36 표기): [색상/외형 backend Python 호출 계측](MODEL_INFERENCE_METRICS.md)을 추가했다. `_color_backend`/`_backend`
(PA100K/PP-Human/HSV) 두 predict() 호출을 requests/completed/errors/seconds로 구분하고, 색상은 crop이
너무 작아 호출 자체를 못한 경우(`skipped_too_small`)도 별도로 센다. 신규 테스트 포함 **243 PASS / 5 SKIP**.
"모델이 로드됐는지"가 아니라 "Python 호출 경로를 몇 번 탔는지"만 보는 계측이며 운영 반영은 남는다.

v0.35: [TCN 실제 윈도 추론 수 계측](MODEL_INFERENCE_METRICS.md)을 추가했다. `process_calls`(subprocess
실행 횟수)와 별개로 실제 model forward 횟수(`window_inferences`)를 센다. 신규 2개 포함 **191 PASS / 5 SKIP**.
GPU 실행 시간 대체가 아니며 운영 반영은 남는다.

v0.34: [Baseline/Candidate temporal 비교 도구의 pose 특징 캐시 재사용](MODEL_INFERENCE_METRICS.md)을
실제 subprocess 실행 4건으로 스모크 검증했다. 현장 클립 4개(전부 낙상, 비낙상 없음) 기준
baseline recall 50%, candidate recall 50%. 표본이 너무 작아 모델 품질 판단 근거가 아니며
cache 배선이 에러 없이 동작함만 확인했다. 전체 validation 2,720건 재비교와 문서 반영은 남는다.

v0.33: [TCN 보조 프로세스 계측](MODEL_INFERENCE_METRICS.md) 로컬 완료, 신규 7개 포함 **190 PASS / 5 SKIP**. 실행/실패/timeout과 출력 해석을 구분한다. 실제 모델 윈도 추론·잔여 Python 계측·운영 검증은 남으며 P5.1 IN_PROGRESS다.


현재 v0.32: [inline RF 실행·캐시 계측](MODEL_INFERENCE_METRICS.md)을 추가했다. 신규 6개 포함 **183 PASS / 5 SKIP**. 호출/실패/시간과 캐시 반환을 구분하며, TCN 등 잔여 계측·운영 활성화는 남는다. P5.1 IN_PROGRESS, I07 OPEN 유지.


P0.1 재확인은 9개 파일의 CPU·모형·임시 SQLite 테스트다. 아래 초기 시험 수치는
그 기준선이며 후속 검증은 E01·4절·진행 기록에 구분한다. 실물 명령이나 GPU 추론은
실행하지 않았다. 과거 결과는 날짜와 범위를 함께 표시했다.

| ID | 확보한 성과와 완료 범위 | 확인 근거 | 남은 일 / 대응 WBS |
|---|---|---|---|
| E01 | 공통 정규화·센서/영상 관측 일부 구현, P1.1 공통 계약 확정 | 기준선 canonical 4개, 후속 로컬 회귀 147 PASS. 경계 진단 9개 중 7개 보관 기준 미충족; [명령 ID 계약](../architecture/COMMAND_CORRELATION_CONTRACT.md) | 계약의 전체 경로 적용·영속 오류 보관·저장/조회·명령 정책·운영 검증 — P1.2~P1.4/P2 |
| E02 | HTTP/MQTT 전달 대기와 센서 EdgeX outbox 저장·상태 전환 구현 | 현재 HTTP 2개·MQTT 3개·parser outbox 7개 PASS | 공통 이벤트와 전달 대기 원자성, DB 장애·재시작·용량 시험 — P2.1~P2.4 |
| E03 | TLV 해석, 온습도(t34954)·기울기(t34955) 예제 해석 구현 | 현재 TLV 테스트 16개 PASS; Go 원본 예제와 필드 대조 | 타 부서 최신 규격·실물 값·단위·원본 추적 재확인 — P4.1 |
| E04 | 센서 규칙에서 온도·기울기 경보 생성 경로 구현 | 현재 sensor rule bridge 테스트 8개 PASS | 경계·복귀·중복·오래된 값·공통 정책, 실물 전체 흐름 — P4.2~P4.3 |
| E05 | 전광판·스피커·사이렌 Device Service와 모형 명령/결과 계약 구현 | 현재 계약 테스트 1개에서 세 장치 `simulated` 확인; 세 컨테이너 healthy | 대상·동시 중복·부분 실패·장치 격리와 실물 확인 — P3.1~P3.6 |
| E06 | 전광판 EdgeX 명령→장치 호출→결과 DB 경로 시험 수행 | [09-17 UAT](../handover/EDGEX_SIGNBOARD_UAT_20260917.md): acknowledged, 약 1.03초 | 현장 육안/사진, 관제 대조, 실패·복구 증거 — P3.4는 VERIFY |
| E07 | API·EdgeX 어댑터·결과 수집·Prometheus/Grafana 서비스 구성 | 현재 Docker 실행 스냅샷; 해당 health가 있는 서비스는 healthy | 실제 외부 관제 수신·화면·ID 대조, 출력 대기와 전달 분리 — P2.3/P3.2/P6.1 |
| E08 | 영상 입력·DeepStream 분석 경로와 모델 파일 구성 | AI 컨테이너 healthy; 로컬 카메라 설정 1대 enabled; 모델 파일 6개 해시 기록 | 현재 프레임 진행, 전체 모델 완료 계측·정확도·복구 — P5.1/P6.2 |
| E09 | Primary 추론 계측 교정과 격리 GPU 검증 수행 | 현재 단위 테스트 23개 PASS; [09-17 교정 시험](../testing/JETSON_PRIMARY_LOAD_TEST_20260917.md) | 로컬 수정의 운영 프로세스 반영, 전체 모델·실물 다중 카메라 검증 — P5.1/P6.2 |
| E10 | 낙상 Baseline/Candidate 학습·비교·shadow 도구와 저장 결과 확보 | [09-17 모델 감사](../reports/TRAINED_MODEL_AUDIT_20260917.md), [09-16 현장 계획](../handover/FIELD_UAT_AND_MODEL_SHADOW_PLAN_20260916.md) | 시간창·pose 부족·캐시·현장 오탐/미탐 보완. Candidate 승격 보류 — P5.2 |
| E11 | 헬멧 엔진·평가 도구·저장된 정확도 결과 확보 | [모델 감사](../reports/TRAINED_MODEL_AUDIT_20260917.md): 저장 평가 precision 90.14%, recall 91.10% | 해당 수치는 헬멧 클래스 평가. 미착용 경보·원거리·ROI·현장 품질 별도 — P5.3 |
| E12 | 의상·색상 분석 경로와 학습/변환 후보 확보 | PA100K·색상 엔진 존재; [모델 감사](../reports/TRAINED_MODEL_AUDIT_20260917.md) | 활성 backend별 평가, 부족 클래스·unknown·관제 확인. 변환 성공은 품질 합격과 구분 — P5.4/P5.5 |
| E13 | 운영·복구·현장 시험·모델 평가 문서와 도구 확보 | [인수인계 문서](../handover/README.md), [시험 문서](../testing/README.md) | 현행 실행 모드와 맞추고 합의한 현장 조건으로 실제 재시험 — P6/P7 |

낙상 과거 저장 결과 예: 공개 320개 영상에서 Baseline recall 91.67% / FP 2건,
Candidate recall 97.50% / FP 7건이다. 현장 원본 23개는 Baseline recall 62.50%,
Candidate 56.25%로 서로 다른 평가 집합이다. 새 추론 성적이나 최종 현장 합격 수치가 아니다.

### EdgeX 진행 현황 — 2026-09-21 보충

이번 확인은 코드·기존 UAT 대조 및 `docker ps` 읽기 조회다. [확인 범위·관측 결과](evidence/edgex_status_review_20260921.json). 서비스 재시작·설정 변경·장치 명령은 실행하지 않았다.

| 영역 | 확보한 구현/근거 | 남은 작업 |
|---|---|---|
| EdgeX 기반 서비스 | Core Data·Metadata·Command·MQTT broker·rules engine 컨테이너 실행 중 확인 | 실행 상태와 API/장치 전체 정상 판정은 구분 |
| 센서 입력 | TLV 파서, Core Data 전달용 `EdgeXForwarder`·outbox와 sensor rule bridge 구현. parser·rule bridge healthy | 타 부서 최신 패킷·단위/주기, 실제 센서→EdgeX→장치/관제 대조(P4) |
| AI/명령 연결 | AI 어댑터·Action Layer·명령 결과 수집기 healthy. 장치 ID별 명령/결과 코드 존재 | 우선순위·후순위 보존·발행/실물 성공 구분·장치별 실패 격리(P1.3/P3) |
| 전광판 | 09-17 실제 호출 로그·결과 DB acknowledged, 약 1.03초. Device Service 현재 healthy | 육안/사진·관제 ID 대조·실패 복구(P3.4 VERIFY) |
| 스피커 | Device Service 현재 healthy, 모형 계약 존재 | 최근 기존 점검 timeout 후 실연결·음성·중복/중단 시험(P3.5) |
| 사이렌 | Device Service 현재 healthy, 모형 계약 존재 | 사용자 확인상 실물 없음(P3.6 BLOCKED) |

코드상 센서는 TLV 파서에서 Core Data로 전달하는 경로가 있으며, 모든 센서의 원본 접수가 Device Service를 거치도록 통일됐다고 확인한 것은 아니다. 출력은 Action Layer→EdgeX MQTT 명령→장치별 Device Service→결과 수집 구조다. 명령 발행 성공과 실제 출력 성공은 I03/P3.3에서 구분해 검증한다.

현재 P2 저장/관제 검증은 후순위 데이터가 장치 실행과 별도로 보존·전달될 기반을 다지는 작업이다. 혼합 300건 재시작 복구에 이어 P1.3 우선순위·만료 계약도 완료했다. P2.4 실제 부하/보관 검증을 추적하고, P3 선행 조건 충족 후 장치별 대기/격리 → 전광판·스피커 실물 시험으로 기존 EdgeX 경로에 연결한다. 이번 현재 정책 회귀 87 PASS와 이전 저장/관제 108 PASS를 새로운 EdgeX 실물 합격 근거로 사용하지 않는다. I06·I15와 새 위험 순서 차이 I23은 OPEN이다. 현재 P5.1 입력 계약과 [GIE 메타 계측](MODEL_INFERENCE_METRICS.md)을 로컬 완료했다. 신규 28개 포함 217 PASS/5 SKIP이며 운영 반영/실제 캐시·Python 모델 계측은 남는다.

## 2. P0.1 기준선 확인 결과 — DONE

### 코드·실행 상태

- Git HEAD: `1c76eac0d6d4b90d1de8890eeac40d2085347f1c`, 브랜치
  `260907-260911-9월2주차`. 마지막 커밋은 09-09이며 최근 변경 전체가 커밋된 상태는 아니다.
- 09:40 KST 스냅샷: staged 185개, unstaged tracked 37개, untracked 905개,
  고유 경로 1,112개. 두 tracked 수에는 양쪽에서 바뀐 파일이 중복될 수 있다.
  미추적 디렉터리 안 파일까지 센 수치이며 이후 문서 추가로 현재 수는 달라질 수 있다.
- 실행 컨테이너 32개: health 구성 17개 모두 healthy, 나머지 15개는 health 미구성.
  실행 상태를 물리 장치 가동이나 모델 정확도 합격으로 해석하지 않는다.
- Action Layer: `EDGEX_COMMAND_MODE=edgex`, 허용 장치 `speaker,signboard,siren`.
  사이렌은 **사용자 확인상 미보유**이고 호스트 설정도 비어 있다. 허용 목록 포함은
  연결·실물 출력 증거가 아니다. 이번에는 라우팅 설정을 변경하지 않았다.
- AI: `FALLDATA_AUX_MODE=shadow`, near-miss 승격·compare veto false,
  `APPEARANCE_BACKEND=hsv`, `DS_CROP_POSE_ENABLED=0`. 환경값만으로 SGIE를 포함한
  실제 전체 모델 실행·품질을 확정하지 않는다.
- AI 시작: 09-17 08:45:46 KST. 계측·프로세서 파일 수정: 같은 날 약 13:19 KST.
  `/app/src`·`/app/scripts`는 bind mount다. 디스크의 수정 코드가 보이는 것과
  이미 로딩된 Python 코드가 바뀐 것은 다르므로 **계측 교정의 운영 반영은 미확인**이다.
- parser와 Device Service는 이미지에 든 소스를 사용한다. 로컬 파일 변경이 바로
  해당 서비스 실행 코드에 반영된다고 판단하지 않는다.

원자료: [Git·Docker·선별 설정 스냅샷](evidence/baseline_20260918.json),
[모델 해시·시험 집계](evidence/baseline_verification_20260918.json).
비밀값·RTSP URL·환경 파일 본문은 수집하지 않았다.

### 테스트 기준선

| 시점·범위 | 결과 | 판단 |
|---|---|---|
| 09-17 전체 CPU 테스트 기록 | 1,805 passed / 1 failed / 75 skipped | 과거 기록. 영상 테스트 제외·환경 제한 포함 |
| 09-18 선별 9개 파일 재확인 | **100 passed / 1 failed / 0 skipped**, 29.98초 | 기존 실패 1개 재현, 선택 범위 내 신규 실패 없음 |

현재 실패: `tests/test_check_compose_runtime_assumptions.py::test_edgex_mode_rollback_contract_accepts_direct_base_and_signboard_uat`.
정책·Compose·검사·테스트의 계약 정합성 문제로 I06에 남긴다. 통과를 위해 임의로
운영 모드를 바꾸거나 실패 검사를 삭제하지 않았다.

실행 명령:

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_canonical_event.py tests/test_http_outbox.py tests/test_mqtt_outbox.py \
  tests/test_sensor_rule_bridge.py tests/test_edgex_device_service_contracts.py \
  tests/test_deepstream_primary_metrics.py tests/test_check_compose_runtime_assumptions.py \
  parser-python/tests/test_tlv_parser.py parser-python/tests/test_edgex_outbox.py \
  -q -o addopts=--tb=short --junitxml=docs/planning/evidence/baseline_tests_20260918.xml
```

[JUnit 결과](evidence/baseline_tests_20260918.xml). 이후 재실행은 새로운 날짜/실행 ID로
저장해 이 기준선 결과를 덮어쓰지 않는다. 이번은 문서 변경이므로 앱 빌드·전체 GPU
시험·실물 출력·배포·복구 실행은 하지 않았다.

### 복구 기준과 보존 범위

운영 컨테이너별 image ID·시작 시각·mount, 주요 소스/설정 11개와 로컬 모델 6개의
SHA-256을 보존했다. 현행 엔진·모델·설정은 교체하지 않았다.
[백업·복구 런북](../handover/BACKUP_RESTORE_RUNBOOK.md)과 원자료의 DB volume/mount
위치를 복구 조사 시작점으로 사용한다.

스냅샷은 **실제 백업이나 복구 리허설이 아니다**. bind mount 이전 코드와 실제
DB 백업 세트까지 확보된 것으로 주장하지 않는다. 운영 적용 전에 정확한 적용
파일의 이전 사본·DB 백업·현행 모드에 맞는 복구 명령을 확보하고 별도 검증한다.
P0.1 DONE은 실행/로컬 차이·기존 실패·복구 위치와 한계를 기록했다는 뜻이다.

## 3. 미해결 이슈 대장

`OPEN` 수정/검증 필요, `BLOCKED` 외부 조건 대기, `CLOSED` 해결·재검증 증거 확보를
뜻한다. WBS 상태는 WBS 표에서만 관리한다. 이슈를 닫을 때 해결일·검증 링크를 추가한다.

| ID | 상태 | 내용·근거 | 대응 WBS / 종료 조건 |
|---|---|---|---|
| I01 | FIXED_LOCAL | 메모리 배치 보존에 이어 새 설정 경로의 SQLite 원본·PostgreSQL 처리 ID/입력별 commit·재시작/중복 방지 구현. [격리 DB 포함 57 PASS](SENSOR_RECOVERY.md) | P2.1 DONE(새 경로 로컬). 운영 미적용; 활성화 전 기존 큐 배수/공유 경로·이미지 확인. 미설정 시 기존 메모리 경로 |
| I02 | OPEN | 후순위 출력 스킵 및 장치 실행 이후 전달/저장 경로 — 기존 감사 | P1.3/P3.1/P3.2: 후순위 기록·만료·장치별 대기, 관제 전달 격리 |
| I03 | OPEN | HTTP 대상·동시 중복·발행 성공/실물 성공·부분 실패 집계 결함 — 기존 감사 | P3.3: 장치별 ID·상태·오류 회귀, 응답 유실 처리 |
| I04 | OPEN | 전체 AI 4카메라×매 프레임 30FPS 미검증. [P0.3 원자료 재계산·검증 경로](PERFORMANCE_DECISION.md) 완료, 성능 미달/변동 위험 유지 | P0.3/P5.1/P6.2: 운영 반영 범위·조건 통제·전체 모델별 계측, 실물 검증 |
| I05 | BLOCKED | 사용자 확인: 사이렌 없음, CCTV 1~2대 보유 | P3.6/P7.1: 사이렌 및 총 4대 실물 접근 가능일 확보. 모형·재생 결과와 구분 |
| I06 | OPEN | Direct 중심 과거 정책과 현재 EdgeX 실행·검사 불일치. 기존 실패 1개 현재 재현 | P1.3 및 배포 검증: 현행/복구 계약을 정리하고 대응 테스트 통과 |
| I07 | OPEN / 10-01 v0.92 격리 전환 완료 | 후보 신규 이벤트31건/외형28건 포함705517행 보존·중복 재접수 변경0, 복귀 후32/28건 추가·DB3개ok. 종료6.37/3.49초 | 로컬 복구/전환 근거 재사용. 실제 운영 반영과 I32/I34/I37/I38 필수 조건 잔여. [근거](VIDEO_INFERENCE_CONTRACT.md#transition-rollback-v092) |
| I08 | OPEN | 모델 입력 시간창·캐시·합격 검사 사각지대와 현장 품질 차이 — 모델 감사 | P5.2~P5.5: 과제별 품질·입력 계약과 재현 가능한 평가 통과 |
| I09 | OPEN | 기존 감사의 제어 API 노출·환경 백업 제외 누락·이중 전광판 제어 위험 | P1.3/P3.2/P6.4: 작은 별도 수정 단위 편성, 배포/복구 검토·사전 시험. 이번 재현/해결은 안 함 |
| I10 | BLOCKED | 실제 관제 대상·센서 수/주기·현장 지연/품질 상한 미확정. 노트북 POC의 TLS 인증·합성 2건 저장 ACK는 실측 완료, 실제 관제 화면·수치 합격은 대기 | P0.2/D04/D06: [POC 범위·시험표](POC_SCOPE.md)와 [원격 ACK 실측](MONITORING_REMOTE.md). 임시 수신기와 실제 관제 적용·수치 합격은 구분 |
| I11 | FIXED_LOCAL | 부분 canonical payload 보강 시 decoded/device/gateway/raw/event 정보 덮어쓰기. 09-18 재현 3건 후 최소 수정, 관련 32 PASS | P1.1: [계약·회귀 근거](COMMON_DATA_CONTRACT.md). 소스·로컬 검증 완료, 운영 로딩/실제 센서·관제 검증은 별도 |
| I12 | OPEN / 일부 로컬 수정 | 외부 MQTT/직접 TLV bytes 선저장·오류 격리·수동 ACK에 이어 영상 선택 경로의 오류 원본/변환 사유 보관·epoch 0 보존을 검증. [영상 근거](VIDEO_INGEST_RECOVERY.md) | 기존 비영속 경로·운영 적용·현장 오류/프로필 시험은 남음 |
| I13 | OPEN / 내부 모형 완료 | 입력/관제 ACK·대조·계측에 이어 [서비스·실제 로컬 MQTT·전체 지연](MONITORING_SERVICE.md)을 검증 | 실제 관제 인증/화면·운영 적용·장기 장애/용량은 남음. 로컬 최소 흐름과 외부 종단 보장 구분 |
| I14 | OPEN / 공통 관측 보완 | 직접 TLV 해석 원본→공통 관측에서는 0/결측/오류를 보존하도록 연결·검증. 기존 _get_float 및 PG 수치 컬럼 보정은 그대로 | P4.1: 기존 컬럼/이행 영향·실물 규격 검토. 공통 관측 보완만으로 기존 저장 문제를 닫지 않음 |
| I15 | OPEN | 로컬 EdgeXForwarder의 Core Data 프로필 매핑에 t34954 없음. 해당 프로필은 _build_edgex_event에서 None 반환(09-18 코드 확인) | P4.1/P4.3: 실제 온습도 프로필·단위 확인, 등록/전송 계약과 재생·실물 검증. MQTT/Alert 전달과 구분 |
| I16 | OPEN | 외형 분석의 작은 crop 처리와 매칭/로그 생성에서 헬멧·가방의 판정 불가/누락을 False로 대체(09-18 코드 확인). 관측 변환으로 원래 품질을 복원할 수 없음 | P5.3/P5.4/P1.3: 생산자 능력·가시성·판정 품질 보존, 실제 미착용과 구분하는 경계/경보 회귀 |
| I17 | OPEN / 새 선택 경로 수정 | [P2.2](TLV_RECORD_DELIVERY.md)·관제 ACK 복구·[로컬 서비스 주기 실행/재시작](MONITORING_SERVICE.md) 검증 | 기존 비영속 경로는 이전 동작. 운영 서비스 배포/연결·실제 관제·실부하 검증은 남음 |
| I18 | OPEN | Action _EventRepo.save/record_command는 sqlite3.Error를 로그 처리하고 반환(09-18 코드 확인). 호출자는 반환만으로 commit 성공 여부를 구분할 수 없음 | P1.2/P2.2/P3.3: 저장 책임/성공 결과 계약과 DB 장애 회귀. 이번에는 해당 실패를 주입하지 않음 |
| I19 | OPEN | 별도 TLV 재수신은 접수별 보존. 같은 접수 ID의 PG·공통 관측·기록 작업 복구는 1건 유지. 안정적인 생산자 메시지 ID/부팅·시퀀스 규격은 미확정 | D02/P4.1/P1.3: 값 기반 임의 제거 금지, 생산자 재수신 식별·경보 정책 확인. 기존 설계의 접수 ID 기준으로 P2.2 로컬 종료. [시험·인계](TLV_RECORD_DELIVERY.md) |
| I20 | OPEN / 09-30 v0.75 현재 GPU 후보 재검증 | 25초 입력 단절 후8.065초 재개·전체 재시작0. 재개 이후 preview208건은 새 세대3만 관측 | 다른 ERROR/EOS·기존 외부 수신 연결 유지·느린 저장/장기/운영 잔여. 결합 누락1건은 I32로 추적. [근거](VIDEO_INFERENCE_CONTRACT.md#p51-closeout-v075) |
| I21 | OPEN / 연결 회복·저장 ACK 실측 | 16:19 실제 노트북 TLS 인증·합성 2건 저장, 16:32 시험 수신기 중지/재기동 후 총 4건 ACK 확인. [복구 실측](evidence/monitoring_laptop_recovery_20260921.json)·[이전 근거](evidence/windows_lan_intermittent_20260921.json) | 정상 LAN 구간에서 수신기 중지/복구 성공. 간헐 AP/이웃 실패의 원인·장시간 지속성은 미확정이며 AP 진단·설정 변경은 추가 실행하지 않았다 |
| I22 | OPEN / 빈 조회·영상 BUSY·혼합 적체 내부 검증 | 빈 worker 40회 쓰기 80→0회([이전 197 PASS](MONITORING_CONTENTION.md)). 영상 T1 BUSY 1회 재시도·활성 혼합 48건 [200 PASS](VIDEO_INTAKE_CONTENTION.md). 후속 혼합 300건 적체·두 차례 실제 종료 복구·저장량 [108 PASS](MONITORING_BACKLOG_RESTART.md) | P2.4: 실제 입력률·영상 지연·장기 보관량 검증 잔여. 순차 적체/제어 시계 시험은 활성 쓰기 성능 합격을 대신하지 않음. 소진된 최초 영상 실패는 DB에 없으므로 무손실·종료 후 자동 복구 보장은 아님 |
| I23 | OPEN / 09-22 재현 | 현재 risk_base+level 숫자 구간이 겹침. head warning/2 활성 슬롯 뒤 zone_dwelling critical/3이 False로 스킵. [실측](evidence/action_policy_priority_gap_20260922.json) | P1.3 계약은 위험 등급을 먼저 비교하도록 확정. P3.1 적용 전 정렬/슬롯 비교·전체 유형/별칭·모든 사용처 회귀로 해결. 현재 정책 87 PASS가 이 결함 해결을 뜻하지 않음 |
| I24 | FIXED_LOCAL / 09-22 | 서로 다른 추적 ID별 자세 조회·후보 연결·캐시/확정 횟수 분리, 세대 변경 시 이전 결과 차단. [수정·291 PASS/5 SKIP](POSE_TRACK_ISOLATION.md) | P5.1/P5.2: 실제 인물 ID 연속성·누락/PTS·모든 재연결 경계·운영 반영/모델 품질은 잔여. 실제 오탐 감소·물리 오작동 해결로 확대하지 않음 |
| I25 | FIXED_LOCAL / 09-28 | inline Pose-RF에 NaN·무한대·깨진 배열·잘못된 크기가 유입되거나 같은 배치의 정상 입력 처리를 중단함. 유효성 검사와 잘못된 인물의 scope 연결 차단. [197 PASS/2 SKIP](VIDEO_INFERENCE_CONTRACT.md) | P5.1/P5.2: 운영 반영·저신뢰 관절/시간 간격 기준·실영상 영향은 잔여. 모델 정확도 합격이나 모든 입력 품질 해결을 뜻하지 않음 |
| I26 | FIXED_LOCAL / 09-28 | 의상 로그/조건 매칭 쿨다운이 camera ID 없이 track ID를 공유해 다른 카메라의 기록·알림을 억제할 수 있음. 카메라·종류별 key와 카메라별 초기화 보완. [211 PASS/2 SKIP 묶음](evidence/context_generation_20260928.xml) | P5.4/P6.2: 실제 모델·다채널 부하·운영 반영 잔여 |
| I27 | FIXED_LOCAL / 09-28 | camera/frame/source PTS·세대·고유 native 토큰으로 영상/메타 결합. 최신 이미지 조회 제거, 오래된 주변 객체 메타·재연결 전 대기 샘플 제외. 실제 후처리 128건 식별/해시 일치·235 PASS/2 SKIP | 운영 반영·실제 얼굴/의상 품질은 후속. 다중 preview는 I28. [근거](VIDEO_INFERENCE_CONTRACT.md) |
| I28 | OPEN / 격리 재현 09-28 | batch2 + CPU preview에서 nvvideoconvert buffer transform failed, preview 0건. 단일 입력 최종 시험은 정상 수신 | P5.1 다중 입력 경계/P6.2: source별 preview 분리·전체 모델 다채널 검증. 기존 preview 비활성 2입력 결과와 구분 |
| I29 | FIXED_LOCAL / 09-28 | 일반 Pose의 ROI probe 미실행·헬멧 단독 SGIE 분기 queue 누락·tracker 탐지 완료 표시 누락. 로컬 보완 후 동일 해상도에서 의상/헬멧 tensor 각 266프레임 보존, 201 PASS/2 SKIP | P5.1: 운영 미반영. 원본/처리 해상도 차이는 I30, 모델 정확도·판정 연결은 P5.2~P5.5. [근거](MODEL_INFERENCE_METRICS.md) |
| I30 | FIXED_LOCAL / 09-28 | YOLO·ROI·Crop-Pose·이벤트의 좌표 기준을 mux로 통일, 경계 밖 ROI 제한. 230 PASS/2 SKIP·4K→1080p/패딩 변환 오류 0 | P5.1: 좌표/ROI 범위 로컬 완료. 실제 영상 내용 대응은 I32, 운영 미반영. [근거](evidence/coordinates_20260928.json) |
| I31 | FIXED_LOCAL / 09-28 | 정확한 track_id_text 저장·API 문자열 응답, 이전 DB 읽기·재시도 보완. 67 PASS·실영상 기록 13건 일치·대형 ID 3종 별도 재현 통과 | P5.4/저장 경계 로컬 완료. 운영 마이그레이션·관제 확인 잔여. [근거](APPEARANCE_TRACK_ID.md) |
| I32 | OPEN / 10-02 v0.97 시작 지연 우회 검증 | GPU OSD 첫 버퍼10.87초, CPU OSD17.64ms. CPU 경로 등록→도착0.11초·후보 단절 복구 시험1450개 정확 대응/누락0 | 기존 CPU OSD 후보 설정을 사전 검사로 고정·85 PASS. 과거13건 원인 소급 확정/운영 반영은 미완료. [근거](MODEL_INFERENCE_METRICS.md#startup-delay-v097) |
| I33 | FIXED_LOCAL / 09-29 v0.58 | 하드웨어 인코더 insert-vui=true 보완. 15/29.97fps 및 재기동 후보 전후6파일 모두 FPS30/1·SPS tick1/scale60·디코드 오류0. 167 PASS/2 SKIP | 양수 videorate/FPS caps와 함께 적용. 명목 FPS와 실처리 FPS 구분, 브라우저/기존 연결·운영 반영은 잔여. [근거](TRACKER_VALIDATION.md#v058--cpu-osd-단절-복구와-vui-보완) |
| I35 | FIXED_LOCAL / 09-29 파일→RTSP 시작·종료 대기 | 수정 전 upstream4/후처리0·NULL 전환 정지 재현. preview async=False·종료 FLUSH_START로 해소 | 실제 CLI30프레임 메모리 오류0·게시/종료0·182 PASS/2 SKIP. I34와 구분·운영 미반영. [근거](TRACKER_VALIDATION.md#v065--파일rtsp-시작종료-대기-수정) |
| I36 | FIXED_LOCAL / 09-29 후처리 큐 포화 | GPU 버퍼/스트림 개선·인물별 최대2Hz·미완료 중복 차단. 실CCTV180초218접수/완료·큐 폐기/실패0, 대기 평균46.6ms/P95 363.1ms·308 PASS/2 SKIP | 단일 입력 후보 검증. 분석 주기/평활화 지연·군중/다중/장시간·운영 반영 잔여. CUDA700 원인 해결과 구분. [근거](TRACKER_VALIDATION.md#v066--후처리-큐-병목-완화) |
| I34 | OPEN / 10-01 v0.86 할당 경계·수명 검사 | 원 보존본/고정 후보 각각60프레임·종료0·최종 memcheck 오류0, 최초 CUDA700 미재현 | padding32·stream-ordered 검사를 추가했으나 원인 제거 근거 아님. 자식 TCN은 검사 제외, 진단 전용 재연결0. 새 조건 없는 반복은 중단하고 I32/I38 일반 실행 대조로 이동. [근거](TRACKER_VALIDATION.md#cuda-bounds-v086) |
| I37 | OPEN / 09-30 v0.71 표본 선택 기준 보존 | TCN 입력 크기·RF 출처/표본/시간 불일치 점검 | 고유 표본 정책/상한 수집→학습 보존·205 PASS. 원본 복원 못함(검색 범위/권한 제한 명시). 점수 기준 이력·정확도/운영 잔여. [점검](MODEL_INFERENCE_METRICS.md#rf-sampling-contract-v071) |
| I38 | OPEN / 10-02 v0.97 재사용 TCN 관측 | 같은 후보 통합 시험18회 반환: Pose30 유효17·최초 Pose1 부족1, timeout0. 준비 후1.24~2.55초 | 별도 모델 관찰의 미반환 요청을 성공으로 세지 않음. 과거 timeout 원인·운영 반영은 잔여, 낙상 정확도 합격 아님. [근거](MODEL_INFERENCE_METRICS.md#startup-delay-v097) |
| I39 | FIXED_LOCAL / 09-30 v0.78 종료 SIGABRT | 보조 영상 쓰기 중 daemon 스레드를 2초 뒤 방치해 Python 종료와 충돌. 기존 후보2회 재현·OpenCV/native 스택 확인 | 실제 작업 완료 join·재사용 모델 자식 회수. 256 PASS/2 SKIP·인코더 포함/제외/재연결 중 종료/RTSP 재시험. SIGINT 범위·운영 미반영·원 CUDA700과 구분. [근거](TRACKER_VALIDATION.md#candidate-shutdown-v078) |
| I40 | OPEN / 10-02 v0.101 | 실제 AI 후처리·소유권5조건 종료2.18~14.65초(SIGTERM 이후), CPU3565개 독립 소유·중지20요청 반환 | flush 제외 대조에서 동일 MainThread stop() 중첩1건. 기존 인코더 native 멈춤과 인과 미확정·재진입 영향 재현 후속·운영 미배포. [근거](TRACKER_VALIDATION.md#buffer-ownership-v101) |
| I41 | FIXED_LOCAL / 10-01 v0.88 | 정기 백업에서 MQTT/HTTP 재전송·이벤트 대기·명령 결과 DB 누락, 과거 백업 보존 옵션/권한 미보완 | 표준8DB 목록·읽기 전용/검사 후 완료·700/600·삭제 끄기 보완, 초기3실패→5 PASS·실제8DB 복원. 다음 정기 실행은 설치된 경로의 수정본 사용. [근거](VIDEO_INFERENCE_CONTRACT.md#rollback-preparation-v088) |
| I42 | FIXED_LOCAL / 10-01 v0.90 | 읽기 전용 모델 루트에 trt_cache 연결 디렉터리가 없어 OCI 기동 실패. v0.89 사전 검사는 이를 놓침 | 빈 연결 지점 준비·사전 검사/회귀 추가 후 전체 AI 기동 확인. 후보 종료 유예60초도 명시, 관련46 PASS. 운영 미배포. [근거](VIDEO_INFERENCE_CONTRACT.md#profile-start-v090) |
| I43 | VERIFY / 10-02 운영 복구·상태 검사 반영 | X11 인증 복구 후에도 frames=0이었으나 같은 이미지 AI 재시작 후 진행 재개. 처리 정지 시503·X11 인증 재시도/SDK 검사 반영. 187초7표본·프레임6942→12535·드롭/이벤트 실패0 | 재부팅·세션 변경·장시간/다중 카메라 미검증. 과거 재시도 후 정지의 native 내부 원인은 미확정. 기본 입력 복구를 기준으로 I32/I38 조사 재개. [근거](evidence/environment_20261002.json) |
| I44 | FIXED / 10-02 호스트 의존성 복구 | 사용자 설치로 RTSP 서버 라이브러리와 GStreamer 플러그인2패키지 보완. 호스트/컨테이너 native SDK7.1·플러그인 각6항목 PASS | 설치 도구(기본 모의 실행)·실제 로드/버전 검사·자동 시작 전후 검사 추가. SDK 업그레이드/패키지 제거0. 새 장비 설치 재현은 I07 후속. [근거](evidence/environment_20261002.json) |
| I45 | OPEN / 10-02 전체 회귀에서 발견 | DB 동시 쓰기 시험의 영상 최초 저장 경로에서 SQLite database is locked 1회. 단독 재시험1 PASS, 관련4소스 해시는 작업 전과 동일 | 전체 회귀2717 PASS/52 SKIP/1 FAIL로 보존. 간헐 경합 조건과 재시도 소진 원인 확인 후 수정 여부 판정; 운영 DB 장애로 단정하지 않음. [근거](evidence/environment_20261002.json) v0.96 전체2737 PASS/52 SKIP에서 통과했으나 간헐 경합 원인 미해결로 OPEN 유지. 10-02 Git 보존 검증: 커밋 사본·외부 의존성 연결·실제 SSD 임시DB에서도 전체2737 PASS/52 SKIP/1 FAIL로 재관측. eMMC만의 문제로 단정하지 않으며 OPEN 유지. |
| I46 | OPEN / 10-02 배포 구성 대조 | Docker COPY/Compose 해석 통과, 기존 구성 검사14중3실패. 현재 실행 RF true/yolo_pose_rf와 재생성 설정 차이는 .env.jetson 명시로 보완. 외형 모델 경로는 현재 컨테이너에도 파일 없음 | SGIE 경로와 별도로 외형 경로 사용 여부·유효 모델을 확인하고, 현행 모델/EdgeX 모드와 검사 기준을 정합화. 통과를 위해 모델/장치 모드를 임의 변경하지 않음. [근거](evidence/match_reason_20261002.json) |

FIXED_LOCAL은 로컬 코드·회귀 검증 완료이며 운영 적용/종단 합격과 구분한다.

스피커의 09-17 timeout은 **과거 관측**이다. 보유 사실은 이번 사용자 답변으로
확정됐지만 현재 연결 불가로 단정하지 않는다. 전광판은 앞선 timeout 이후 09-17
09:14 UAT 성공 기록이 있으므로 이전 연결 실패를 현재 상태로 사용하지 않는다.

## 4. 이번 최신화 이후

- P0.1 기준선과 P0.3 성능 조기 판단을 완료했다. P0.2는 외부 조건 대기다.
- **P1.1 DONE(계약 산출물):** 공통/관측·진입점·실패/재처리 규칙과 현행 차이를 확정했다.
  앞선 센서/영상 회귀 **147 PASS**와 이번 **9개 경계 재현 중 정상 대조 2개 충족/7개 미충족**을
  구분한다. [계약·완료 판정](INGEST_FAILURE_CONTRACT.md). 무손실·오류 보관 구현 합격은 아니다.
- **P1.2 DONE:** [저장 설계](STORAGE_DESIGN.md)에 공통 SQLite·쓰기 책임·트랜잭션·ID/조회·
  기존 데이터 이행/복구를 확정하고 임시 DB 모형으로 검증했다. 운영 스키마 이행은 미실행이다.
- **P2.1 DONE(로컬):** [센서 저장 복구](SENSOR_RECOVERY.md)에 원본 접수·입력별 PostgreSQL
  commit·중복 방지·자식 프로세스 종료 복구 57 PASS와 적용 준비를 기록했다. `AIOT_INGEST_DB`
  설정 경로와 parser Dockerfile COPY를 추가했으며 운영 설정/DB·서비스는 변경하지 않았다.
- **P2.2 DONE(로컬):** [외부 MQTT 저장 1차 구현](EVENT_PERSISTENCE.md)에서 원본 선저장·이벤트/전달 대기
  원자성·강제 종료 복구에 이어 [외부 수신 ACK/브로커 재전송](MQTT_ACK_RECOVERY.md)을 검증했다.
  [직접 TLV 원본/ACK·PG 복구](TLV_MQTT_RECOVERY.md)도 로컬 검증했다.
  [DeepStream/OpenCV 영상 접수·저장 복구](VIDEO_INGEST_RECOVERY.md)도 로컬 완료했다.
  [기록 목적지별 worker·lease·PUBACK·재시도](RECORD_DELIVERY.md)도 격리 검증했다.
  [직접 TLV 공통 관측·기록 전달](TLV_RECORD_DELIVERY.md)까지 연결해 490 PASS / 5 SKIP로 검증했다.
  센서 생산자 재수신 식별 I19는 미해결로 유지한다. 동일 접수 ID의 복구 기준으로 모듈을 닫았다.
  관제 최종 ACK는 P1.4/P2.3, 물리 경보 재실행 정책은 P1.3/P3 범위다.
- **P1.3 DONE(계약):** [장치별 우선순위·만료·중복/재실행·결과 집계](ACTION_POLICY_CONTRACT.md)를 확정했다.
  14개 인수 예제와 현재 코드 차이를 정리했다. 기존 회귀 87 PASS, 별도 슬롯 재현에서 I23 미충족 확인.
  앱·설정·출력 동작은 그대로이며 영속 큐·격리/중복 실행 구현은 P3, TTL 등 운영 수치는 D04에 남긴다.
- **P1.4 DONE(내부 계약):** [관제 전달·저장 ACK·조회·계측 계약](MONITORING_DELIVERY_CONTRACT.md)과
  임시 DB/기존 wire 예제 **55 PASS**. PUBACK 이후 저장 확인·고정 bytes/ID·충돌·재조회·누락 대조를 정의했다.
  앱/DB 스키마/운영 설정은 변경하지 않았다. I10/I13/I17은 미해결이며 실제 수신처 D06과 지연 상한 D04도 대기다.
- **P2.3 DONE(내부 모형):** [관제 수신 저장·ACK·재조회](MONITORING_RECEIVER.md)를 앱 2파일로 구현했다.
  신규 28개/인접 회귀 포함 **158 PASS**, 실제 commit 전/후 자식 프로세스 종료 복구를 확인했다.
  후속 [송신 상태·관제 ACK 영속화·조회/재전송](MONITORING_SENDER.md)도 구현하고
  실제 MQTT·종료 5경계를 격리 검증했다. 이후 [snapshot 대조·적체/구간 계측](MONITORING_RECONCILIATION.md)을
  신규 23개·회귀 포함 202 PASS로 검증했다. 별도 manifest DB만 추가하고 source/receiver schema는 유지했다.
  후속 [서비스·로컬 MQTT/조회·전체 지연](MONITORING_SERVICE.md)은 277 PASS/5 SKIP로 검증했다.
  5 SKIP은 기존 PostgreSQL 전용 시험이며 신규 시험은 모두 통과했다. 실제 관제/운영 반영과
  P2.4 장기 장애/용량은 남는다. 앞선 송신 store 호출 시 source schema v4가 되므로
  적용 전 모든 공통 DB writer의 호환 배포와 기존 worker 전환이 필요하다.
- **P5.1 IN_PROGRESS:** [실행 대조·입력/추론 계약](VIDEO_INFERENCE_CONTRACT.md) 첫 단위 완료.
  운영 두 metrics 표본에서 Primary 완료 지표 미노출 I07 확인, 인물 혼합 I24 재현, 기존 189 PASS/5 SKIP.
  후속 [GIE 메타 계측](MODEL_INFERENCE_METRICS.md) 로컬 구현, 신규 28개 포함 217 PASS/5 SKIP. [I24 인물별 시간창/캐시 분리](POSE_TRACK_ISOLATION.md)도 신규 20개 포함 291 PASS/5 SKIP로 FIXED_LOCAL. 실제 GPU·Python 모델 완료·운영 반영·입력 단절/품질 부족 검증이 남아 DONE으로 처리하지 않는다.
- **P2.4 BLOCKED(운영 기준 대기):** 기존 내부 검증을 보존하고 D02/D04/D06 합의 후 실제 부하/장시간 시험을 재개한다.
  [DB 잠금·혼합 입력 누적 복구](MONITORING_OUTAGE.md) 신규 4개·관련 회귀 **158 PASS**.
  두 시나리오 각각 24건의 ID/hash·저장 ACK 일치, 누락/논리 중복/잘못된 저장 ACK 0건.
  후속 [빈 관제 조회 쓰기 잠금 개선](MONITORING_CONTENTION.md)은 앱 1파일 수정·신규 9개·관련 **197 PASS**.
  빈 worker 40회 쓰기 80→0회, MQTT/영상 최초 저장 실패 경계를 확인했다. I22 잔여·용량·오래된 데이터는 남는다. 운영 DB·배포는 변경하지 않았다.
  [원격 ACK/노트북 패키지](MONITORING_REMOTE.md)는 회귀 73 PASS·TLS 통합/재시작 각 1 PASS,
  합성 센서/영상 2건 저장 ACK·중복·인증 거부를 격리 검증했다. 노트북 broker/receiver 기동은 사용자 확인,
  두 포트 TCP 접속도 직접 확인했다. [기동/접속 근거](evidence/monitoring_laptop_startup_20260921.json).
  이어서 16:19 KST Jetson client 전달·TLS 인증·두 장비 합성 2건의 저장 ACK 재조회/ID/hash 대조를 완료했다.
  [실제 LAN 근거](evidence/monitoring_laptop_delivery_20260921.json).
  실제 수신기 중지 중 기존 완료 2건+새 미확인 2건·ID/bytes/hash 보존·잘못된 ACK 0건도 확인했다.
  [중지 중 근거](evidence/monitoring_laptop_outage_20260921.json).
  재기동 후 기존 CLI로 송신을 재개해 총 4건 ACK를 대조했다. [복구 근거](evidence/monitoring_laptop_recovery_20260921.json).
  짧은 수신기 중지/재기동 단위는 완료, 장시간 장애·상시 운영·부하·용량은 남는다.
  후속 [격리 SQLite 용량 부족·보존·복구](MONITORING_CAPACITY.md)는 신규 6개 포함 **71 PASS**다.
  접수 실패·정규화/대기 rollback·수신 ACK 보류와 송신 ACK 기록 FULL 후 24건 재조회 완료/재발행 0건을 확인했다.
  원본 미접수 자동 복구는 구현하지 않았으며 I22는 유지한다. 오래된 데이터 표시·실제 부하/보관량 기준·파일시스템 고갈 시험은 남는다.
  후속 [센서값 나이·수신 당시 상태 조회](MONITORING_FRESHNESS.md)는 신규 30개 포함 **131 PASS**다.
  같은 스냅샷도 조회 시각에 따라 fresh→stale로 바뀌며 지연/역순 도착을 보존한다. 기본값은 기준/시계 미확인으로 unknown이다.
  내부 조회 단위 완료이며 노트북/화면 배포·실센서 유효시간·활성 쓰기/미접수 정책·실제 보관량 합격은 남는다.
  09-22 후속 [영상 최초 저장 잠금 재시도·활성 쓰기](VIDEO_INTAKE_CONTENTION.md)는 신규 14개 포함 **200 PASS / 0 SKIP**다.
  INSERT/COMMIT 첫 잠금 후 같은 원본/접수 ID로 1회 재시도, 두 시도 소진/FULL의 실패 경계를 확인했다.
  활성 혼합 48건의 저장 ACK·ID/hash 대조를 완료했다. 관제 실패 주기 2회 후 복구했으며 영상 시험 생산자 추가 재제출은 0회다.
  앱 2파일·신규 테스트 2파일, 공개 API/schema/wire/설정 변경·배포 없음. I22 운영 부하 및 P2.4 적체 누적/장시간·실센서/보관 기준은 미완료다.
  후속 [혼합 300건 적체·실제 프로세스 재시작](MONITORING_BACKLOG_RESTART.md)은 신규 1개 포함 **108 PASS / 0 SKIP**다.
  수신 37번째 commit 후/송신 51번째 ACK commit 후 종료, 새 프로세스로 300건 전체 ACK 복구·누락/논리 중복 0건을 확인했다.
  source 적체 4.29MiB→완료 4.89MiB·receiver 1.93MiB, 복구 구간 33.748초는 모형 통신/제어 시계 조건이다.
  원본·wire/ID/hash 보존, 물리 작업 150건 held·시도 0회, 앱 204파일 해시 동일. 운영·장시간 부하/보관 기준은 남긴다. 후속 P1.3 계약은 완료했으며 다음 개발은 P5.1이다.
- 기존 구현/시험이 있는 항목은 WBS에 E-ID와 잔여 작업을 연결했다. VERIFY는
  기존 증거의 재검증·보완 대상이며 선행 작업 전체가 완료됐다는 의미는 아니다.
- [POC 범위·시험표](POC_SCOPE.md)에서 보유 장비로 먼저 확인할 순서와 외부 조건을 관리한다.
- [사내 진행판](http://192.168.2.242:18081/)에 v0.44 전체 WBS·검증·이슈를 게시했다.
  전용 nginx 서버 HTTP 200·원본 대조·Firefox 화면 확인, 다른 PC 접속은 별도 미확인이다.
  [운영·갱신 절차](ONLINE_PROGRESS.md)를 따라 매 업무 종료 시 같은 주소를 갱신한다.
  FigJam은 호출 한도로 제목만 추가된 미완성 상태이며 [시각화 안내](figma_exports/README.md)에서 구분한다.

