# 사내 개발 진행판

갱신: 2026-10-02 KST · 계획 v0.101

**열람 주소: <http://192.168.2.141:18081/>**

같은 회사 네트워크의 Windows/노트북 브라우저에서 위 주소를 연다.
Jetson과 Docker가 실행 중이어야 하며, IP가 바뀌면 서버의 바인딩 주소와 안내를 함께 수정한다.
외부 인터넷 공개·포트 포워딩은 설정하지 않았다. 사내 주소로 접속하는 읽기 전용 문서 페이지이며
로그인 기능은 없다. 해당 네트워크에서 주소에 접근 가능한 사람이 볼 수 있는 진행 요약만 제공한다.

2026-09-30: Wi-Fi 주소가192.168.2.242에서192.168.2.141로 바뀌어 문서 서버 바인딩과 안내를 변경했다. 고정 IP를 설정한 것은 아니며 향후 주소 변경 시 재확인한다.

## 게시 내용과 확인 범위

v0.101: **실제 AI 후처리·버퍼 소유권5조건 확인, 종료 재진입 관측.** CPU 영상3565개 모두 독립 소유·인코더/디코더 중지20요청 반환, SIGTERM 이후2.18~14.65초 종료. flush 제외 대조에서 동일 스레드 stop() 중첩1건 확인·기존 native 멈춤과 인과 미확정. 제품206파일 불변·운영 정상, I40 OPEN·P5.1 IN_PROGRESS. [근거](evidence/buffer_ownership_20261002.json) · [분석](TRACKER_VALIDATION.md#buffer-ownership-v101).

v0.100: **실녹화·디코더·미리보기 종료6조건 확인.** 약89~90초씩 처리·인코딩15950개, 종료39~59ms·인코더/디코더 중지24요청 모두 반환. 미리보기 CPU 매핑5347개 확인. I40 원 실패는 미재현이며 해결 판정하지 않는다. 제품206파일 불변·운영 정상. [근거](evidence/encoder_decode_20261002.json) · [분석](TRACKER_VALIDATION.md#encoder-decode-v100).

v0.99: **Ponytail 검토·인코더 최소 재현6조건 확인.** 공통 생성/종료 함수와 기존 요청 계측을 재사용, 합성 입력·RTSP·단절·NVMM·mux6조건 종료21~35ms/중지 요청 모두 반환. 실제 AI 통합의90초 실패 조건과 달라 I40 해결 근거 아님. 진단 성공 판정에 실제 게시·단절 순서 확인 추가. 제품206파일 불변·운영 정상, I40 OPEN·P5.1 IN_PROGRESS. [근거](evidence/encoder_minimal_20261002.json) · [검토/시험](TRACKER_VALIDATION.md#encoder-minimal-v099).

v0.98: **종료 대기를 인코더 입력 STREAMOFF로 특정·flush 제외안 미채택.** 격리10회 중 정상 종료8회·30초 초과2회(서로 다른 조건, 실패율 아님). 실패 스택/동일 시점 주소와 SDK 바이너리 대조로 NvVideoEncAbortBuffer→NvOsSemaphoreWait 확인. 요청 계측에서도 인코더 입력 STREAMOFF만 반환되지 않음. 제품 소스·SDK·운영 변경 없음, I40 OPEN·P5.1 IN_PROGRESS. [실행 근거](evidence/native_shutdown_20261002.json) · [분석](TRACKER_VALIDATION.md#native-shutdown-v098).

v0.97: **첫 프레임 지연 구간 확인·새 후보 통합 검증, 종료 실패 잔여.** GPU OSD 첫 처리10.87초→CPU OSD17.64ms, 등록→CPU10.97→0.11초. 새 고정 후보 단절 복구8.16초·CPU1450개 정확 대응/누락0·TCN18회(유효17/Pose 부족1), 구성 회귀85 PASS. 추가 관찰 종료 실패2회(일반30초/진단60초 초과), 전체 NULL→NVIDIA 버퍼 풀/드라이버 대기 native 확보·I40 OPEN. 운영 미배포·P5.1 IN_PROGRESS. [실행 근거](evidence/startup_delay_20261002.json) · [분석](MODEL_INFERENCE_METRICS.md#startup-delay-v097).

게시 확인: 10-02 v0.101 HTML/PDF HTTP200·로컬 바이트 일치·WBS33행 상태 일치. 첫 PDF 화면 글자/배치 확인. 브라우저 조작/다른 PC 접속은 이번에 재시험하지 않았다. WBS33건 상태/완료9건은 유지하며 27.3%는 작업 수 기준이다.

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
v0.65: [파일→RTSP 시작·종료 대기 수정](TRACKER_VALIDATION.md#v065--파일rtsp-시작종료-대기-수정). I35 FIXED_LOCAL: preview 첫 버퍼 대기 해제·NULL 전환 전 FLUSH_START. 실제 앱 전체30프레임 CUDA 메모리 오류0·RTSP 게시·정상 종료. 182 PASS/2 SKIP. 최초 CUDA700 원인(I34)은 미확정·운영 미반영.

- WBS 33개, 상태 검색/필터, 완료·진행·대기 수, 이번 검증·주요 이슈·다음 작업.
- 현재 P5.1 IN_PROGRESS: v0.93 보존 기록 시간축 대조 완료. 누락 사유 구별을 위한 프레임별 관찰 시험 준비. I32/I38/I07/I34 등 OPEN·운영 미배포.
- P2.4는 실제 부하·보관/지연 기준 대기로 BLOCKED. P1.3은 계약 DONE이며 I23 수정은 후속 P3 범위다.
- 9/33 = 27.3%는 작업 수 기준이며 전체 구현률·잔여 공수 비율이 아니다.
- PDF 3쪽은 내려받아 읽는 보조 자료다. 현재 CJK 글꼴 PDF의 텍스트 추출은 정상 동작하지 않아
  검색/복사는 HTML 페이지를 사용한다. PDF 출력 화면은 별도 육안 확인했다.
- Jetson에서 자체 LAN 주소의 HTTP 200과 게시본을 확인했다. 다른 회사 PC에서의 실제 접속,
  AP 단말 간 통신, Jetson 재부팅 후 자동 시작은 아직 검증하지 않았다.
- v0.95는 기존 배치를 유지하고 HTML/PDF 문구를 갱신했다. 이번 확인은 HTTP 응답·게시 bytes·WBS 대조다.
  실제 브라우저 화면/필터 검증은 [v0.31 근거](evidence/pose_track_isolation_validation_20260922.json)를 유지하며 이번에 반복하지 않았다.
- 이번 검증은 [프레임별 누락 사유 관측](evidence/match_reason_20261002.json)이다. v0.94 전체 회귀의 I45는 해결하지 않았으며 새 제품 소스 변경이 없어 전체 회귀를 반복하지 않았다.

### v0.93 게시 대조 (2026-10-02)

- 보존243표본 시간축 대조·누락13건 중 TCN 반환 후8·CPU 제한 증가 없는 구간9.
- 후보 기록/복귀 코드4파일 해시 일치, 미등록/만료/용량 제거의 개별 사유는 미확정.
- WBS33행 ID/상태·HTML/PDF 게시 bytes 일치, PDF 첫 페이지 육안 확인·Ruff/문법 검사 통과.
- 제품 소스206파일 해시 불변. AI 회귀/GPU 시험·운영 재배포·다른 PC 브라우저 시험은 실행하지 않음.
- [분석·게시 검증 근거](evidence/match_timing_20261002.json). I32/I38/I34/I07 OPEN·P5.1 IN_PROGRESS 유지.

### v0.92 게시 대조 (2026-10-01)

- 후보 신규31건 포함705517행 보존·이전 API 중복 재접수 변경0·복귀 후 이벤트32건/외형28건 추가.
- 종료6.37/3.49초·영상3009프레임 디코드. 복귀 매칭 누락13·TCN Pose 부족은 별도 이슈.
- 운영 미배포·I07 OPEN·P5.1 IN_PROGRESS·DONE9/33. HTML/PDF/33행 대조는 `evidence/transition_rollback_20261001.json` publication 참조.

### v0.91 게시 대조 (2026-10-01)

- 복구본·암호화 설정·이미지 내보내기/재적재 및 이전 구성 격리 기동 확인.
- 기존705398행 보존·대기32건/외형28건 추가·SIGTERM2.59초 종료·영상1435프레임 디코드.
- 첫 기동 timeout·TCN timeout1·매칭 누락17·최대 영상 간격1.936초는 해결 미판정.
- 운영 미배포·I07 OPEN·P5.1 IN_PROGRESS·DONE9/33. HTML/PDF/33행 대조는 `evidence/restore_bundle_20261001.json` publication 참조.

### v0.90 게시 대조 (2026-10-01)

- 후보 전체 격리 기동·I42 보완·46 PASS, 누적 대기82건/외형73건·DB 무결성 정상.
- SIGINT25.12초/SIGTERM4.86초 정상. 영상2개5015프레임 원 시간기준 디코드 오류0.
- 운영 미배포·I07 OPEN·P5.1 IN_PROGRESS·DONE9/33. 외부 의존성/비밀설정 복구는 후속.
- HTML/PDF 게시/행 대조는 `evidence/profile_start_20261001.json` publication 참조.

### v0.89 게시 대조 (2026-10-01)

- 후보 프로필·44 PASS·합성 대기2건 재생성 보존, 이미지359/모델8 해시 일치.
- 운영 미배포·I07 OPEN·P5.1 IN_PROGRESS·DONE9/33. 전체 통합 기동/복구는 후속.
- HTML/PDF HTTP·33행 상태 대조 결과는 `evidence/candidate_profile_20261001.json` publication에 기록한다.

### v0.88 게시 대조 (2026-10-01)

- 재전송 백업 누락 I41 보완·5 PASS, 실제8DB·1,717,566행/스키마/상태 복원 대조.
- [실행·보존 해시·한계·게시 대조](evidence/rollback_preparation_20261001.json).
- 정기 백업 도구는 보완했으며 AI 재배포·운영 DB 덮어쓰기·명령 재전송 없음.
- I07 OPEN·P5.1 IN_PROGRESS·DONE9/33 유지. HTTP/행 대조 결과는 JSON publication에 기록한다.
- 브라우저 조작·다른 PC 접속은 이번에 재시험하지 않는다.

### v0.87 게시 대조 (2026-10-01)

- 일반 실행 매칭·TCN 대조, 복구8.23초·영상2파일 정상·외형 DB54/56행 무결성ok.
- [실행·소스 해시·관측 한계·게시 대조](evidence/match_tcn_20261001.json).
- 제품 추론 코드/운영 배포 변경 없음. 과거 I32/I38 원인·I34 OPEN, P5.1 IN_PROGRESS·DONE9/33 유지.
- HTML/PDF 게시 재조회 및 WBS 행 대조 결과는 위 JSON의 publication에 기록한다.
- 브라우저 조작·다른 PC 접속은 이번에 재시험하지 않는다.

### v0.86 게시 대조 (2026-10-01)

- WBS33행·HTML/PDF 게시 bytes 대조 및 첫 PNG 육안 확인은 [근거](evidence/cuda_bounds_20261001.json)에 기록한다.
- CUDA 근본 원인 미확정·I34 OPEN, 검사 중 TCN timeout 별도 표시·운영 미배포.
- 브라우저 조작과 다른 회사 PC 접속은 이번에 재시험하지 않았다.

### v0.85 게시 대조 (2026-10-01)

- I40 FIXED_LOCAL·운영 미배포, I34/I32/I07 잔여·WBS33행 상태 유지.
- HTTP 게시 bytes·WBS 대조와 PNG 육안 확인은 [검증 근거](evidence/rtsp_shutdown_fix_20261001.json)에 기록한다.
- 브라우저 조작/다른 회사 PC 접속은 이번에 재시험하지 않았다.

### v0.84 게시 대조 (2026-10-01)

- 정상 실행 통과와 출력 단절 실패·패치 미해결을 함께 게시한다.
- [실행/게시 근거](evidence/output_shutdown_20261001.json): WBS33행 일치·HTML/PDF 게시 bytes 일치·PNG 첫 페이지 확인. 브라우저 조작/다른 PC는 재시험하지 않았다.
- 운영 배포 없음·I40/I34/I32/I07 잔여·P5.1 IN_PROGRESS·DONE9/33 유지.

### v0.83 게시 대조 (2026-10-01)

- 종료 순서 최소 수정·46 PASS, 실제 후보 정상/입력 단절 종료 확인.
- [실행 근거와 게시 대조](evidence/shutdown_recovery_20261001.json).
- I40 추가 회귀·I34/I32/I07 잔여. 운영 배포 없음·P5.1 IN_PROGRESS·DONE9/33 유지.

### v0.82 게시 대조 (2026-09-30)

- HTML 22,704 bytes·PDF 126,102 bytes, 사내 게시본과 로컬 생성본 일치·WBS 33행 상태 일치.
- [종료 단계·운영 조회와 게시 SHA-256](evidence/shutdown_stages_20260930.json).
- 종료 단계 시간/Python 스택 확보, 최초30초 초과 미재현·native 스택 미확보를 명시했다. 운영AI 재배포/계측 미반영과 bind mount 제한도 기록했다.
- 진단 관찰기/게시 도구 Ruff·문법8파일·문서 공백 검사 통과·PNG 첫 페이지 확인. 신규 제품 단위 시험은 없다.
- 브라우저 조작/다른 PC 접속은 재시험하지 않았다. I40/I34/I32/I07 OPEN·P5.1 IN_PROGRESS·DONE 9/33 유지.

### v0.81 게시 대조 (2026-09-30)

- HTML 22,627 bytes·PDF 124,361 bytes, 사내 게시본과 로컬 생성본 일치·WBS 33행 상태 일치.
- [모델별 실행 대조와 게시 SHA-256](evidence/model_state_audit_20260930.json).
- GPU 시험3회 중 종료 시간 초과1회·정상 종료2회로 기록. 영상3파일 디코드/원본 PTS·DTS 대조 통과, 외형 DB 각28행·무결성ok.
- 제품 추론 코드 변경·새 제품 단위 시험 없음. 관찰기/게시 도구 Ruff·문법·문서 공백 검사 통과·PNG 첫 페이지 확인.
- 브라우저 조작/다른 PC 접속은 재시험하지 않았다. I40/I34/I32/I07 OPEN·P5.1 IN_PROGRESS·DONE 9/33 유지.

### v0.80 게시 대조 (2026-09-30)

- HTML 22,566 bytes·PDF 125,477 bytes, 사내 게시본과 로컬 생성본 일치·WBS 33행 상태 일치.
- [CUDA 조건 대조와 게시 SHA-256](evidence/cuda_comparison_20260930.json).
- 진단 도구 회귀 12 PASS·Ruff·문서 공백 검사 통과. PNG 첫 페이지 확인, 브라우저 조작/다른 PC 접속은 재시험하지 않았다.
- AGX Orin/TensorRT 10.3에서 DLA 2코어 인식 확인. 성능 시험과 운영 적용은 하지 않았으며 GPU 안정화 후 P6.2 비교 후보로 기록했다.
- CUDA700 원인 미확정·I34 OPEN·P5.1 IN_PROGRESS·DONE 9/33 유지.

### v0.79 게시 대조 (2026-09-30)

- HTML22,589bytes·PDF125,277bytes, 사내 게시본과 로컬 생성본 일치·WBS33행 상태 일치.
- [시간·매칭 대조와 게시 SHA-256](evidence/match_timing_20260930.json).
- 격리2회 재연결/정상 종료·수신4파일 디코드 exit0/stderr0·패킷PTS 중복/역전0. 외형 DB27/29행 무결성ok.
- 추론 코드는 수정하지 않았다. 게시 도구 Ruff·diff 검사 통과·PNG 첫 페이지 확인. 브라우저 조작/다른 PC 접속은 재시험하지 않았다.
- 기존 매칭 누락은2회 미재현으로 원인 미확정 유지. CUDA700/I32/I07 OPEN·P5.1 IN_PROGRESS·DONE9/33.
- 운영 healthy/restart0·기존 시작 시각 유지·격리 시험 컨테이너0. 다음은 최초 CUDA 실패 조건과 현재 후보 차이 대조.

### v0.78 게시 대조 (2026-09-30)

- 사내 HTML/PDF와 로컬 생성본 바이트 일치·WBS33행 상태 일치, PNG 첫 페이지 육안 확인.
- 신규 종료 회귀2건 수정 전 실패·수정 후 통과. 관련256 PASS/2 SKIP·Ruff/diff 검사 통과.
- 기존 후보 종료 SIGABRT2회 보존. 수정 후보는 인코더 포함/제외·재연결 중 종료·RTSP 경로에서 정상 종료.
- 수신803프레임·외형DB24행 무결성ok. 기본 null 시간 변환 경고8줄 보존, 입력 시간 기준 전체 디코드 오류0·원본PTS/DTS 증가 확인.
- 수신 한도55초 대비 파일40.58초와 복구 매칭 누락은I32 잔여. 관제 ACK/전체 모델/운영 합격으로 확대하지 않음.
- I39 FIXED_LOCAL·I34 OPEN·P5.1 IN_PROGRESS. 운영 healthy/restart0·기존 시작 시각 유지·시험 컨테이너0.
- [진단/수정/시험 및 게시 근거](evidence/candidate_shutdown_20260930.json). 브라우저 조작/다른 PC 접속 시험은 수행하지 않음.

### v0.77 게시 대조 (2026-09-30)

- 사내 HTML/PDF와 로컬 생성본 바이트 일치·WBS33행 상태 일치, PNG 첫 페이지 육안 확인.
- 입력/코덱 경계 기록기9 PASS·Ruff/diff 검사 통과. 진단용 변경이며 제품 코드/API/DB/운영 설정 변경 없음.
- 검사 전용 재연결60초에서40프레임 처리 목표·정상 종료·최종 memcheck 오류0. 운영 재연결값은 유지.
- 네이티브 파일40프레임 시험은 정상 종료·최종 memcheck 오류0·출력40프레임 전체 디코드 오류0.
- 네이티브 RTSP 시험은 EOS/종료/협상 실패를 보존했으며 원 장애 재현이나 C++ 전환 효과의 근거로 사용하지 않음.
- 원 CUDA700 근본 원인 미확정·I34 OPEN·P5.1 IN_PROGRESS. 운영 healthy/restart0·기존 시작 시각 유지·시험 컨테이너0.
- [진단/시험 및 게시 근거](evidence/source_boundary_20260930.json). 브라우저 조작/다른 PC 접속 시험은 수행하지 않음.

### v0.76 게시 대조 (2026-09-30)

- 사내 HTML/PDF와 로컬 생성본 바이트 일치·WBS33행 상태 일치, PNG 첫 페이지 육안 확인.
- 최초 오류 기록기7 PASS·Ruff/diff 검사 통과. 시작3조건에서 CUDA700 미재현.
- RTSP 메모리 검사2회 timeout 원자료 보존. 검사 전용 재연결 해제 대조는40처리프레임 목표 달성·종료0·최종 오류0.
- 원 CUDA700 근본 원인 미확정·I34 OPEN·P5.1 IN_PROGRESS. 운영 healthy/restart0·기존 시작 시각 유지·시험 컨테이너0.
- [진단/시험 및 게시 근거](evidence/cuda_capture_20260930.json). 브라우저 조작/다른 PC 접속 시험은 수행하지 않음.

### v0.75 게시 대조 (2026-09-30)

- HTML/PDF 사내 게시본과 로컬 bytes 일치·WBS33행 상태 대조 통과.
- [GPU/VIC·현재 후보 복구·종료 검토 근거](evidence/preprocess_comparison_20260930.json).
- 각120프레임×2회 완료·장치별 반복 일치, 장치 간 관절 차이로 VIC 승격 보류.
- 현재 GPU 후보359파일 고정·25초 단절 후8.065초 복구. 두 출력 디코드 정상.
- 복구 경계 매칭 누락1건·I34/I07 잔여. P5.1 IN_PROGRESS·10-01 종료 검토 목표, DONE 확약 아님.
- PNG 첫 페이지 확인·Ruff/diff 통과. 제품 코드 변경/단위 테스트 재실행 없음.
- 운영 healthy/restart0·시험 컨테이너 잔류0. 브라우저 조작·다른 PC 접속 시험 아님.

### v0.74 게시 대조 (2026-09-30)

- HTML/PDF 사내 게시본과 로컬 bytes 일치·WBS33행 상태 대조 통과.
- [CUDA 시작 경로·계측/소스 해시·게시 근거](evidence/cuda_startup_20260930.json).
- 보존본2회/후보2회·Primary VIC1회 정상, 출력5파일 디코드 stderr0.
- 현재 GPU 후보120프레임 Compute Sanitizer 오류0·정상 종료. I34 OPEN 유지.
- PNG 첫 페이지 확인·Ruff/diff 통과. 제품 코드 변경/단위 테스트 재실행 없음.
- 운영 healthy/restart0·격리 컨테이너 잔류0. P5.1 IN_PROGRESS·DONE9/33.
- 브라우저 조작·다른 PC 접속 시험 아님. VIC 정확도/장기/운영 검증 잔여.

### v0.73 게시 대조 (2026-09-30)

- HTML/PDF 사내 HTTP 게시본과 로컬 bytes 일치·WBS33행 상태 대조 통과.
- [동시 부하·소스/원자료 해시·게시 근거](evidence/tcn_loaded_20260930.json).
- PNG 첫 페이지 확인·Ruff/diff 통과. 새 제품 코드 변경/단위 테스트 재실행 없음.
- 격리600초·TCN47회 관찰, 주기 마지막 계측46회(종료 시점 차이). timeout/CUDA 서명0.
- Pose SGIE/RF·얼굴·관제 ACK 미검증. 출력 간격0.435초·장기 메모리·I34 원인 잔여.
- 운영 healthy/restart0·시험 컨테이너 잔류0. P5.1 IN_PROGRESS·DONE9/33.
- 브라우저 조작·다른 PC 접속 시험 아님.

### v0.72 게시 대조 (2026-09-30)

- HTML/PDF 사내 게시본과 로컬 생성본 bytes 일치·WBS33행 상태 대조 통과.
- [TCN 실제 모델·변경 SHA-256·게시 근거](evidence/tcn_worker_20260930.json).
- 고유194개 테스트·Ruff·Compose config·diff 검사 통과. PNG 첫 페이지 육안 확인.
- 단발11.26~12.42초, 준비 후1.16~1.60초·결과 일치. 최초 호출13.7초는 별도.
- 영상 교체·실제 자식 강제 종료 복구 확인. 전체 AI 동시 부하/장시간·다른 PC 접속 시험 아님.
- 옵션 기본false·운영 AI healthy/restart0 유지. I38 원timeout 및 I37/I32/I34/I07 OPEN.
- P5.1 IN_PROGRESS·DONE9/33 유지. 다음 전체 AI 병행·연속 메모리 검증.

### v0.71 게시 대조 (2026-09-30)

- HTML 22579bytes·PDF 125775bytes, 사내 게시본과 로컬 생성본 일치.
- [표본 계약·변경 SHA-256·게시 근거](evidence/rf_sampling_contract_20260930.json).
- 관련205 PASS·라벨 포함 통합2건 재확인(중복 합산 안 함)·Ruff·diff 검사·WBS33행 일치.
- PNG 첫 페이지 육안 확인. 실CCTV/GPU·브라우저 조작/다른 PC 접속 재시험 아님.
- AI 운영 배포/모델 교체 없음. 다음 I38 지연 보완, RF 원본/점수 기준은 I37로 추적.
- I37 전체/I38/I32/I34/I07 OPEN, P5.1 IN_PROGRESS·DONE9/33 유지.

### v0.70 게시 대조 (2026-09-30)

- HTML 22588bytes·PDF 126335bytes, 사내 게시본과 로컬 생성본 일치.
- [시간창 검증·변경 SHA-256·게시 근거](evidence/rf_window_contract_20260930.json).
- 관련155 PASS·Ruff·diff 검사·WBS33행 상태 대조 통과. PNG 첫 페이지 육안 확인.
- 합성 Pose 수집→학습→모델 재로딩→추론 경로 확인. 기존 후보 원본은 기록 경로5개 미존재.
- 실CCTV/GPU·브라우저 조작/다른 PC 접속 재시험 아님. AI 운영 배포 없음.
- I37 전체/I38/I32/I34/I07 OPEN, P5.1 IN_PROGRESS·DONE9/33 유지.

### v0.69 게시 대조 (2026-09-30)

- HTML22724bytes·PDF127072bytes, 사내 게시본과 로컬 생성본 일치.
- [출처 검사·실제 모델·게시 근거](evidence/rf_feature_source_20260930.json).
- 회귀142/DeepStream 낙상 연동32, 총174 PASS·Ruff·diff 검사·WBS33행 상태 대조 통과.
- 실제 기존 모델 예측0회·호환 후보 합성 입력 예측1회. 모델 교체·새 실CCTV 시험·AI 운영 배포 없음.
- 현재 RF는 새 후보의 inline 경로에서 사용 불가. 호환 후보 세부 계약/현장 라벨 검증 후 선택.
- 브라우저 조작/다른 PC 접속 재시험 아님. I37 전체/I38/I32/I34/I07 OPEN, DONE9/33 유지.

### v0.68 게시 대조 (2026-09-30)

- HTML22692bytes·PDF124254bytes, 사내 게시본과 로컬 생성본 일치.
- [실행·게시 SHA-256 근거](evidence/fall_sampling_timing_20260930.json), RF 표본 회귀116 PASS·Ruff·WBS33행 상태 대조 통과.
- 실CCTV600초 TCN22회 완료/timeout0·출력 영상 디코드 정상·외형DB361건 무결성 확인.
- PNG 첫 페이지 육안 확인. 브라우저 조작/다른 PC 접속 재시험 아님.
- AI 운영 healthy/restart0·시작 시각 유지·격리 시험 컨테이너 잔류0. 배포 없음.
- I37 중복 표본만 로컬 수정. 나머지 특징 계약과 I38 기존 timeout 원인·I32/I34/I07은 OPEN.

### v0.67 게시 대조 (2026-09-30)

- 새 주소 `http://192.168.2.141:18081/`: HTML22647bytes·PDF122850bytes, 로컬/HTTP bytes 일치.
- SHA-256·시험별 지표는 [실행/게시 근거](evidence/fall_contract_stability_20260930.json)에 기록했다.
- 단일 CCTV30분·단절 후7.828초 복구·출력3파일 디코드 정상·TCN320/640 별도 실행 확인.
- 회귀170 PASS·Ruff·Compose 검사 PASS·WBS33행/상태 일치. PNG 첫 페이지 육안 확인.
- 브라우저 조작/다른 PC 접속 재시험 아님. 시험 컨테이너 잔류0. 문서 서버만 주소 재연결, AI 운영 배포 없음.
- I37/I38/I32/I34/I07 잔여. P5.1 IN_PROGRESS·DONE9/33 유지.

### v0.66 게시 대조 (최종 확인 2026-09-30, 시험 09-29)

- HTML: 22577 bytes, SHA-256 `21a1145b59a518183638b547e82faff3c62f754a23f5880f0d6e694c8cd9345b` — LAN 게시본과 일치.
- PDF: 123887 bytes, SHA-256 `37b3c6bf53b8b41b7f75110c1135c4dd94b13324e25d0199bfa43b01a4260bd9` — LAN 게시본과 일치.
- 308 PASS/2 SKIP·Ruff·Compose 구성 검사·WBS33행 상태 대조·변경 범위 diff 검사 PASS. 수정 밖 기존 Compose 공백은 유지.
- 최종 실CCTV180초218접수/완료·큐 폐기/실패0·수신150.226초 디코드 오류0. 최종 앱30프레임 CUDA 메모리 검사 오류0.
- PNG 첫 페이지 육안 확인, 브라우저 조작/다른 PC 접속 재시험 아님. 운영 healthy/restart0·시험 컨테이너 잔류0·배포 없음.
- I36 단일 입력 FIXED_LOCAL. I34 CUDA700 원인/I32 영상 매칭·부하/I07 운영 확인 잔여.

### v0.65 게시 대조 (2026-09-29)

- HTML: 22564 bytes, SHA-256 `270c3decb2b5b23e5c7eb4b7ef7ed4cb1ac24f3044ba895ec74cdae920886647` — LAN 게시본과 일치.
- PDF: 123980 bytes, SHA-256 `d72643eb336ef1b86d1e65ad0dc74f5053000396166a521954bfcbf4e703be22` — LAN 게시본과 일치.
- 제품 시작·종료 대기 수정, 182 PASS/2 SKIP·Ruff 통과. 실제 CLI30프레임 Compute Sanitizer 오류0·RTSP 게시·정상 종료.
- 실CCTV60초 관찰에서1422프레임 처리·재시작0·CUDA 오류 로그0. 수신 영상30.203초 전체 디코드 오류0. 후처리 큐 폐기307건으로 I32는 OPEN 유지.
- PNG 첫 페이지 배치 육안 확인 후 시각만 갱신. 브라우저/다른 PC 접속 재시험 아님. 운영 healthy/restart0·시험 컨테이너 잔류0.
- I35 FIXED_LOCAL, I34 최초 CUDA700 원인 미확정·운영 적용 보류. TCN 별도 프로세스의 CUDA 커널은 이번 application-only 메모리 검사 범위 밖.

### v0.64 게시 대조 (2026-09-29)

- HTML: 22583 bytes, SHA-256 `7e592b1a628a7a402e882f0db33f64292fc766a26162c73bf3ad3159704e4468` — LAN 게시본과 일치.
- PDF: 123223 bytes, SHA-256 `749083093367915f629fdc0faff8a0b1efdb9d85dd8e0541ebe3ebefd4cee89a` — LAN 게시본과 일치.
- 실CCTV2회·메모리4경로 검사 완료, 정상 수신2파일 디코드 오류0. 시험 구성 실패2건은 별도 보존/제외.
- 제품 코드 추가 변경 없음: 이전255 PASS/2 SKIP 회귀는 반복하지 않음. 진단 스크립트4개 문법·renderer Ruff·WBS33행/상태·diff 확인.
- PNG 첫 페이지 육안 확인. 브라우저/다른 PC 접속 재시험 아님. 운영 healthy/시작 시각 유지/restart0·시험 컨테이너 잔류0.
- I34 최초 원인 미확정·I32 실CCTV 큐 폐기1535/821건·운영 적용 보류.

### v0.63 게시 대조 (2026-09-29)

- HTML: 22624 bytes, SHA-256 `7a9dc3b081e63e35cca6290ce75f7268a2618f5d9d75be0ad1a5176a6ffde7fc` — LAN 게시본과 일치.
- PDF: 123419 bytes, SHA-256 `55c67be5c995fcd897c7e7d1e0653a154c0d45ec349113848764492238de730f` — LAN 게시본과 일치.
- 255 PASS/2 SKIP·변경 코드 Ruff·WBS33행/상태·Python AST·변경 코드/문서 diff 검사 통과. 기존 Compose의 수정 외 공백 경고는 별도 유지.
- PNG 첫 페이지 육안 확인. 브라우저 동작/다른 PC 접속 재시험 아님. 운영 healthy/시작 시각 유지/restart0·격리 컨테이너 잔류0.
- I34 치명적 오류 종료 보완·600초/교차 시험. 최초 원인 미확정·실CCTV/장기/운영 잔여.

### v0.62 게시 대조 (2026-09-29)

- HTML: 22601 bytes, SHA-256 `2568beb26d704fa450fc1235d4c663df34ff56b6583fa2e0b46fd2e8a5a943c5` — LAN 게시본과 일치.
- PDF: 124344 bytes, SHA-256 `eadfe275bd98952628a73db9b8d38f0f652dc5d70b7b121175e8f4af993d0f0e` — LAN 게시본과 일치.
- 248 PASS/2 SKIP·변경 코드 Ruff·WBS33행/상태·Python AST·변경 코드/문서 diff 검사 통과. 기존 Compose의 수정 외 공백 경고는 별도 유지.
- PNG 첫 페이지 육안 확인. 브라우저 동작/다른 PC 접속 재시험 아님. 운영 healthy/시작 시각 유지/restart0·격리 컨테이너 잔류0.
- I34 VIC 우회 로컬 검증, 원인 확정/영구 해결과 구분. 장시간·실CCTV·운영 검증 잔여.

### v0.61 게시 대조 (2026-09-29)

- HTML: 22661 bytes, SHA-256 `27f2988fdcdd604c374f3a69e3aabbc70f1b59028df2d977f7828b99f8a9b261` — LAN 게시본과 일치.
- PDF: 125186 bytes, SHA-256 `52cc984f8ce1917ebc471338f0d411a6d7e4fede82547a2030599271d3e58cb7` — LAN 게시본과 일치.
- 관련 회귀213 PASS/2 SKIP·변경 파일 Ruff·WBS33행/상태·Python AST4파일·diff 검사 통과. 브라우저 동작과 다른 PC 접속은 재시험하지 않음.
- 운영 healthy/시작 시각 유지/restart0·격리 컨테이너 잔류0. 후처리 큐 포화와 I34는 OPEN 유지.

### v0.60 게시 대조 (2026-09-29)

- HTML: 22804 bytes, SHA-256 `2341f23c7197a8514622360ab1f829c2e856361722b74a6117201f4e3dbb2e42` — LAN 게시본과 일치.
- PDF: 124149 bytes, SHA-256 `285603c3c0cbe6071a3d6f515d4b911f7ba0b58d9ef7bd2c74730e49bb30c262` — LAN 게시본과 일치.
- WBS 33행/상태 대조·renderer Ruff·Python AST4파일·diff 검사 통과. 생성 PNG 첫 페이지 육안 확인. 브라우저 동작/다른 PC 접속은 재시험하지 않음.
- 첫 CUDA 실패와 후처리 큐 폐기를 기록했으며 운영 전환을 하지 않음. 시험 컨테이너 잔류0.

### v0.59 게시 대조 (2026-09-29)

Jetson 자체 LAN 주소에서 조회한 HTML/PDF가 생성본과 바이트 단위로 일치했다.

- HTML: 22719 bytes, SHA-256 `fcfcd1a1e6cdc258eab12decbd90911787f85002f43b302d65fbfd22961926df` — 일치.
- PDF: 127120 bytes, SHA-256 `9ccd31e3aece27f7c85dae62c8bf3a869aac218e3f9e6189bfa800979f6ee315` — 일치.

RC2 소스357개/모델8개 해시·영상/DB 복사본 실제 시험·보조 코드 문법·렌더러 Ruff·WBS33개 상태/집계·변경 범위 공백 검사 PASS. 제품 변경이 없어 단위 테스트는 반복하지 않았다. 브라우저/필터·다른 PC 접근은 재시험하지 않았다. [실행·게시 근거](evidence/rc2_content_20260929.json).

### v0.58 게시 대조 (2026-09-29)

Jetson 자체 LAN 주소에서 조회한 HTML/PDF가 생성본과 바이트 단위로 일치했다.

- HTML: 22673 bytes, SHA-256 `34f15d97daa528b1800c1d189958c1d911e37562cdf2ee3b9843b5bb526914bb` — 일치.
- PDF: 124871 bytes, SHA-256 `0081d445165db98653493c9e742605114a660bd1eb1b3bbc61e6338f85d1316a` — 일치.

167 PASS/2 SKIP·Ruff/문법·WBS33개 상태/집계·변경 범위 공백 검사 PASS. 수정 후보 격리3회·전후6파일 시간 헤더/디코드 검증, I33 FIXED_LOCAL. 브라우저/필터·다른 PC 접속은 재시험하지 않았다. [실행·게시 근거](evidence/osd_recovery_vui_20260929.json).

### v0.57 게시 대조 (2026-09-29)

Jetson 자체 LAN 주소에서 조회한 HTML/PDF가 생성본과 바이트 단위로 일치했다.

- HTML: 22648 bytes, SHA-256 `facd3e6f70586ab1be178a24b52acac8202adf185c75aec39c5c6c6c26abfa56` — 일치.
- PDF: 125429 bytes, SHA-256 `9c615cddae64b18e7c7101f039e67383e80ecce6fb986501b7348a0f0f003bab` — 일치.

167 PASS/2 SKIP·Ruff/문법·Compose 빈 기본값·WBS33개 대조 PASS. 제품/문서 변경 범위 공백 검사 PASS; Compose 전체 diff에는 이번 수정 밖의 기존 공백 경고(1425/1430행)가 남아 있다. 브라우저/필터·다른 PC 접속은 재시험하지 않았다. [실행·게시 근거](evidence/osd_jit_20260929.json).

### v0.56 게시 대조 (2026-09-29)

Jetson 자체 LAN 주소에서 조회한 HTML/PDF가 생성 파일과 바이트 단위로 일치했다.

- HTML: 22707 bytes, SHA-256 `b8e027dc6e2989b9fd61aec33877d9de1c08430363165400c5d8bf1467e9de88` — 일치.
- PDF: 124706 bytes, SHA-256 `b526a38edb6692d9cdeff8819d5bb600c6f61f30acb359d6dee0c410af9a2ac3` — 일치.

WBS33개 상태/집계·시험 보조 코드 문법·렌더러 Ruff/문법·문서 공백 검사 PASS. 제품 수정이 없어 제품 단위 테스트는 반복하지 않았다. 격리 시험의 실패와 초기 정지를 유지하며 확정 수정안 없음·I20/I33 OPEN으로 기록했다. 실제 브라우저/필터·다른 PC 접근은 재시험하지 않았다. [전체 근거](evidence/output_stall_20260929.json).

### v0.55 게시 대조 (2026-09-29)

Jetson 자체 LAN 주소에서 받은 HTML/PDF가 생성 파일과 바이트 단위로 일치했다.

- HTML: 22654 bytes, SHA-256 `4fef3121728165e21ead1a7779d8a9d20d9a815346edb254059a8fb975283738` — 일치.
- PDF: 122076 bytes, SHA-256 `00e5888617580bd6393d47eeba84846ce06e9ac08087ede67d37f043a452fc2d` — 일치.

WBS33개 상태/집계·시험 보조 코드5개와 렌더러 문법·렌더러 Ruff·공백 검사 PASS. 제품 코드 변경이 없어 기존 제품 단위 테스트는 반복하지 않았다. 이번 격리4회 중 RTSP 수정 후보1회는 실패했고 나머지3회는 관측 구간을 완료했다. RF 부품 실행·TCN 유효 결과와 I33 재개 근거를 [JSON](evidence/rf_tcn_20260929.json)에 보존했다. 운영 AI 엔진 healthy·재시작0, 시험 컨테이너 잔여0. 브라우저 화면/필터·다른 PC 접속 재시험은 하지 않았다.

### v0.54 게시 대조 (2026-09-29)

Jetson 자체 LAN 주소에서 받은 HTML/PDF와 생성 파일을 바이트 단위로 대조했다.

- HTML: 22595 bytes, SHA-256 `17c602a01cd6b3932ee54b3d8b9c3b3ae2373d8c7d675761a8a58b9bbbd578b4` — 일치.
- PDF: 125695 bytes, SHA-256 `1f32b3235c206a33f0eb4f6ff17b74fbd1c00a57ae43586a202616b60df06891` — 일치.

WBS 33개 상태/집계·렌더러 Ruff/문법·문서 공백 검사 PASS. 제품 회귀158 PASS/2 SKIP,
최종 실제 소스 GPU 시험의 수신386프레임·정상 헤더·디코드 경고0은 [I33 근거](evidence/h264_timing_20260929.json)에 기록했다.
브라우저 화면/필터 재시험은 하지 않았다. I33 FIXED_LOCAL이며 운영 적용·다중 성능 합격은 별도다.
모델·영상·얼굴 자료는 진행판에 게시하지 않았다.

### v0.53 게시 대조

Jetson 자체 LAN 주소에서 받은 HTML/PDF와 생성 파일을 바이트 단위로 대조했다.

- HTML: 22628 bytes, SHA-256 `549703772729b1ff4c5309b4c7164d42c0a90ad320d9b9f4fa3d25270be46939` — 일치.
- PDF: 125043 bytes, SHA-256 `77aff1ff760912c41a40ae2184366618854f89576c62affd15564c19b489b9c2` — 일치.

WBS 33개 상태/집계·시험 스크립트3개 문법·렌더러 Ruff/문법·문서 공백 검사 PASS.
GPU 통합 실행2회·영상 디코드/헤더·시험 DB 확인 결과는 [RC1 실행 근거](evidence/rc1_integration_20260928.json)에 기록했다.
출력 시간 정보는 I33 OPEN이며, 브라우저 화면/필터 재시험·모델 품질 합격·운영 반영과 구분한다.
모델·영상·얼굴 자료는 진행판에 게시하지 않았다.

### v0.52 게시 대조

Jetson 자체 LAN 주소에서 받은 HTML/PDF와 생성 파일이 바이트 단위로 일치했다.

- HTML: 22628 bytes, SHA-256 `ef6dd5951de90873c355b5f97304f2ee171bd888b72c901db78b2d0995d98664` — 일치.
- PDF: 123832 bytes, SHA-256 `5bb49edebcd1433d9fa837ba554a7829c9d3d5c0acaa4df7efaabd21eb9ebc3e` — 일치.

WBS 33개 상태/집계·후보 Python 339개 문법·렌더러 Ruff/문법·문서 공백 검사 PASS.
이번에는 GPU 통합 실행·브라우저 화면 재검증을 하지 않았다. 운영 시작 시각 유지·재시작 0·healthy 재확인.
근거: [RC1 보존 JSON](evidence/profile_freeze_20260928.json). 후보 모델·영상·환경 파일은 진행판에 게시하지 않았다.

### v0.51 게시 대조

Jetson 자체 LAN 주소의 HTML/PDF와 생성 파일을 바이트 단위로 대조했다.
- HTML: 22665 bytes, SHA-256 `a30395e41797e6e1175d63b853fa07147ee01aa3bd1fe8ed9149d122f0a00a11` — 일치.
- PDF: 126694 bytes, SHA-256 `6b6ec43d658828eedeb54ca07b7e0e97997e4514bd2f2d964695ffde79c3c7a0` — 일치.

WBS 33개 ID/상태·집계, 렌더러 Ruff·문법·문서 공백 검사 PASS. 브라우저 재시험은 하지 않았다.
제품 코드 변경이 없어 제품 단위 테스트는 추가 실행하지 않았으며 복원 후보 GPU 실행 3회를 검증했다.
근거: [관측 부하/복원 JSON](evidence/latency_restore_20260928.json).

### v0.50 게시 대조

Jetson 자체 LAN 주소에서 HTML/PDF를 받아 생성 파일과 바이트 단위로 대조했다.
- HTML: 22521 bytes, SHA-256 `02a01e8b94f40eedbf6fb1355beedb6cfa54eaaac1b91ff3d2059270014c01e6` — 일치.
- PDF: 124876 bytes, SHA-256 `80b86b8994d8aebb634b7f0667dd76f6f744204bbb5693f080d48c1cb909e243` — 일치.

WBS 33개 ID/상태·집계, 렌더러 Ruff·문법·문서 공백 검사 PASS. 브라우저/필터 재시험은 하지 않았다.
제품 코드 변경이 없어 새 제품 단위 테스트는 실행하지 않았으며 실CCTV 수집 영상의 격리 GPU 재생을 검증했다.
근거: [실CCTV 검증 JSON](evidence/real_camera_20260928.json). 영상 파일과 스냅샷은 진행판에 게시하지 않았다.

### v0.49 게시 대조

Jetson 자체 LAN 주소의 HTML/PDF를 생성 파일과 바이트 단위로 대조했다.
- HTML: 22575 bytes, SHA-256 `0aa779d1263db1b05590e1c24515c929d6d634335957716c78de3b59926a2772` — 일치.
- PDF: 124344 bytes, SHA-256 `b8274e4fd2f59e67d84d81dee58cd92e7c20a5f18d53fa997bcb18691b34e787` — 일치.

WBS 33개 ID/상태·집계, 렌더러 Ruff·문법·문서 공백 검사 PASS. 브라우저/필터 재시험은 하지 않았다.
제품 코드 변경이 없어 새 제품 단위 테스트는 실행하지 않았으며 실제 GPU 단절/복구 시험 2회를 수행했다.
근거: [복구 검증 JSON](evidence/input_recovery_20260928.json).

### v0.48 게시 대조

Jetson 자체 LAN 주소에서 HTML/PDF를 받아 생성 파일과 바이트 단위로 대조했다.
- HTML: 22600 bytes, SHA-256 `b4a9b14f36459bb7cd1559a0e23066dc6eaf8a4ce6742b2e17adc261dea5eeaa` — 일치.
- PDF: 125310 bytes, SHA-256 `c2a96d928f018eb5c21898eb5e032b7011e1cd52c4ddbd512b263b78652f01a3` — 일치.

WBS 원본 33개 ID/상태·집계 일치, 렌더러 Ruff·문법·문서 공백 검사 PASS.
새 제품 단위 테스트는 실행하지 않았다(제품 코드 변경 없는 격리 실행/문서 작업).
근거는 [입력 전송 검증 JSON](evidence/input_rate_20260928.json)에 보존했다.

## 파일과 서비스

| 구분 | 위치/값 |
|---|---|
| 원본 | [WBS](WBS.md), [진행 기록](PROGRESS.md), [이슈](STATUS.md) |
| 렌더링 데이터 | [progress_v095.json](figma_exports/progress_v095.json) |
| 생성 도구 | [render_progress_snapshot.py](../../scripts/ops/render_progress_snapshot.py) |
| HTML 요약·서식 | [progress_snapshot_template.html](../../scripts/ops/progress_snapshot_template.html) |
| 게시물 | [HTML](online/index.html), [PDF](online/CCTV_project_progress.pdf) |
| 별도 Compose | [deploy/progress/compose.yml](../../deploy/progress/compose.yml) |
| 서버 설정 | [nginx.conf](../../deploy/progress/nginx.conf) |
| 컨테이너 | `cctv-progress-web`, 기존 로컬 `nginx:1.27-alpine` 재사용 |
| 네트워크 | `192.168.2.141:18081` → 전용 nginx `8080` |
| 재시작 설정 | `unless-stopped`; 운영 CCTV Compose와 분리 |

서버는 진행판 폴더를 읽기 전용으로 마운트하며 `/`, `/index.html`, PDF만 응답한다.
프로젝트 전체·DB·영상·비밀 설정은 마운트하지 않는다. CPU 0.25개·메모리 64MiB 제한을 적용했다.
기존 CCTV 컨테이너를 재기동하거나 장치 명령을 실행하지 않았다.

## 매 업무 종료 시 갱신

실시간 자동 동기화나 백그라운드 개발 추적은 없다. 이후 작업도 같은 주소로 수동 게시한다.

1. WBS·PROGRESS·STATUS의 상태/근거/이슈를 먼저 갱신한다.
2. 렌더링 JSON의 버전·기준 시각·WBS 행/집계를 원본과 맞춘다. 새 버전 파일을 만들면 이 문서의 경로도 갱신한다.
3. HTML 템플릿과 PDF 생성 함수의 이번 완료·다음 작업·검증/이슈 요약을 함께 수정한다.
   **현재 요약 문구는 코드/템플릿에 명시돼 있다. WBS JSON만 바꾸면 요약은 자동 갱신되지 않는다.**
4. 아래 명령으로 HTML/PDF를 재생성한다. 읽기 전용 bind mount를 통해 서버에 즉시 반영된다.
5. 같은 URL의 HTTP 응답·버전·WBS ID/상태를 원본과 대조하고 근거를 기록한다.
   실패하면 게시본을 최신이라고 표시하지 않는다.

```bash
rtk proxy env MPLCONFIGDIR=/tmp/cctv-progress-mpl .venv/bin/python scripts/ops/render_progress_snapshot.py --data docs/planning/figma_exports/progress_v101.json --output docs/planning/online/CCTV_project_progress.pdf
rtk proxy curl --noproxy '*' --max-time 5 -I http://192.168.2.141:18081/
```

문서 내용 변경에는 컨테이너 재시작이 필요 없다. 시작/상태 확인/중지는 프로젝트 루트에서:

```bash
rtk proxy docker compose -f deploy/progress/compose.yml up -d --no-build
rtk proxy docker compose -f deploy/progress/compose.yml ps
rtk proxy docker compose -f deploy/progress/compose.yml stop
```

## 기존 온라인 도구 상태

- [FigJam](https://www.figma.com/board/2GQKlBrmSVjZljX74IJLij): 09-22 새 영역 `9:53`에 제목/갱신 중 안내만
  추가된 상태다. 이후 호출 한도로 WBS 본문은 추가하지 못했다. 최신 현황은 사내 진행판을 사용한다.
- Google Docs: 생성 403 거부, 생성 완료 문서 없음.
- Google Drive: `ChatGPT` 폴더는 생성됐지만 PDF 업로드는 `storageQuotaExceeded`로 거부됐다.
  업로드 완료 파일이나 PDF 공유 링크는 없다. 기존 Drive 파일·공유 권한은 변경하지 않았다.
