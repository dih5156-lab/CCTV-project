# P5.1 — 영상 입력·추론 계약과 계측 기준선

<a id="transition-rollback-v092"></a>

## v0.92 — I07 후보→이전 구성 전환·신규 데이터 보존 (2026-10-01)

**이번 단위 완료:** 같은 격리 DB에서 고정 후보를 실행·종료·컨테이너 제거한 뒤 이전 이미지와
보존 코드로 복귀했다. 후보가 새로 기록한 이벤트31건·외형28건을 포함해 **복귀 직전705,517행의
모든 열 값이 그대로 보존**됐다. 복귀 후에도 이벤트32건·외형28건이 추가됐고 DB3개 무결성은 모두ok다.
이전 구성의 저장 API에 후보 이벤트1건을 재접수해 같은 행 반환·전체 업무 행 변경0을 확인했다.
운영 AI는09-17 시작 시각·healthy/restart0 유지, 운영 배포·API/DB 스키마/모델 변경 없음.

### 순서와 범위

1. v0.91 시험 DB3개를 새 경로로 복제하고 `before` 스냅샷을 만들었다. 운영 DB는 사용하지 않았다.
2. 이미지 `809693da4ed0…`의 내장 코드와 고정 후보 환경으로90초 관측, SIGTERM 종료 후 `candidate` 스냅샷을 보존했다.
3. 후보 컨테이너만 제거했다. DB를 과거 백업으로 덮어쓰지 않고 같은 경로를 유지했다.
4. 이미지 `1ace5b7832a5…`와 v0.88 선택 디스크 코드, 이전 환경으로90초 관측·SIGTERM 종료했다.
5. 이전 저장 API의 동일 ID 재접수 검사 후 `rollback` 스냅샷을 만들고 후보 단계의 모든 기존 행과 비교했다.

두 실행 모두 network none·내부 RTSP 녹화1채널·브로커 차단·CPU2/메모리8GB 조건이다.
v0.91의 가상환경/SDK/모델 복구본을 재사용했고 고정 모델8개는 기록된 SHA와 일치했다.
후보는 TCN worker 재사용/10초 주기, 이전 구성은 단발 자식/300초 주기다.
코드·이미지·환경이 함께 달라지므로 실행 차이의 원인을 특정 설정 하나로 확정하지 않는다.

### 데이터 보존 결과

| 항목 | 전환 전 | 후보 종료 후 | 이전 구성 복귀 후 |
|---|---:|---:|---:|
| MQTT 전체 행 | 173767 | 173798 (+31) | 173830 (+32) |
| pending / sent | 32 / 173735 | 63 / 173735 | 95 / 173735 |
| 외형 기록 | 531689 | 531717 (+28) | 531745 (+28) |
| 얼굴 업무 행 | 2 | 2 | 2 |
| 직전 단계 기존 행 대조 | 기준 보존 | 705458행·변경/누락0 | 705517행·변경/누락0 |
| DB integrity_check / ID·대상 중복 그룹 | 3개ok / 0 | 3개ok / 0 | 3개ok / 0 |

비교 도구는 원래 열을 모두 대조하며 pending의 재시도 횟수/최종 시각/오류3열만 별도 변경으로 허용한다.
실제 이번 결과는 **허용한 열까지 변경0**이었다. 합성5조건으로 동일 데이터·정상 재시도 허용,
payload 변경·상태 변경·삭제 거부를 확인했다. 행 원문·개인정보·인증값은 공개 근거에 포함하지 않는다.
중복 검사는 후보가 실제 생성한 한 이벤트의 ID/대상으로 이전 API를 호출했으며, 전체173830행의
내용 해시와 행 수가 전후 동일했다. ID를 새로 부여한 의미상 중복·최종 장치 중복 실행까지 검증한 것은 아니다.

### 실행·영상 관측과 남은 문제

| 항목 | 고정 후보 | 이전 구성 |
|---|---:|---:|
| 마지막 Primary / 앱 처리 | 2404 / 2401 | 2001 / 1998 |
| 후처리 완료 / 매칭 누락 | 163 / 0 | 162 / 13 |
| 후처리 큐 폐기·실패 / 재시작 | 0 / 0 | 0 / 0 |
| TCN | 재사용7회 완료·Pose 부족1 | 단발1회20.32초 반환·Pose 부족1 |
| 앱/수신기 exit / SIGTERM 종료 | 0 / 6.37초 | 0 / 3.49초 |
| 영상 패킷/디코드 프레임 | 1613 | 1396 |
| 원본 PTS/DTS 중복·역행 / 디코드 오류 | 모두0 | 모두0 |
| 최대 영상 패킷 간격 | 414ms | 458ms |

영상은 각각60.176초/60.196초이며 합계3009프레임을 입력 시간기준으로 전부 디코딩했다.
연속성·카메라별30FPS·탐지 정확도 합격과 구분한다. 직접 main.py 자식에 SIGTERM을 보냈으며
강제 크래시·운영 `docker stop`·전체 서비스 재시작은 미검증이다.
이전 구성의 매칭 누락13은 I32에 보존했다. 이번 TCN 반환 성공은 v0.91 시간 초과의 원인 해결 증거가 아니다.
Pose 부족 결과를 정상 낙상 판정으로 계산하지 않는다. 검색한 CUDA700 치명 서명은0이나 I34 OPEN을 유지한다.

복귀 코드는09-17 시작 당시 메모리와 동일하다고 보장하지 않는다. 시작 DB에는 v0.91의 `track_id_text`
열이 이미 있었고 이번 전환에서 열 변화는 없었다. 따라서 파괴적 스키마 변경의 역방향 이행 시험은 아니다.
AI용3DB 범위이며 EdgeX/parser/관제 전체 저장소·과거 crop 파일·외부 ACK/실물 출력을 검증하지 않았다.

**다음 한 가지:** 이번 기록의 후보0/복귀13 매칭 누락 발생 시각과 TCN 시작/반환·CPU 지표를 대조한다.
새 원인 구별 조건 없이 GPU 시험을 반복하지 않는다. I07의 이번 격리 전환 단위는 완료했으며,
실제 운영 반영과 I32/I34/I37/I38 필수 조건은 남아 I07 OPEN·P5.1 IN_PROGRESS·DONE9/33을 유지한다.

[실행·DB 대조 근거](evidence/transition_rollback_20261001.json) · [복구 절차서](../handover/BACKUP_RESTORE_RUNBOOK.md).
원자료 `data/runtime/p51_transition_20261001_v093`, 시험 컨테이너는 정리하고 DB 스냅샷·로그·영상은 보존했다.

<a id="restore-bundle-v091"></a>

## v0.91 — I07 복구 묶음·이전 이미지 격리 기동 (2026-10-01)

**이번 단위 완료:** 가상환경·SDK·모델·선택 코드의 복구본, 암호화 설정, 이전 이미지 내보내기/재적재를 확인했다.
이전 이미지에 복구본을 연결해 입력→추론→영상 출력·DB 저장·SIGTERM 종료를 격리 검증했다.
기존 DB 행 보존은 통과했지만 첫 기동 시간 초과·TCN 시간 초과·매칭 누락이 있어 **전체 복귀/분석 품질 합격은 아니다**.
운영 AI는 기존 시작 시각·healthy/restart0 유지. 제품 코드·운영 구성·배포 변경 없음.

### 보존한 범위

| 항목 | 실제 확인 |
|---|---|
| 가상환경3개·DeepStream/NVIDIA SDK·모델·InsightFace 캐시·선택 코드 | 일반 파일49,917개, 논리 크기9,345,325,452바이트(약9.35GB). 파일별 복사 후 SHA 일치, 링크62개 별도 보존 |
| 이전 Docker 이미지 | `sha256:1ace5b7832a5…` 내보내기6,309,383,680바이트·59레이어, OCI index/config 해시 검사·동일 ID로 재적재 완료 |
| 비밀설정·실제 컨테이너 구성·얼굴 등록 파일 | AES-256-GCM 암호화, 복호화12파일 일치·변조 거부·파일600. 키는 별도 비공개 경로에 보관 |
| 기존 TensorRT 캐시 볼륨 | 읽기 전용 복사 시 실제 파일0개. 복구 모델 루트에 빈 `trt_cache` 연결 지점 준비 |
| DB | v0.88의8개 복원본을 별도 참조. 이 중 AI 사용3개를 시험 전용으로 복제, 운영 DB에 쓰지 않음 |

이번9.35GB는 모델·캐시·코드까지 포함한 **논리 크기**이며 v0.90 의존성 약7.69GB의 `du` 할당량과 범위/집계 방식이 다르다.
복구 묶음과 키는 Git/공개 문서 서버에 포함하지 않는다. 같은 디스크의 복구본이므로 디스크 고장 대비나
OS·커널·GPU 드라이버 복원, 다른 Jetson/서버 이식 검증으로 확대하지 않는다. 절대 링크7개도 같은 시스템 경로를 전제한다.

### 격리 실행 결과와 실패 보존

| 항목 | 결과 |
|---|---|
| 첫 실행 | 초기 진행 대기150초 시간 초과, 총176.08초·앱 로그0바이트. 같은 시각 이미지 재적재 진행, 인과관계 미확정 |
| 재실행 | 재적재 완료 후 import 시간 기록 활성화. 입력 준비46.00초·출력 준비46.83초, 이후90초 관측 |
| 프레임/후처리 | 마지막 Primary2218·새 추론1109, 앱2214. 후처리154완료·큐 폐기/실패0, 매칭 누락17 |
| TCN | 단발 자식 실행1회·시간 초과1회. 후보 재사용 경로의 앞선 성공과 구분, I38 OPEN |
| 종료 | 앱/수신기 exit0, 직접 SIGTERM 후2.59초. 운영 `docker stop` 시험 아님 |
| 영상 | 60.198초·1435패킷/프레임, PTS/DTS 중복·역행0, CPU 디코드 오류0. 최대 간격1.936초로 연속성 합격 아님 |
| DB | 3개 integrity_check 모두ok. 기존705,398행의 원래 열 전부 일치·신규 대기32건/외형28건 저장 |
| 오류 판정 | 검색한 CUDA700 치명 서명0·파이프라인 재시작0. I34 근본 해결 증거는 아님 |

첫 시험 실패 기록을 남겼다. 재실행은 이미지 재적재 종료와 import 계측이라는 두 조건이 달라졌으므로
시간 초과 원인을 디스크 부하로 확정하지 않는다. 첫 기동에서 앱 로그가 없어 동일 수준의 비교도 불가능하다.
TCN 시간 초과·매칭 누락·영상 간격의 인과관계도 이번 결과만으로 확정하지 않는다.

DB 최초 `SELECT *` 비교는 외형531,661행 모두 불일치로 실패했다. 기동 시 nullable `track_id_text` 열이
추가되어 튜플 길이가 달라진 것이며, **기존22개 열을 빠짐없이 다시 대조한 결과 변경0**이었다.
추가 열은 기존 행에서 모두NULL이고, MQTT173,735행·얼굴2행도 값/상태 동일하다.
신규 대기32건은 브로커를 의도적으로 차단한 결과다. 전달 성공·관제 ACK·실물 경보 합격을 뜻하지 않는다.

### 복귀 검증의 경계와 다음 단계

이 시험은 실제 운영의 명시적 추론 환경/CLI를 기반으로 하되, 카메라/브로커/비밀값/포트는 격리용으로 치환했다.
네트워크none, 녹화1채널의 내부 RTSP 재생, DB/쓰기 경로 분리. 코드/모델/가상환경/SDK는 복구본에서 읽었다.
코드는 **v0.88의 현재 디스크 선택351파일**이며09-17 시작 당시 메모리 코드를 복원한 것이 아니다.
기존 마운트에 없던 MediaPipe/FallData 가상환경2개는 추가했고, Sample/학습 데이터셋 마운트는 제외했다.
원래 RTSP 플러그인을 사용했으므로 I40 출력 장애 종료 보완이 적용된 후보 이미지와도 구분한다.

**다음 한 가지:** 같은 격리 DB에서 고정 후보 실행→이전 이미지·보존 코드 복귀를 순서대로 검증한다.
후보가 새로 기록한 대기 이벤트의 ID/내용/상태와 기존 행, 스키마 호환·중복 여부를 대조한다.
이번에는 그 전환 순서나 백업 이후 운영 데이터 무손실을 검증하지 않았다.
I07/I32/I34/I37/I38 OPEN·P5.1 IN_PROGRESS·DONE9/33, 필수 조건 충족 전 운영 배포 보류.

[검증 근거](evidence/restore_bundle_20261001.json) · [복구 절차서](../handover/BACKUP_RESTORE_RUNBOOK.md).
원자료 `data/runtime/p51_restore_20261001_v092`, 첫 시험/재시험 컨테이너만 정리했고 복구본·증거는 보존했다.

<a id="profile-start-v090"></a>

## v0.90 — I07 후보 전체 AI 기동·복구 의존성 점검 (2026-10-01)

**이번 단위 완료:** 새 후보 Compose를 기준으로 전체 `main.py` 기동·입력/추론/영상 송출·DB 저장과
두 종료 신호를 격리 환경에서 확인했다. 기동 전 발견한 I42 캐시 마운트 누락을 보완하고
후보 종료 유예60초를 명시했다. 관련 구성 회귀46 PASS·Ruff 통과.
운영 AI는09-17 시작 시각·healthy/restart0 유지, 후보 이미지 재빌드·운영 배포는 하지 않았다.

### I42 — 실제 기동에서 발견한 마운트 누락

v0.89는 후보 이미지/저장소 프로브와 Compose 정적 검사까지 통과했으나,
읽기 전용 모델 폴더와 그 안의 쓰기 가능한 `trt-cache` 볼륨을 함께 연결하지는 않았다.
이번 전체 마운트에서는 원본 모델 폴더에 `trt_cache` 디렉터리가 없어
Docker가 `read-only file system`으로 거부했다. **AI/CUDA 실행 이전 OCI 단계의 결함**이다.

연결용 빈 디렉터리를 준비하고 사전 검사에서 누락을 거부하도록 수정했다.
새 회귀가 수정 전 실패 → 수정 후 전체46 PASS, 실제 기동도 통과했다.
모델8개 파일 SHA는 변하지 않았다. I42 FIXED_LOCAL, 운영 미반영.
복구/새 모델 묶음을 만들 때도 이 빈 디렉터리를 함께 준비한다.

```bash
rtk proxy mkdir -p data/runtime/p51_rollback_20261001_v089/candidate_models/trt_cache
```

### 실제 후보 통합 실행 결과

| 항목 | 정상180초 관측 | SIGTERM 종료 회귀60초 관측 |
|---|---:|---:|
| 종료 전 Primary 프레임 | 5106 | 1633 |
| 앱/출력 수신기 종료 | 모두0 | 모두0 |
| 종료 신호/소요 | SIGINT·25.12초 | SIGTERM·4.86초 |
| 저장 영상 | 1080p H.264·150.331초·4048패킷 | 약40.3초·967패킷 |
| 원본 PTS/DTS 중복·역행 | 0 / 0 | 0 / 0 |
| 최대 패킷 간격 | 641ms | 530ms |
| 원 시간기준 전체 디코드 | 4048프레임·오류0 | 967프레임·오류0 |
| 누적 MQTT DB | pending61·고유 event_id61 | pending82·고유 event_id82 |
| 누적 외형 DB | 54행 | 73행 |
| DB integrity_check | 모두ok | 모두ok |

첫 시험 앱 처리5104·후처리317완료·후처리 큐 폐기/실패0·파이프라인 재시작0.
종료 직전 TCN13요청/12완료(결과 중 Pose 부족1), 마지막 요청의 종료 시점 결과는
최종 지표가 없어13완료로 계산하지 않는다. CUDA700 등 확인한 치명 서명은0이지만 **I34 해결 근거는 아니다**.
관측 구간 Primary 증가율은 약28.27/s이며 모델별 새 추론 FPS나4대×30FPS 합격과 구분한다.

최초 CPU 디코드 명령은 출력 시간기준 변환으로64건의 null muxer DTS 경고를 냈다.
파일 자체 패킷 PTS/DTS를 별도 검사하고, 입력 시간기준을 유지한 디코드에서
전체5015프레임·오류0을 확인했다. 최초 경고를 숨기거나 파일이 손상됐다고 단정하지 않는다.

**격리 범위:** 외부 네트워크 none, 기존 카메라 녹화본1채널을 내부 RTSP로 재생,
쓰기 경로/캐시/DB는 시험 폴더, 인증값은 시험값으로 치환했다. 기본 후보의 CLI 인자는 유지하고
브로커 주소/포트/토픽만 격리용으로 변경했다. 원본 카메라·관제·출력 장치에는 연결하지 않았다.
MQTT 브로커를 의도적으로 차단해 발행 실패 로그가 발생했고, 생성 데이터의 DB 대기를 확인했다.
따라서 이 시험은 브로커 전달 성공·관제 ACK·실물 출력·탐지 정확도 합격을 뜻하지 않는다.

SIGINT25.12초는 기존 시험보다 길다. 로그상 프로세서 정리에 약10초, 이후 프로세스 반환에 추가 시간이
걸렸으나 정확한 대기 원인은 미확정이다. 별도 SIGTERM4.86초 성공이 이를 해결했다는 증거는 아니다.
후보 Compose에60초 유예를 명시하고 사전 검사로 누락/짧은 설정을 거부한다.
이번 신호는 시험 도구가 **실제 main.py 자식 프로세스에 직접 전달**했으며, 운영 컨테이너에
`docker stop`을 실행하거나 운영 종료/복구 시간을 측정한 것은 아니다.

### 복구 의존성 확인과 남은 범위

| 후보가 외부에서 읽는 항목 | 현재 확인 | 복구 묶음 상태 |
|---|---|---|
| `.venv-jetson-train` | 약5.31GB, Python3.10.12 실행·Torch2.8.0/NumPy1.26.4/Ultralytics8.4.104 메타데이터 | 별도 복사/복원 미검증 |
| `.venv-mediapipe` | 약0.93GB, Python 실행·MediaPipe0.10.14/NumPy2.2.6 메타데이터 | 실제 RF/MediaPipe 분석 호출 합격과 구분 |
| `.venv-falldata` | 약0.26GB, Python 실행·scikit-learn1.7.2/NumPy1.26.1 메타데이터 | 별도 복사/복원 미검증 |
| 호스트 DeepStream | 약0.78GB, 7.1 경로·libnvds_infer SHA 확인, 실제 후보 추론 실행 | 호스트 SDK 복구본 미포함 |
| 호스트 NVIDIA 라이브러리 | 약0.41GB, 필요 경로 존재·후보 마운트 사용 | 호스트 의존성 복구본 미포함 |

합계 약**7.69GB(du 할당량)**. 가상환경95개 배포판 메타데이터 SHA를 기록했지만 전체 파일 백업은 아니다.
운영 이미지 `1ace5b7832a5…`와 후보 `809693da4ed0…`는 로컬 Docker에 존재한다.
이미지 내보내기·비밀설정/전체 마운트 묶음·이전 이미지로 전체 기동·실제 운영 복귀는 아직 미검증이다.
앞선 v0.88의 실제8DB 복원 근거는 재사용한다.

**다음 한 가지:** 외부 의존성·비밀설정/마운트를 포함한 복구 묶음을 만들고 이전 이미지로 격리 복귀 확인.
I07 OPEN·P5.1 IN_PROGRESS·DONE9/33, I32/I34/I37/I38 미해결과 운영 전환 보류를 유지한다.
[검증 근거](evidence/profile_start_20261001.json) · [회귀 JUnit](evidence/profile_start_20261001.xml).
원자료 `data/runtime/p51_profile_start_20261001_v091`.

<a id="candidate-profile-v089"></a>

## v0.89 — I07 운영용 후보 프로필·영속 저장 계약 (2026-10-01)

**이번 단위 완료:** `docker-compose.jetson-candidate.yml`과 기존 구성 검사 도구의 후보 검사 옵션을 추가했다.
운영 Compose 위에 병합해 사용할 검토용 파일이며, 자동 적용·systemd 변경·운영 AI 재시작은 하지 않았다.
I07 OPEN·P5.1 IN_PROGRESS·WBS DONE9/33을 유지한다.

| 항목 | 후보에 명시한 조건 | 이번 확인 |
|---|---|---|
| 코드/RTSP 플러그인 | 시험한 이미지 `sha256:809693da4ed0…`, build 제거·pull 금지, src/scripts/config 마운트 제거 | 별도 컨테이너에서 이미지 목록359파일 SHA 일치 |
| 모델 | `CCTV_CANDIDATE_MODEL_ROOT` 필수·읽기 전용, 기존 trt-cache는 영속 쓰기 유지 | v0.88 보존본8개 SHA 일치. TCN/Pose/RF/외형 경로 명시 |
| DB | `/app/data` 기존 영속 경로 유지, MQTT outbox/외형/얼굴 DB 경로 명시 | 후보 저장소 코드로 합성2건 저장 → 컨테이너 제거/재생성 → pending2 유지·재삽입 중복0·integrity ok |
| 비밀값 | `.env.jetson` 필수, MQTT 사용자/비밀번호·내부 토큰의 빈 주입 거부 | 운영값3종과 일치 여부만 확인, 빈 MQTT_PASSWORD는 Compose가 거부 |
| 외부 의존성 | 가상환경3개·호스트 SDK/라이브러리 읽기 전용, 카메라/구역/얼굴/캐시 등 필요 경로 유지 | bind 원본 존재·모드 검사. 없는 경로 자동 디렉터리 생성 금지 |
| 운영 연결 보존 | 기본 파일의 AI 명령·포트·healthcheck·네트워크·의존 서비스 상속 | 기본 Compose와 대조 일치, 다른 서비스 구성 전체 동일 |

검증한 추론 옵션(인물0.5초, GPU 복사1/CPU OSD0, TCN worker/640 등)은 후보에 명시했다.
기본 Compose 병합본과 환경 키21개가 달랐으며, 실제 운영 메모리와의 동작 비교는 아니다.
이 프로필은 외부 SDK·가상환경·개인 설정에 의존하므로 다른 서버에 이미지 하나만 전달하는 배포본이 아니다.

### 검사·격리 시험 범위

구성 회귀 **44 PASS**(신규 후보 경계7건 포함), Ruff 통과.
최초 전체 실행의 기존2실패는 실제 `.env`에 의존하던 단위 시험과 삭제된 Compose 옵션 선언을
요구하던 검사였다. 명시적 빈 환경 fixture와 현재 env_file 경로 검사로 보완해 재실행했다.
후보 검사는 재전송 DB 누락, 비밀값 누락, 코드/DB를 가리는 마운트·tmpfs,
읽기 전용 DB, 없는 원본/자동 생성, 가변 이미지/build, 잘못된 네트워크 등을 거부한다.

실제 컨테이너 시험은 **네트워크 none·새 시험 DB·발행기/재전송 worker 없음** 조건이다.
이미지와 선택 모델, outbox 저장소의 재생성 보존을 확인했으며 전체 main.py·카메라·GPU 추론·SDK/가상환경
로딩·브로커 인증 접속·최종 관제 ACK까지 실행한 시험은 아니다. 모델 해시 일치는 모델 실제 호출 합격과 다르다.
v0.88의 실제8DB 격리 복원 근거는 재사용했다. 복원된 운영 이벤트를 장치로 재전송하지 않았다.

### 읽기 전용 사전 검사 실행

프로젝트 루트에서 아래 명령은 Compose를 병합·검사할 뿐 컨테이너를 기동하지 않는다.
현재 검증한 Docker Compose는5.5.1이며 `!reset`/`!override` 지원이 필요하다.
비밀값이 포함된 병합 JSON은 파일·터미널에 출력하지 않고 검사기 표준입력으로 바로 전달한다.

```bash
rtk proxy env CCTV_CANDIDATE_MODEL_ROOT=/media/sawwave/Learning11/CCTV-project/data/runtime/p51_rollback_20261001_v089/candidate_models docker compose --env-file .env.jetson -f docker-compose.jetson.yml -f docker-compose.jetson-candidate.yml config --format json | rtk proxy .venv/bin/python scripts/health/check_compose_runtime_assumptions.py --candidate-compose-json - --json
```

원자료 `data/runtime/p51_candidate_20261001_v090`, [검증 근거](evidence/candidate_profile_20261001.json),
[회귀 JUnit](evidence/candidate_profile_20261001.xml).
다음은 새 프로필의 전체 AI 격리 통합 기동과 이전 구성의 SDK/가상환경/설정 의존성 점검이다.
I32/I34/I37/I38 미해결 조건·운영 전환 보류를 유지한다. 이번 검사를 출하 승인으로 사용하지 않는다.

<a id="rollback-preparation-v088"></a>

## v0.88 — I07 구성 차이·DB 복구와 I41 백업 누락 보완 (2026-10-01)

**이번 준비 완료:** 운영/후보의 명시적 환경·마운트·선택 파일 차이표, 현재 디스크351파일과
후보 모델8개 보존, 실제 SQLite8개 백업·격리 복원 및 전체 행/상태 대조.
설치된 정기 백업 스크립트의 전달 대기 DB 누락은 I41로 기록하고 보완했다.
AI 배포·기동 중인 코드의 재로딩·운영 DB 덮어쓰기·이벤트 재전송은 하지 않았다.

| 비교 대상 | 운영 조회 | 검증 후보 | 적용 전 의미 |
|---|---|---|---|
| 이미지 | `1ace5b7832a5…` | `809693da4ed0…` (v0.85) | 후보는 RTSP 종료 플러그인+Python 조합. 같은 이미지로 간주하지 않음 |
| 코드/설정 | 후보 목록 중345같음·6다름·7없음, 개인 얼굴 JSON 제외 | 이미지에 고정된359목록 | 실제 존재351개만 별도 보존. 시작 당시 메모리나 모든 bind mount 복구본은 아님 |
| 모델 | 대응8개 파일 해시 일치 | 8개92,405,483bytes 별도 보존 | 가상환경/호스트 SDK 의존성은 별도 |
| 명시적 환경 | 후보와138개 키의 유무/값 차이 | 시험용 키·경로·비활성화도 포함 | 138개가 모두 동작 차이라는 뜻은 아님. 빠진 키의 기본값 추정 금지 |
| 후처리/TCN | 새 주기·재사용 옵션 명시 없음 | 인물0.5초·TCN 재사용true·Pose640 명시 | 기존 메모리 실행과 비교한 성능 개선으로 단정하지 않음 |
| 영속 전달 | `MQTT_EVENT_OUTBOX_DB=/app/data/runtime/mqtt_event_outbox.db` | 시험 Compose/이미지 환경에 없음 | 운영용 후보 프로필에 필수 보존. 시험 환경 그대로 배포 금지 |
| 네트워크·DB | 운영 네트워크·공유 `/app/data` | network:none·시험 `/output` DB | 시험 격리 조건은 배포 설정이 아님 |

상이한6파일은 진행판 renderer, RF 학습 도구, 이미지에 있는 Alert/Action/Kuiper runner3개,
event type map이다. 후보의 장치/관제 runner7개는 현재 운영 AI 이미지 경로에 없었다.
현재 디스크/선택 목록의 비교이며, 전체 서비스 의존성이나 실행 중 import 상태를 확정한 것은 아니다.
비밀값은 비교 결과에 저장하지 않았고 `.env`·얼굴 등록 파일의 완전한 복구 묶음을 만들었다고 주장하지 않는다.

### I41 — 재전송 DB 백업 누락 수정과 회귀

기존 스크립트는 외형·명령/이벤트·검수·얼굴 DB4개만 백업해 실제 설정에 있는
MQTT/HTTP 재전송 DB가 빠졌다. 이벤트 대기/EdgeX 명령 결과도 포함해 표준 runtime DB8개를 명시했다.
읽기 전용 SQLite online backup·quick_check·완성 후 파일 전환, 신규700/600 권한,
`CCTV_RUNTIME_BACKUP_PRUNE=0` 선택을 보완했다. 기존7일 보관/기본 삭제 정책은 유지한다.
초기3실패(누락/삭제 방지/권한) 재현 후 **5 PASS**, 손상 원본·기존 보관 동작도 검증했다.
설치된 systemd 서비스가 이 파일을 직접 호출하므로 다음 정기 실행부터 목록8개를 사용한다.
새 백업8.86GB(약8.25GiB)로 저장·I/O 대상이 늘었으며 운영 DB 자체를 압축/정리하지 않았다.

### 실제 격리 복원 결과

| DB | 업무 행 수 | 복원 판정 |
|---|---:|---|
| action_events.db | 이벤트922799·명령5649 | 스키마·전체 행·명령 상태 동일 |
| action_http_outbox.db | 83691 | pending1·sent83690 보존 |
| appearances.db | 531661 | 외형 전체 행 동일·검색 조건0 |
| mqtt_event_outbox.db | 173735 | pending0·현재 저장소 코드로 조회 가능 |
| commercial_faces.db | 2 | 사람/표본 각1, 데이터는 비공개 백업 경로에만 보존 |
| edgex_command_results.db | 28 | 응답 상태 보존 |
| event_reviews.db | 1 | 검수 상태 보존 |
| event_outbox.db | 0 | 빈 큐/스키마 정상 복원 |

총 **1,717,566 업무 행**, sqlite_sequence 포함 스키마/모든 테이블 내용 해시·상태 분포 대조,
복원8개 integrity_check 모두ok. 미사용 페이지를 제외해 복원한4.486GB와 백업8.856GB의
파일 크기는 다르지만 논리 데이터는 같았다. MQTT 저장소 조회는 publisher/worker를 생성하지 않았고
복구된 과거 명령을 장치로 실행하지 않았다. 실제 pending0은 합성 WAL 대기행 회귀와 구분한다.

DB별 스냅샷 시각이 달라 서비스 전체 같은 시점 복구는 미검증이다. 백업 이후 기록,
외부 ACK/물리 출력 상태, 구버전 코드 전체 기동·PostgreSQL/Redis·비밀값/전체 마운트 복구도 미검증이다.
운영AI healthy/restart0·09-17 시작 시각 유지. I41 FIXED_LOCAL, **I07 OPEN·P5.1 IN_PROGRESS** 유지.
다음은 운영용 후보 프로필의 영속 DB/비밀값/마운트 조건을 구체화하고 격리 환경에서 확인하는 것이다.
I32/I34/I37/I38 잔여 조건을 해결하지 않은 운영 전환은 보류한다.

[정량 근거·해시·차이표](evidence/rollback_preparation_20261001.json) · [회귀 JUnit](evidence/backup_restore_20261001.xml)
· [복구 런북](../handover/BACKUP_RESTORE_RUNBOOK.md). 원자료 `data/runtime/p51_rollback_20261001_v089`.

<a id="operating-deployment-20260930"></a>

## 운영 배포 확인 — 2026-09-30 16:44 KST

사용자 질문에 따라 Docker 상태와 운영 `/metrics`를 다시 읽었다.
확인 당시 상시 서비스33개와 격리 시험1개가 실행 중이었다.
`cctv-ai-engine`은09-17 08:45:46 KST 시작 이후 약13일8시간 실행 중이며
컨테이너 생성09-17 08:45:39 KST·Docker restart count0·healthy였다.
현재 운영 이미지 ID는 `sha256:1ace5b7832a50d6675b66146b0edc307021795732fc7c9a770e2cd628d7fd261`다.
계획 수립 후 이 운영 컨테이너를 재생성한 배포는 없으며, 새 후보는 같은 기반 이미지와 별도 소스 마운트로 시험했다.

**재배포하지 않았다는 사실과 모든 파일이 옛 버전이라는 주장은 다르다.**
실행 컨테이너의 `/app/src`, `/app/scripts`, `/app/config/deepstream`, `/app/models`는 로컬 폴더의 bind mount다.
로컬 수정은 컨테이너 파일시스템에도 보이지만, Python에 이미 로드된 모듈은 자동으로 모두 갱신되지 않는다.
늦게 import하거나 자식 프로세스/설정 재로딩으로 읽는 파일은 새 버전을 사용할 수 있다.
따라서 현재 메모리의 전체 코드 구성을 과거 이미지나 현재 디스크와 동일하다고 단정하지 않는다.

운영 metrics HTTP200에서 기존 `cctv_processor_frames_processed`는 존재했으나
`cctv_camera_primary_`, `cctv_model_`, `cctv_falldata_temporal_`,
`cctv_processor_context_tasks_completed`는 모두 없었다. 새 계측의 운영 활성화는 확인되지 않는다.
관련 환경에는 `DS_PRIMARY_INTERVAL=1`만 명시됐고 `DS_CONTEXT_MIN_INTERVAL_SEC`,
`FALLDATA_AUX_TEMPORAL_WORKER_ENABLED`, `DS_OSD_PROCESS_MODE`는 없었다.
환경값이 없다는 것만으로 실제 실행 기본값을 추정하지 않는다.

### 반영 상태와 전환 조건

2026-09-30 사용자 지시: **알려진 문제를 해결·검증한 뒤 배포한다.**
컨테이너가 오래 실행됐다는 이유만으로 재기동하거나, 오류가 재현되지 않은 시험만으로
원인 미확정 항목을 해결 처리하지 않는다. 다음 순서로 진행한다.

1. I40 종료 지연(10-01 v0.85 소스·플러그인 조합 후보에서 FIXED_LOCAL), I34 CUDA700, I32 영상/분석 결과 정합과 후보에서 사용하는
   RF/TCN의 I37/I38 잔여 조건을 확인한다. 결함 수정과 해당 실패 조건의 회귀 근거를 연결한다.
2. 검증한 소스·모델·환경을 고정하고, DB 호환 및 이전 구성 복구를 격리 환경에서 확인한다.
   이미지뿐 아니라 bind mount 파일도 복구 대상이다.
3. 위 조건을 충족한 후보를 AI 서비스 한 개에 반영한 뒤 새 계측·입력/출력·저장을 확인한다.
   I07의 실제 운영 반영 확인은 이 단계에서 판정한다. 배포 전 준비와 배포 후 검증을 구분한다.

다중 카메라 성능·전체 탐지 품질·실물 장치 시험은 기존 P5/P6 및 P3/P4의 완료 조건을 유지한다.
단일 AI 서비스의 수정 배포를 전체 제품/현장 POC 합격으로 표시하지 않는다.

| 구분 | 현재 상태 |
|---|---|
| 시간/좌표/인물 분리, 후처리 큐·RF/TCN·종료 보완 | 로컬 수정과 격리 검증을 진행. 운영 반영·전체 활성 코드 일치는 미검증 |
| 새 모델별 계측 | 후보에서 관측, 운영 metrics에서는 미노출 |
| WBS·진행판 | 작업 종료마다 갱신. 문서 갱신이 AI 컨테이너 배포를 뜻하지 않음 |
| 다음 배포 대상 | `edgex-jetson`의 `cctv-ai-engine` 한 서비스. EdgeX·브로커·장치 서비스까지 일괄 갱신하지 않음 |
| 아직 필요한 조건 | 10-01 보완: I40 조합 후보 로컬 검증 완료·운영 미배포. I34/I32 및 선택 모델 조건, 실행 마운트/환경 차이표, 고정 소스·모델과 검증된 복구 구성, DB 호환/복구 절차 잔여 |

이미지 태그만 되돌려도 bind mount의 소스는 이전 버전으로 돌아가지 않는다.
현재 작업 디렉터리 전체를 그대로 재기동하는 대신, 적용할 파일/설정과 복구본을 구체적으로 고정해 검토해야 한다.
운영 전환은 이 자료가 준비된 뒤 대상 서비스와 중단 범위를 명시해 진행한다.
이번 확인에서는 운영 컨테이너·환경·마운트·DB를 변경하지 않았다. I07 OPEN 유지.

---

> v0.87: [일반 실행 매칭·TCN 대조](MODEL_INFERENCE_METRICS.md#match-tcn-v087). 기준180초·단절/복구 시험 정상 종료, CPU 영상1497개 전부 등록 메타와 대응·후처리327/정상 제한1170. TCN18회 반환(유효17·Pose 부족1), timeout0·재연결8.23초. 과거 I32/I38 원인은 미확정·I34 OPEN·운영 미배포. 다음은 I07 고정 후보·이전 구성의 차이표와 격리 복구 준비다.

> v0.86: [CUDA 할당 경계·수명 검사](TRACKER_VALIDATION.md#cuda-bounds-v086). padding32·비동기 할당 수명 검사를 추가해 원 보존본/고정 후보 각60프레임·정상 종료·메모리 오류0을 확인했다. 원 CUDA700은 미재현·I34 OPEN. 검사 중 TCN timeout과 매칭 누락은 별도 보존, 제품 코드·운영 배포 변경 없음. 다음은 일반 실행의 I32 매칭 사유·I38 상태 대조, 이후 I07 복구 준비다.

> v0.85: [RTSP 종료 재활성화 수정·후보 이미지 검증](TRACKER_VALIDATION.md#rtsp-shutdown-fix-v085). 출력 bin이 종료 중 다시 열리는 결함 재현·상태 잠금 보완, 관련36 PASS. 공식 RTSP 보완을 포함한 조합5조건 종료1.62~5.27초·복구6.36초, 영상6파일 디코드/원본 DTS 정상. 후보 이미지 고정·I40 FIXED_LOCAL, I34 CUDA700/I32/I07 잔여·운영 미배포.

> v0.84: [출력 장애 종료 대기 분리](TRACKER_VALIDATION.md#output-shutdown-v084). 진단 없는 정상 종료6.25초·출력 서버만 단절하면30초 초과 재현. RTSP caps 대기 native 스택 확보·공식1.20.2 패치 격리 빌드. 수정본 일반 실행의 전체 NULL 대기가 남아 I40 OPEN·배포 보류. CUDA700 해결 근거와 구분.

> v0.83: [종료 시간 초과 재현·순서 보완](TRACKER_VALIDATION.md#shutdown-recovery-v083). 인코더 버퍼 풀 회수 대기 native 스택 확보·RTSP 출력 우선 종료로 최소 수정. 실제 후보 정상/입력 단절 종료7.23/2.49초·46 PASS. I40 추가 회귀·I34/I32/I07 잔여, 문제 해결·검증 전 배포 보류.

> v0.82: [종료 단계·운영 배포 대조](TRACKER_VALIDATION.md#shutdown-stages-v082). 종료 상태 전환7.47초/보조 회수0.34초 분리·추가2회 정상이나 최초30초 초과 미재현. native 스택 미확보·I40 OPEN. 운영AI09-17 이후 재배포 없음·새 계측 미노출, bind mount와 실행 코드 차이 명시.

> v0.81: [현재 후보 모델 실행 대조](MODEL_INFERENCE_METRICS.md#model-state-v081). 격리90초×3회·모델 실행/무결과/미호출 구분. 얼굴 Haar 검출 실행/결과0, RF 미호출·Python 속성 Null 경로와 SGIE 구분. 종료30초 초과1회·재시험2회 정상이나 I40 OPEN. 영상3파일 디코드·PTS/DTS 대조 통과·운영 미반영.

> v0.80: [CUDA 최초 조건 대조·GPU 덤프 준비](TRACKER_VALIDATION.md#cuda-comparison-v080). 원 보존본 비동기/동기 각120프레임 통과·실제 적재 영상 라이브러리4개 호스트 해시 일치. 예외 덤프 선택 기능·최초719 기록 보완/12 PASS·수동 덤프719는 원700과 구분. I34 OPEN·운영 미반영.

> v0.79: [단절 후 녹화 시간 대조](TRACKER_VALIDATION.md#match-timing-v079). 같은 출력의55/90초 수신기를 동시 대조·2회 확인. 약72초의 복구 프레임이55초 제한을 넘어 짧은 파일 생성;90초 파일에는 복구 후 영상 포함. 4파일 디코드 정상·매칭 누락 재현0. 원 매칭/CUDA 원인 미확정·I32/I34/I07 OPEN·운영 미반영.

> v0.78: [현재 후보 종료 SIGABRT 수정](TRACKER_VALIDATION.md#candidate-shutdown-v078). 보조 영상 쓰기 중 2초 대기 후 종료하는 결함 재현·native/Python 스택 확인. 작업 완료 대기·모델 자식 회수, 256 PASS/2 SKIP. 인코더 포함/제외·재연결 중 종료·실제 RTSP 재시험 완료. I39 FIXED_LOCAL·I34 OPEN·운영 미반영.

> v0.77: [입력 감시·NVDEC/NVENC·네이티브 대조](TRACKER_VALIDATION.md#source-boundary-v077). 무수신 경고 순간에도 디코더/추론은 직전0.05초까지 진행·큐21개 관측. 검사 전용 재연결60초는40프레임 목표/메모리 오류0. 네이티브 RTSP 대조는 EOS 개입으로 원인 판정 제외·파일 경로 별도 확인. 9 PASS·I34 OPEN·운영 미반영.

> v0.76: [CUDA 최초 오류 보존·RTSP 메모리 검사](TRACKER_VALIDATION.md#cuda-capture-v076). 17개 입출력 지점·최근4096기록 보존, 합성 오류/과거 로그 수집 확인·7 PASS. 시작3조건 CUDA700 미재현. 메모리 검사2회 시간초과를 보존, 검사 전용 자동 재연결 해제 후40프레임·최종 memcheck 오류0. I34 OPEN·운영 미반영.

> v0.75: [GPU/VIC 비교·P5.1 종료 점검](VIDEO_INFERENCE_CONTRACT.md#p51-closeout-v075). 각120프레임×2회 반복 결과 일치, 장치 간 관절 최대81px/유효점 경계119개 차이로 VIC 전환 보류. GPU 후보359파일 고정·단절 후8.07초 복구. 10-01 종료 검토 목표·I34/I07 등 미해결로 DONE 확약 불가.

> v0.74: [CUDA 시작 경로 분리](TRACKER_VALIDATION.md#cuda-startup-v074). 보존본2회/현재 후보2회·Primary VIC1회 시작/35초 관측 통과. GPU 후보120프레임 메모리 검사 오류0·출력5파일 디코드 정상. Primary 전처리와 후단 복사 설정 구분·I34 근본 원인 미확정·운영 미반영.

> v0.73: [TCN 재사용 동시 부하600초](MODEL_INFERENCE_METRICS.md#tcn-loaded-v073). 녹화1채널·TCN47회 완료, 첫12.92초/후속1.40~3.13초. timeout·후처리 큐 폐기/실패·CUDA 서명0. Pose SGIE/RF·얼굴 추론·관제 ACK 미검증, 출력 간격/I34 원인 확인 잔여·운영 미반영.

> v0.72: [TCN 전용 모델 재사용](MODEL_INFERENCE_METRICS.md#tcn-worker-v072). 선택형 자식 프로세스 재사용·오류/timeout 종료·새 요청 복구. Jetson 동일 영상 단발11.26~12.42초→준비 후1.16~1.60초, 결과 일치. 영상 교체/자식 강제 종료 복구·194개 고유 테스트 통과. 기본false·운영 미반영·I38 원timeout/장시간 및 I37/I32/I34/I07 잔여.

> v0.71: [RF 표본 선택 기준 보존](MODEL_INFERENCE_METRICS.md#rf-sampling-contract-v071). 캡처에 고유 균등 표본 정책/실제 최대 개수 기록·학습48개 고정 제거. 미상/혼합 기준 거부·205 PASS, 라벨 도구 포함 통합2건 재확인. 원본 탐색 범위에서 복원 못함·운영 미반영. I37 점수/원본·I38/I32/I34/I07 잔여.

> v0.70: [RF 수집 시간창 보존](MODEL_INFERENCE_METRICS.md#rf-window-v070). 학습 시 수집 시간창 무시·3초 고정 결함 수정. 누락/비정상/혼합 시간창 거부, 수집→학습→재로딩→추론 합성 통합 포함155 PASS. 기존 후보 원본5개는 기록 경로에 없어 학습 조건 검증 보류. 운영 미반영·I37/I38/I32/I34/I07 잔여.

> v0.69: [RF 학습 출처 검사](MODEL_INFERENCE_METRICS.md#rf-source-v069). 이름/개수만 같은 다른 출처 모델의 inline 예측 차단. 기존 fallback 정책 유지, 현재 RF 전환 영향·후보 검증 잔여. 운영 미반영.

> v0.68: [RF 중복 표본 수정과 TCN 계측](MODEL_INFERENCE_METRICS.md#fall-sampling-v068). 실제 기록 수만큼 RF 입력을 구성하며 인물/세대/최소 Pose 보호 유지. I37 전체·I38 기존 timeout 원인은 OPEN.

> 2026-09-30 v0.67: [낙상 전처리·Pose 재점검](MODEL_INFERENCE_METRICS.md#fall-contract-v067). TCN 입력640 명시 전달 수정, RF 특징 임계값/시간창 불일치 I37 및 보조 추론 timeout I38은 추가 검증. 운영 미반영.

> 최신 v0.64: [실CCTV·CUDA 최소 경로 검사](TRACKER_VALIDATION.md#v064--실cctv와-cuda-최소-경로-검사). RC4 실CCTV VIC600초/GPU180초·CUDA 오류 미재현. 단독 Primary와 최소 경로 메모리 검사 결과 확보. VIC 후처리 큐 폐기1535건은 I32 잔여. 최초 CUDA 원인 미확정·I34 OPEN·운영 미반영.

> 09-28 v0.39: 실제 CLI의 2입력 단절 시험과 선택 RTSP 재연결을 확인했다. 기본 미복구·3초 실패·15초 재개 결과는 10절이다. 운영 설정은 변경하지 않았다.

> 09-28 v0.38: 보존본의 실제 격리 영상 실행과 별도 복원본 재실행을 확인했다. 상세 결과·한계·재현 명령은 9절이다.

> 09-28 v0.37: Pose 수치 경계 보완(I25 FIXED_LOCAL), 197 PASS/2 SKIP. 적용 준비·복구 제한은 아래 8절에 기록했다. 1~7절의 실행 관측은 최초 기준선이다.

> 09-22 v0.31 후속: [I24 인물별 자세/캐시 분리](POSE_TRACK_ISOLATION.md)를 로컬 수정했다(291 PASS/5 SKIP). 아래 실행 대조와 최초 혼합 재현은 수정 전 기준선으로 보존한다. 실제 인물 추적/운영 반영·전체 입력 세대는 잔여다.

2026-09-22 KST · 계획 v0.29 · **첫 실행 단위 완료 / P5.1 IN_PROGRESS**

현재 실행 상태와 로컬 소스·설정·모델 파일을 대조하고, 입력/결과 계약과 다음 계측 범위를 정했다.
기존 관련 회귀 **189 PASS / 5 SKIP**다. 실제 모델별 완료율, 입력 단절·복구, 인물별 시간창의
구현과 실측이 남아 P5.1 전체를 DONE으로 처리하지 않는다. GPU 시험·서비스 재시작·장치 출력은 하지 않았다.

## 1. 이번에 확인한 사실

| 확인 대상 | 관측 | 판정 범위 |
|---|---|---|
| 실행 프로세스 | `cctv-ai-engine` running, 시작 09-17 08:45:46 KST, Docker restart count 0 | 컨테이너 수명이다. 내부 파이프라인 재생성 횟수는 증명하지 않음 |
| 영상 입력 | Stream API 200, 등록 카메라 `camera_1` 1개. 파일상 enabled RTSP 1개 | `/health`의 ok는 카메라 실제 신선도/추론 합격을 검사하지 않음 |
| 프레임 진행 | 서로 다른 두 scrape에서 `frames_processed` 13,313,358 → 13,316,224 | 처리 진행 확인. 전체 모델 완료 FPS/촬영 시각 지연 아님 |
| 기존 FPS | 첫 scrape 약 29.93 | 처리된 프레임 기준이며 카메라별 모델 완료 근거로 사용하지 않음 |
| 새 Primary 계측 | 두 응답 모두 `cctv_camera_primary_*`와 `primary_metrics_errors` 미노출 | **I07 OPEN: 계측 운영 적용을 확인할 수 없음** |
| 코드 반영 경계 | processor·pipeline builder·Primary metrics·exporter 파일 수정이 컨테이너 시작보다 늦음 | 디스크 소스와 메모리에 로드된 코드가 같다고 판단하지 않음 |
| 추론 소요시간 표시 | `avg_inference_ms`와 `yolo_postprocess_avg_ms`가 모두 3.82 | exporter가 후처리 값을 복사하는 기존 별칭. GPU 추론 시간 아님 |
| 입력 시간창 | inline Pose-RF의 카메라별 버퍼에서 인물 101→202→101 혼합 재현 | **I24 OPEN**. 합성 입력 버퍼 시험이며 실제 분류 오탐/장치 오작동을 측정한 것은 아님 |

근거: [런타임·파일 목록](evidence/video_runtime_inventory_20260922.json),
[첫 scrape](evidence/video_runtime_metrics_before_20260922.prom),
[두 번째 scrape](evidence/video_runtime_metrics_after_20260922.prom),
[인물 혼합 재현](evidence/pose_track_gap_20260922.json).
인증값·카메라 접속 URI·영상은 보고서에 저장하지 않았다.

## 2. 모델·전처리 대응

아래는 **컨테이너 환경과 현재 디스크 설정/코드**의 대응이다. 살아 있는 GStreamer 요소의
속성이나 GPU에 로드된 엔진의 해시를 직접 읽은 결과는 아니다. 엔진 파일 존재·SHA-256과
실행 중 엔진 동일성은 구분한다. 환경값을 근거로 실제 처리율을 계산하지 않는다.

| 경로 | 현재 설정상 모델·입력 | 전처리·주기 | 미확인 사항 |
|---|---|---|---|
| Primary Pose/Person, GIE 1 | `yolov8n-pose.engine`, CHW 3×640×640 | RGB, 1/255, 종횡비 유지·대칭 padding. Primary interval 환경값 1 | 실제 요소 속성·카메라별 tensor 완료·엔진 실행 동일성 |
| Helmet/Head, GIE 2 | `helmet_model.engine`, 3×320×320, GIE 1 ROI | RGB, 1/255, 종횡비 유지·대칭 padding. batch 환경값 1, interval 설정 시도 1 | 실제 ROI 처리·추론/추적 결과 구분·간격 적용 범위 |
| Appearance, GIE 3 | `config_infer_pa100k.txt` → `pa100k_resnet50_attr.engine`, 3×256×192 | RGB, scale 0.0171247538, offsets 123.675/116.28/103.53. batch 환경값 1, interval 4/reinfer 15 설정 시도 | 모델명에 PP-Human 변수명을 쓰지만 선택 파일은 PA100K. ROI 완료·전처리 학습 일치·캐시 나이 |
| 색상 | `appearance_color_yolov8n.engine` 환경값 | Python 외형 분석 경로. nvinfer GIE 완료와 별도 계측 | 실제 로딩/호출·crop 시각 일치·색상별 품질 |
| 낙상 보조 RF/TCN | 환경의 비교 모델 경로·카메라별 override 및 temporal 모델 | inline Pose-RF enabled, mode shadow, confirm-borderline true | 인물별 시간창·간격/결측·실제 로딩과 호출·판정 적용 경계 |
| crop-Pose | `DS_CROP_POSE_ENABLED=0` | 별도 Person Primary/Pose SGIE 구성 비선택 | 실험용 GIE 1~4 레지스트리를 현재 운영 구성으로 사용하지 않음 |

세 nvinfer 파일의 batch-size는 8이나 코드가 Primary는 카메라 수, SGIE는 환경값으로
덮어쓴다. 파일 숫자만으로 실제 batch 8이라고 기록하지 않는다. 현재 정밀도도 파일명/과거
manifest만으로 확정하지 않으며 실행 엔진 식별·shape 확인에 연결한다.
`shadow`와 `confirm-borderline=true`가 공존하므로 shadow라는 이름만으로 모든 경보 판정에
영향이 없다고 판단하지 않는다. 기존 판정 연결은 P5.2와 함께 검증한다.

NVIDIA 7.1 기준 interval은 건너뛸 배치 수이며 파일 표에서는 Primary 적용으로 구분한다.
SGIE에는 ROI·재추론/캐시 조건이 있으므로 환경값 4를 곧바로 매 5프레임 실제 추론으로 환산하지 않는다.
Primary raw tensor는 frame user metadata, Secondary는 object user metadata에서 관측한다.
[공식 nvinfer 설정·tensor 설명](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvinfer.html).

## 3. 입력·결과 계약

아래 이름은 후속 구현을 위한 논리 필드다. 이번에 API·DB schema·기존 wire를 변경하지 않았다.
현재 `DetectionEvent`·`video_observation`을 확장할 때 기존 timestamp/ID 의미를 보존하고
선택 metadata로 추가한 뒤 저장·관제 소비자와 회귀한다.

| 항목 | 규칙 | 현재 코드 차이 |
|---|---|---|
| 카메라 식별 | 설정 camera ID 사용. `source_id`/pad index는 해당 파이프라인의 매핑 정보 | 현재 `_pad_to_camera`로 연결. source ID를 영속 장치 ID로 단독 사용하지 않음 |
| 프레임 식별 | `(camera_id, stream_epoch, frame_num)`로 구분. 재연결·재생 rewind 시 epoch 변경 | frame_num은 있으나 재생/연결 세대 ID의 일관된 전파는 없음 |
| 인물 식별 | `(camera_id, stream_epoch, tracker_epoch, track_id)` 범위. unknown을 임의 인물 0으로 합치지 않음 | object_id는 있지만 세대 구분이 없음. inline Pose-RF 버퍼는 인물별로 분리되지 않음 |
| 영상 시각 | source PTS는 원래 단위/출처와 함께 보존. 촬영 wall-clock은 검증된 동기화 근거가 있을 때만 사용 | 이벤트 timestamp는 생성 시 `time.time()`. PTS/NTP가 이벤트에 전달되지 않음 |
| 공통 관측 시각 | `processed_at`와 `received_at` 유지. 촬영 시각이 없으면 `observed_at=null` | 현재 `video_observation`의 missing 처리가 올바른 한계 표시이며 수신 시각으로 덮어쓰지 않음 |
| 지연 | 같은 프로세스 구간은 monotonic 차이. PTS·UTC·monotonic 간 직접 뺄셈 금지 | 후처리 평균·마지막 프레임 나이·전체 지연을 구분해서 수집해야 함 |
| 좌표·전처리 | bbox/keypoint가 어떤 영상 해상도/ROI 좌표인지 기록. crop·resize·padding 역변환을 검증 | frame_width/height는 있으나 모델 입력과 원본/ROI 대응을 모델별로 확인해야 함 |
| Pose 품질 | 17×3 keypoint와 confidence/유효 점 개수 확인. 부족한 관절을 정상 자세로 대신하지 않음 | shape 일부 검증은 있으나 시간창 전체 품질/결측 판정의 통합 계약 미연결 |
| 속성 품질 | 미검출·미지원·판정 불가는 false/미착용과 구분. 캐시 결과는 원래 산출 시각/나이 보존 | I16·외형 캐시 후속 검증. 새 프레임의 신규 추론으로 재사용 결과를 세지 않음 |
| 모델 식별 | task/GIE ID뿐 아니라 모델 artifact hash·전처리/라벨 버전·실행 구성 식별을 연결 | GIE/task 일부 존재, 실제 모델 버전 전파와 실행 증거 부족 |
| 분석 상태 | disabled, no_target, insufficient_input, cached, completed, error를 구분 | 단순 프레임 수/이벤트 발생 수로 분석 완료를 대신할 수 없음 |
| 저장/출력 | P2 공통 접수·관제 계약 재사용. 품질 부족/단절을 정상 관측이나 새 물리 경보로 만들지 않음 | 현재 선택 영속 경로와 운영 비영속 경로 적용 여부는 별도 I12/I17 |

`frame_num`, `buf_pts`, `ntp_timestamp`, `source_id`의 SDK 필드는
[NVIDIA NvDsFrameMeta](https://docs.nvidia.com/metropolis/deepstream/7.1/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html)를
기준으로 구분했다. PTS를 Unix 촬영 시각으로 해석하지 않는다.

## 4. 인물별 시간창과 장애 경계

- 인물별 동작 분류의 창에는 동일한 인물·동일한 연결/추적 세대의 관측만 들어간다.
  현재 카메라 단위 RF가 의도한 장면 수준 요약을 유지할지, 사람별 판정으로 바꿀지는
  학습 입력/라벨과 대조한다. 여러 사람의 특성을 섞은 결과를 특정 인물 판정으로 사용하지 않는다.
- I24 재현은 매 프레임 두 사람을 넣고 최고 fall_score 인물을 교대로 바꾸면 하나의
  카메라 버퍼에 101→202→101이 저장되고 track ID도 사라지는 동작을 확인했다.
  현장의 실제 오탐률이나 영향도를 측정한 결과는 아니다.
- 프레임 rewind, PTS 역행/불연속, 재연결, tracker reset에서는 해당 시간창·속성 캐시를
  구분/초기화한다. 영상 끊김 구간을 직전 정상 자세의 반복으로 채우지 않는다.
- 최소 유효 관절/프레임 수, 최대 간격, 시간창 길이는 모델 학습 조건과 D07에서 확정한다.
  이 계약만으로 임의 숫자를 운영 기본값에 넣지 않는다.
- 현재 소스 ERROR/EOS는 전체 파이프라인 재시작 요청으로 연결될 수 있다(I20).
  정상 카메라가 계속 처리되는지는 격리 2입력 장애 시험에서 확인해야 한다.

## 5. 계측 위치와 다음 구현 단위

| 경계 | 남길 근거 | 합격으로 오해하면 안 되는 값 |
|---|---|---|
| source decode/mux 입력 | 카메라·연결 세대·frame/PTS·도착 monotonic·last-seen | `/health=ok`, 미리보기 FPS |
| Primary 출력 | 고유 GIE의 tensor 있는 프레임/없는 프레임/판정 불가, 오류·초기화 세대 | 전체 후처리 `frames_processed`, bInferDone 단독 |
| SGIE 출력 | 카메라/GIE별 대상 ROI·새 tensor 결과·no-target·unknown·cache reuse | 사람 수가 다른 ROI 처리율을 프레임 FPS로 환산 |
| 낙상/외형/색상 후처리 | 트랙/시간창 ID·품질 상태·실행/캐시·완료 monotonic | 이벤트가 없다는 이유로 모델 미실행/정상 완료 단정 |
| 공통 저장·관제 | 기존 ingest/record/delivery ID와 commit/ACK 계측 | GPU 추론 완료를 관제 수신/장치 출력 성공으로 사용 |

**후속 GIE 메타 계측 구현은 [별도 결과](MODEL_INFERENCE_METRICS.md)로 완료했다(217 PASS/5 SKIP).** 아래는 최초 실행 순서이며, 실제 캐시/ROI·Python 모델 계측·운영 반영은 잔여다. 다음 작업은 I24 입력/시간창 보완이다.

최초 실행 단위는 P5.1의 모델별 계측 보완이다.
기존 Primary 계측과 단위 테스트를 재사용해 GIE별 실제 속성·관측 가능 여부·모델별 완료 근거를
작게 추가하고, 파일 존재·환경 설정·실제 실행 증거를 구분한다. 모델/추론 간격은 바꾸지 않는다.
그다음 I24를 인물별 학습/추론 입력 계약과 함께 보완한다. 변경/회귀가 준비되면 운영 적용 파일,
재시작 범위·복구본·관찰 항목을 제시하고, 허용된 점검 범위에서 한 서비스씩 반영한다.

## 6. 인수 시나리오와 검증 결과

| ID | 시나리오 | 기대 결과 | 이번 근거 |
|---|---|---|---|
| V01 | 모델 파일만 있고 runtime 완료 지표 없음 | 실행/성능 미확인 | 운영 두 scrape·파일 대조 |
| V02 | bInferDone false이나 Primary tensor 존재 | 해당 Primary 프레임 1회 완료 | 기존 단위 회귀 PASS |
| V03 | 다른 GIE tensor 또는 잘못된 metadata | 다른 모델 완료로 세지 않고 구분 | 기존 단위 회귀 PASS |
| V04 | source ID 매핑/재등록·카메라 단절 | 원래 카메라와 오류/관측 없음 구분 | 모형 회귀; 실물 단절 미실행 |
| V05 | object_id 미추적 sentinel, 촬영 시각 없음 | track unknown/missing, observed_at null | 기존 영상 관측 회귀 PASS |
| V06 | 두 사람의 최고 점수가 프레임마다 교대 | 인물별 창 유지 또는 장면 수준으로 명시 | **현재 I24 미충족 재현** |
| V07 | frame rewind·연결 세대 변경 | 오래된 시간창을 새 세대에 결합하지 않음 | frame rewind 기존 모형 PASS, 세대 전파 후속 |
| V08 | ROI 없음·추론 disabled·캐시 재사용 | 서로 다른 상태, 신규 완료로 합산하지 않음 | [GIE 메타 모형 검증](MODEL_INFERENCE_METRICS.md) 완료; 실제 ROI 필터·캐시 판정은 잔여 |
| V09 | 전체 운영 AI 1채널 정상/품질 부족/끊김 | 모델별 카메라·트랙·시각·완료/실패 대조 | 미실행, 이번 성능 합격 없음 |

```bash
rtk proxy .venv/bin/python -m pytest tests/test_deepstream_primary_metrics.py tests/test_deepstream_element_config.py tests/test_deepstream_pipeline_builder.py tests/test_deepstream_event_factory.py tests/test_deepstream_processor.py tests/test_video_observation.py tests/test_falldata_aux.py -q -o addopts=--tb=short -o junit_family=xunit1 --junitxml=docs/planning/evidence/video_inference_baseline_20260922.xml
rtk proxy .venv/bin/python -m docs.planning.evidence.reproduce_pose_track_gap_20260922
```

**189 PASS / 0 FAIL / 5 SKIP, 18.37초.** 5 SKIP은 로컬 `.venv`에 pyds가 없어
실제 DeepStreamProcessor 생성이 필요한 기존 시험을 건너뛴 것이다. 운영 컨테이너의 pyds
설치 여부와는 별개다. 모형 회귀 통과를 GPU/실물 카메라 합격으로 확장하지 않는다.
[JUnit](evidence/video_inference_baseline_20260922.xml) · [종합 검증](evidence/video_inference_contract_validation_20260922.json).

## 7. WBS 판정

P5.1 **READY → IN_PROGRESS**. 첫 단위인 상태 대조·입력 계약·계측 위치·회귀 기준선 정리를 완료했다.
모델별 실제 완료 계측·I07 운영 반영 증거·I24 입력 경계·단절/품질 부족 시험이 남아 DONE이 아니다.
P2.4는 내부 시험 근거를 유지하고 D02/D04/D06의 실부하·보관/지연 기준 대기로 **BLOCKED**로 옮겼다.
해제 조건은 해당 기준 합의 후 실제 부하/장시간 시험 재개다. P3 선행 조건을 삭제하지 않는다.
WBS 완료는 **9/33, 작업 수 기준 27.3%**이며 전체 기능 구현률이 아니다.


## 8. v0.37 — 입력 수치 경계와 운영 적용 검토 묶음

### 로컬 보완·검증

`FallDataAuxVerifier.add_pose_events`는 기존에 17×3 shape만 검사했다. NaN·무한대와 잘못된 크기가
버퍼에 들어갔고, 깨진 배열/해상도 변환 예외는 같은 배치의 정상 인물 처리까지 중단했다.
16개 실패를 재현한 뒤 유한 수치, confidence 0~1, 양수 bbox/영상 크기와 변환 오류 처리를 추가했다.
잘못된 PERSON에는 동일 bbox의 정상 인물에서 얻은 `falldata_pose_scope`도 연결하지 않는다.
확인되지 않은 사람의 입력을 RF의 유효한 시간창으로 취급하지 않기 위한 수정이다.

변경은 `src/core/ai/_falldata_aux.py`와 기존 `tests/test_falldata_aux.py` 두 파일이다.
입력을 억지로 1×1 해상도로 보정하지 않고 제외한다. 원본 DetectionEvent를 삭제하거나 모든 경보를
막는 변경은 아니다. 기존 보조 판정의 unavailable/fail-open 정책은 유지하며 실제 출력 정책은 별도로 검증한다.
낮은 관절 confidence의 유효 점 개수, 최대 프레임 간격·PTS 불연속·실제 추적 ID 오류는 이번에 해결한 범위가 아니다.
기존 0.35 관절 계산 기준이나 모델별 최소 프레임 설정을 새 값으로 바꾸지 않았다. D07 및 학습 조건과 대조한다.

- 신규 19개: 잘못된 수치/배열/크기 18개, 같은 bbox의 잘못된 인물 scope 연결 차단 1개.
- 인물 분리·낙상 정책·processor 회귀 포함 **197 PASS / 0 FAIL / 2 SKIP** (11.06초).
- SKIP 2개는 DeepStream 미설치 환경 전용. GPU·실영상 검증은 아니다. Ruff·컴파일 통과.
- [JUnit](evidence/input_quality_20260928.xml). 수정 전 두 파일은 `data/runtime/input_quality_20260928/`에 보존했다.

```bash
rtk proxy .venv/bin/python -m pytest tests/test_falldata_aux.py tests/test_falldata_pose_tracks.py tests/test_fall_aux_policy.py tests/test_deepstream_processor.py -q -rs -o addopts=--tb=short -o junit_family=xunit1 --junitxml=docs/planning/evidence/input_quality_20260928.xml
```

### 적용 대상·보존본·복구 제한

09-28 읽기 전용 Docker inspect 결과:

| 항목 | 확인/준비 결과 |
|---|---|
| 대상 서비스 | `cctv-ai-engine`, `docker-compose.jetson.yml` |
| 실행 시작 | 2026-09-17 08:45:46 KST, running. 이번에 재시작하지 않음 |
| 현재 이미지 | `sha256:1ace5b7832a50d6675b66146b0edc307021795732fc7c9a770e2cd628d7fd261` |
| 소스 경계 | 호스트 `src` → `/app/src` bind mount 확인. Compose에도 `scripts` bind mount 정의 |
| 검토 보존본 | `data/runtime/p51_review_20260928_v037/candidate_python_sources.tar.gz` |
| 파일 목록/해시 | 같은 폴더 `manifest.json`, 현재 src/scripts Python 321개. tar 항목별 SHA-256 대조 통과 |
| 제외 | 모델·비Python 설정·환경 비밀값·DB·실행 중 메모리 상태 |
| 운영 반영 | 미실행. 위 보존본은 현재 후보 소스이며 이전 정상 운영 코드 복구본이 아님 |

재시작하면 이번 두 파일만이 아니라 src/scripts의 누적 변경이 로드될 수 있다.
이미지 ID를 그대로 유지해도 bind mount 소스는 이전 버전으로 되돌아가지 않는다.
지금 디스크 파일이나 Git HEAD를 09-17에 로드한 코드라고 단정할 근거가 없다.
따라서 이 묶음은 검토/격리 실행 준비용이며 운영 배포 승인 대기 상태로 과장하지 않는다.
실제 반영 전에 정상 동작이 확인된 소스+설정+모델 조합을 복구 기준으로 확보하거나,
격리 실행에서 해당 조합을 검증해 운영 전환 기준으로 삼아야 한다. 소스가 바뀌면 보존본 해시를 다시 대조한다.

### 영상/단절 시험 순서와 판정 근거

| 순서 | 작업 | 기록·통과 조건 |
|---|---|---|
| 1 | 보존 소스로 격리 실행 구성 준비 | 운영 카메라 설정/공용 AP 변경 없이 입력·출력 분리, 실물 명령 전달 없음, 로드 소스와 manifest 대조 |
| 2 | 정상 영상·인물 2명·입력 부족 재생 | 인물별 scope·모델별 완료/실패/캐시 대조, 잘못된 Pose가 정상 인물 처리나 유효 입력 수를 오염시키지 않음 |
| 3 | 격리된 2입력 중 1개 중단/재개 | ERROR/EOS 또는 silent stall 구분, 정상 입력 프레임/이벤트 진행과 첫 복구 프레임 시간 기록 |
| 4 | 재시작/미식별 오류·재시도 거절 대조 | I20 카운터·전체 중단 시간·실제 프레임 재개 비교. stop/start 완료를 복구 합격으로 대체하지 않음 |
| 5 | 운영 적용 범위 확정 | 누적 변경 리뷰·검증된 복구 기준·점검 시간/출력 범위 확정 후 AI 서비스 1개 반영 여부 결정 |
| 6 | 허용된 운영 점검 | 활성 코드/새 지표·정상 입력·오류/지연/출력 결과 대조. 실패 시 검증된 조합으로 복구하고 프레임·이벤트 재확인 |

정상 입력도 멈추면 장애 격리 실패로 기록하고 I20을 유지한다. 프레임이 끊겨도 오류가 오지 않는 silent stall과
모든 재연결의 세대 초기화는 별도 확인 대상이다. 허용 지연·누락·품질 수치는 D04/D07과 합의 전 임의 합격하지 않는다.
현재 **P5.1 IN_PROGRESS, I07/I20 OPEN, I25 FIXED_LOCAL**이다. 운영 적용·실제 단절·현장 품질 합격은 수행하지 않았다.


## 9. v0.38 — 실제 격리 실행·복원본 재실행 결과

[실행·해시·운영 대조 근거](evidence/isolated_execution_20260928.json).
8절의 v0.37 보존본을 `data/runtime/p51_isolated_20260928_v038/candidate`와 `restored`에 각각 풀어
321개 Python 파일의 해시를 대조했다. 기존 Primary 실행 도구를 그대로 사용했으며 제품 코드는 수정하지 않았다.

| 검증 | 결과 |
|---|---|
| 격리 | network none(lo만 존재), 포트/운영 환경 비밀값/장치 명령 연결 없음. 소스·모델·영상 읽기 전용. 결과 전용 bind만 쓰기 허용, tmpfs 임시 공간 |
| 자원/종료 | CPU 2개 제한, 메모리 8GiB, pids 256, 개별 명령 180초 제한. 실제 CPU/GPU는 운영과 공유 |
| 실제 입력 | `00003_H_A_FY_C1.mp4` 로컬 파일 1채널, 기존 `yolov8n-pose.engine`, mux 1280×720, batch 1, interval 0 |
| 후보 실행 | warmup 1초 뒤 3.000초 구간, tensor 메타 190프레임, errors=[] |
| 복원본 실행 | 별도 경로·새 컨테이너, 3.006초 구간, tensor 메타 215프레임, errors=[] |
| 복원본 회귀 | 225 PASS / 0 FAIL / 2 SKIP, 7.01초. 입력 검증·인물 분리·낙상 정책·processor·모델 계측 |
| 운영 관측 | 컨테이너 시작 09-17 08:45:46 KST 유지, restart count 0→0, healthy→healthy, 전후 frames_processed +21,661 |
| 보존/정리 | 원본 모델·설정·영상 해시 동일. 시험 컨테이너 모두 --rm 종료, 잔여 0 |

[후보 영상 보고서](evidence/isolated_candidate_replay_20260928.json) ·
[복원본 영상 보고서](evidence/isolated_restored_replay_20260928.json) ·
[복원본 회귀 JUnit](evidence/isolated_restored_tests_20260928.xml).

처음에는 읽기 전용 /app 아래 중첩 마운트 대상 폴더 부재와 결과 폴더 UID 차이로 준비가 실패했다.
격리 복원 디렉터리에 빈 마운트 대상 생성·호스트 UID 사용으로 해결했다. 이미지에 pytest가 없어
현재 호스트에 설치된 순수 Python 시험 의존성을 별도 `/test-deps` 읽기 전용 마운트로 연결했다.
프로젝트 conftest가 상대 경로 `tmp_test_dirs`를 만들어 35개 준비 오류가 발생한 건은
pytest 작업 경로를 컨테이너 `/tmp`로 바꾸어 해결했다. 실패 로그/XML은 로컬 실행 폴더에 보존했다.
DeepStream의 미사용 UDP/Triton/Argus 플러그인 경고가 있었지만 필요한 nvurisrcbin/nvstreammux/nvinfer/fakesink는
존재했고 두 실제 영상 실행에서 오류 없이 tensor 메타가 생성됐다. 새 라이브러리 설치·운영 환경 변경은 없었다.

### 재실행

실행 정의는 `data/runtime/p51_isolated_20260928_v038/compose.json`이다. 이미지 ID는 8절과 같고
모델/설정/영상 해시는 실행 근거에 기록했다. 아래 fresh output 경로는 기존 디렉터리가 없어야 한다.

```bash
rtk proxy docker compose -p p51-isolated-v038 -f data/runtime/p51_isolated_20260928_v038/compose.json run --rm -T restored /app/scripts/ops/benchmark_deepstream_primary.py --video /input/01.원천데이터/영상/Y/FY/00003_H_A_FY_C1/00003_H_A_FY_C1.mp4 --config /app/config/deepstream/config_infer_primary.txt --engine /app/models/fall/yolov8n-pose.engine --output /output/restored-replay-next --channels 1 --warmup 1 --seconds 3 --width 1280 --height 720
rtk proxy docker compose -p p51-isolated-v038 -f data/runtime/p51_isolated_20260928_v038/compose.json run --rm -T --workdir /tmp restored -m pytest /tests/test_falldata_aux.py /tests/test_falldata_pose_tracks.py /tests/test_fall_aux_policy.py /tests/test_deepstream_processor.py /tests/test_deepstream_model_metrics.py -q -rs -p no:cacheprovider -o addopts=--tb=short --junitxml=/output/restored_tests_next.xml
```

### 판정의 한계와 다음 작업

이번에 검증한 복원은 **후보 Python 보존본을 다시 풀어 Primary 도구와 회귀를 실행하는 범위**다.
09-17에 운영 프로세스가 읽은 이전 소스 복구, 전체 앱·모든 모델·외부 전송·물리 출력 복구는 검증하지 않았다.
src/scripts 보존본 외의 main.py·app/runners·실행 설정 등은 전체 앱 격리 단계에서 함께 고정해야 한다.
3초 표본은 초기 버퍼 따라잡기의 영향을 받으므로 표시 FPS를 지속 처리율/4채널/전체 모델 합격 근거로 사용하지 않는다.
전후 운영 프레임 증가와 healthy는 관측했지만 같은 GPU를 공유하므로 순간 성능 영향이 없었다고 단정하지 않는다.
실제 2입력 단절·재연결 및 I20 전체 파이프라인 중단 범위는 아직 시험하지 않았다.
**P5.1 IN_PROGRESS, I07/I20 OPEN 유지.** 다음은 전체 앱 격리 실행 구성과 2입력 단절/복구 시험이다.


## 10. v0.39 — 실제 CLI 2입력 단절·재연결 시험

이번에는 `main.py`→`app.main`→`start_processor_runtime`→`DeepStreamProcessor`를 실제로 실행했다.
컨테이너 network none 안의 MediaMTX와 두 영상 발행 프로세스가 같은 샘플 영상을 서로 다른 RTSP 경로로 공급한다.
Pose Primary·tracker·후처리·메트릭 서버를 실행하되 helmet/appearance/face/RF/TCN은 비활성화했다.
실물 CCTV 두 대나 전체 모델 조합 시험과 구분한다. MQTT는 컨테이너 loopback의 닫힌 포트로 지정해
외부 전송을 차단했다. 연결 거부·sent=0/failed 이벤트는 예상된 시험 조건이며 관제 전달 합격이 아니다.

### 준비 중 확인한 제약

- GstRtspServer Python typelib과 ffmpeg가 이미지에 없었다. 운영 미디어 서버의 실행 파일과 호스트에 이미 있는
  ffmpeg/실행 의존성을 시험 폴더에 복사했다. 새 패키지 설치·운영 미디어 서버 수정은 없다.
- 호스트 UID의 inotify 감시 등록이 errno 28로 실패했다. 디스크는 여유가 있었으며 호스트 감시 한도를 바꾸지 않고
  시험 MediaMTX를 설정 파일 감시가 없는 환경변수 방식으로 실행했다.
- 기본 `models/fall/yolov8n-pose.engine`는 maxBatchSize=1이어서 이 격리 기본 Pose 경로의 카메라 2개 요청을 거절했다.
  전체 운영 구성에 대한 단정이 아니라 실제 오류 로그로 확인한 해당 엔진/설정 조합의 제한이다.
- 기존 `data/benchmarks/pose_batch_candidate_20260917.vHdXep/pose_b1_4_fp16_l0.engine`의 이전 2채널 실행 성공 기록을
  확인하고 **시험 복사본에만** 연결했다. 새 엔진 빌드나 운영 모델 교체는 하지 않았다. 후보 모델 정확도 승인도 아니다.

### 실패 재현과 최소 보완

설치된 nvurisrcbin introspection 결과 `rtsp-reconnect-interval` 기본값은 0(무수신 시간 제한 비활성),
`rtsp-reconnect-attempts` 기본값은 -1이었다. 기존 source 생성 경로는 이 시간 제한을 설정하지 않았다.

`src/core/_deepstream_source_manager.py`에 **`DS_RTSP_RECONNECT_INTERVAL_SEC`** 환경 설정을 연결했다.
기본값 0을 유지하고 양수일 때만 RTSP/RTSPS URI의 `rtsp-reconnect-interval`에 적용한다.
잘못된 숫자/음수는 기존 env 파서의 기본값 처리에 따라 비활성으로 남고, 미지원 플러그인은 경고를 남긴다.
파일 입력에는 적용하지 않는다. 재시도 횟수는 플러그인 기존 설정을 유지한다.
운영 .env/Compose는 변경하지 않았으므로 이 선택 기능은 운영에서 활성화하지 않았다.

기존 source attach 시험을 확장해 기본/양수/0/음수/잘못된 값/파일 입력/미지원 옵션을 확인했다.
신규 경우 6개 포함 **140 PASS / 2 SKIP**, Ruff·컴파일·diff 공백 검사 통과.
[JUnit](evidence/rtsp_reconnect_20260928.xml). SKIP은 DeepStream 미설치 환경 전용 2개다.

### 실제 단절 비교 결과

[조건·단계별 카운터·운영 대조](evidence/rtsp_reconnect_comparison_20260928.json) ·
[15초 설정 원시 표본](evidence/rtsp_reconnect_samples_20260928.jsonl).

| 설정 | 단절/재발행 시험 | 관측 결과 |
|---|---|---|
| 기본 0초 | 발행 12초 중단 후 30초 관찰 | 입력 1 프레임 195→195로 미복구. 입력 2는 510→1564로 진행. 앱 ERROR/전체 재시작 카운터 증가 없음 |
| 시험 3초 | 정상 시작 확인 단계 | 초기화 중 반복 재연결, 90초 내 두 입력 준비 실패. 단절 단계 미진입 |
| 시험 15초 | 발행 20초 중단 후 45초 관찰 | 입력 1 재발행 프로세스 시작 약 14.859초 후 프레임 증가 최초 관측. 재개 구간 198→762 |
| 정상 입력 2, 15초 시험 | 중단 구간 / 재개 관찰 구간 | 각각 +496 / +974프레임. 표본상의 최대 마지막 프레임 나이 약 0.102 / 0.116초. 카운터 감소 없음 |
| 전체 파이프라인 | 15초 시험 | 앱 restart requested/executed/failed 모두 0. source 수준 native 재연결로 프레임 재개 |

14.859초는 **발행 프로세스 재시작부터 첫 프레임 증가를 표본으로 관측한 시간**이다.
서버가 입력 준비를 완료한 시각, 정확한 RTSP 복구 지연이나 촬영→관제 종단 지연과 같지 않다.
표본 목표 간격은 0.5초이고 실제 간격은 실행 부하에 따라 다르다. 15초를 운영 권장값/합격 기준으로 확정하지 않았다.
서로 다른 대기/관찰 시간의 기능 시험이며 통계적 성능 A/B 비교가 아니다.

### 보존·재현·남은 일

실행 구성·전체 CLI 소스/설정·실행 도구 해시는 `data/runtime/p51_app_20260928_v039/manifest.json`,
실행 정의는 같은 폴더 `compose.json`, 드라이버는 `run_experiment.py`다.
`attempt_batch1`, `attempt_no_reconnect`, `attempt_reconnect3`와 최종 `output`에 각 로그/결과를 나눠 보존했다.
재실행 시 이전 output을 먼저 별도 보존하고 빈 output을 준비한다. 동일 파일에 표본을 덧붙여 해석하면 안 된다.

```bash
rtk proxy docker compose -p p51-app-v039 -f data/runtime/p51_app_20260928_v039/compose.json run --rm -T app-test /experiment/run_experiment.py
```

네트워크/외부 포트/운영 비밀값 연결 없음, 소스·모델·영상 읽기 전용, 별도 결과 폴더만 writable bind다.
시험 종료 후 컨테이너 잔여 0. 운영 AI는 09-17 시작 시각 유지·restart 0·healthy였고 전후 frames_processed가
40,337 증가했다. 운영 모델 해시 동일. CPU/GPU 공유에 따른 순간 성능 영향이 없었다고 단정하지 않는다.

**I20 OPEN, P5.1 IN_PROGRESS** 유지. native 재연결은 기존 전체 파이프라인 재시작 카운터로 세지지 않으므로
별도 카메라 상태 관측이 필요하다. 다음은 재연결 시 추적 ID/입력 세대·RF/TCN/속성 캐시가 이전 입력과 섞이지 않는지
검증하는 작업이다. 실제 CCTV, 전체 모델, 다른 ERROR/EOS·장시간 단절, 운영 활성화와 현장 지연 기준은 남아 있다.


## 11. v0.40 — native 재연결 세대·캐시 경계 (2026-09-28)

기존 v0.39의 실제 CLI/2입력 격리 구성을 재사용해 단절 20초·재발행 후 45초를 관찰했다.
이번에는 inline Pose 버퍼를 활성화하되 RF 모델 경로를 의도적으로 없는 파일로 지정했다.
실제 Pose 후처리·버퍼 세대 검증이며 RF/TCN 모델 추론·정확도 합격이 아니다.

### 재현과 최소 보완

설치된 DeepStream 7.1 `gst-nvevent.h`의 STREAM_RESET(103454),
`gstnvtracker.cpp`의 직렬화 이벤트 처리와 실행 중 `tracking-id-reset-mode=1`을 확인했다.
한 복구 과정에서 서로 다른 seqnum의 reset 신호 2개가 출력 probe에 전달됐다.
프레임 번호가 증가하므로 기존 `frame_index` 역행 조건은 세대를 바꾸지 못했다.
수정 전에는 reset 시점의 Pose 101개가 복구 후에도 남았고 세대가 1로 유지됐다.

- `_deepstream_pipeline_builder.py`: 프레임 probe와 같은 출력 pad에 EVENT_DOWNSTREAM callback 등록. 다음 프레임 처리 전 직렬화된 reset을 받는다.
- `deepstream_processor.py`: 신호의 uint source-id를 카메라에 매핑하고 기존 카메라 제거 시 정리를 공통 함수로 재사용한다. 등록/AI 설정/정상 채널은 유지하고 Pose 기록·세대/inline 결과, synthetic ID, TrackManager, 위반/디바운싱, 얼굴·컨텍스트 캐시를 초기화한다.
- `event_debouncer.py`: 해당 카메라의 누적 판정·중복 억제 상태만 잠금 내에서 제거한다. DB나 발행 큐에 저장된 이벤트를 삭제하지 않는다.
- `processor_metrics.py`: `cctv_processor_source_stream_reset_events_total`, `cctv_processor_source_stream_reset_unidentified_total` 노출. 전자는 **수신한 신호 수**다. 복구 시도/성공 수와 다르며, 식별 실패 때 임의로 모든 채널을 초기화하지 않는다.

### 전후 실제 앱 관찰

| 확인 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 실제 reset 신호 | 2개 | 2개, 식별 실패 0 |
| reset 직전 해당 입력 Pose 기록 | 101개 잔류 | 90개 제거 |
| 해당 입력 세대 | 1 유지 | 1→3→4 |
| 첫 복구 프레임 관찰의 Pose 기록 | frame 0부터 남음 | frame 182의 기록 1개만 존재 |
| 정상 입력 세대 | 2 유지 | 2 유지 |
| 정상 입력 중단/복구 구간 프레임 증가 | +536 / +1162 | +556 / +1006 |
| 전체 파이프라인 재시작 요청/실행/실패 | 모두 0 | 모두 0 |

신호는 매번 초기화해 연속 reset도 안전하게 처리한다. 첫 복구 표본의 synthetic ID와
단절 전 표본의 ID 집합은 겹치지 않았다. 이번 Pose-primary 구성의 native object meta는
비어 있으므로 **nvtracker 실제 인물 ID 단절/연속성 합격으로 확대하지 않는다**.

검증: 신규 10개 포함 **258 PASS / 2 SKIP**(DeepStream 미설치 환경 전용), Ruff·컴파일 통과.
증가하는 프레임 번호·연속 reset·이전 Pose 결과 무효화·다른 채널 캐시 유지·
미식별 이벤트·hook 연결과 지표 노출을 확인했다. 기존 fixture 두 곳에 새 callback/의존성을 추가했다.

- [JUnit](evidence/native_reset_20260928.xml) · [전후 실측 및 범위](evidence/native_reset_20260928.json)
- 실행 보존: `data/runtime/p51_reset_20260928_v040/`의 `before/`, `baseline/`, `output/`, `manifest.json`.
- 재현: 별도 빈 output 경로를 준비한 뒤 해당 `compose.json`으로 `/audit/run_experiment.py` 실행. 기존 output은 증거이므로 덮어쓰지 않는다.
- 네트워크 none·외부 출력 미연결·읽기 전용 모델/소스 유지. 물리 GPU/CPU는 운영과 공유하며 운영 연속 지연을 계측한 시험은 아니다.
- 운영 시작 시각·restart 0·healthy 유지, 시험 컨테이너 잔여 0. 운영 재시작/설정 변경 없음.

### 남은 범위

I20 OPEN, P5.1 IN_PROGRESS를 유지한다. 캐시를 비우는 것만으로 이미 실행 중인
얼굴/의상 비동기 작업이 이전 결과를 다시 넣는 것을 막았다고 볼 수 없다.
다음은 해당 작업의 입력 세대 전달·지연 결과 차단 검증이다. native 인물 추적,
실CCTV·전체 모델·저신뢰/시간 간격·다른 ERROR/EOS·운영 활성화 확인도 남아 있다.


## 12. v0.41 — 얼굴·의상 비동기 입력 세대 (2026-09-28)

### 변경과 영향

기존 worker의 작업에는 입력 세대가 없고, 얼굴/의상 함수는 결과를 반환하기 전에
캐시·스냅샷·속성 이력·DB를 갱신했다. 마지막 발행만 검사하면 이 내부 경로가 남는다.

- `_deepstream_context_worker.py`: 작업에 camera별 generation을 저장한다. 대기 중 구작업은 추론 전 제외하며, 추론 중 reset은 결과 반영 시 다시 검사한다. 초기화·결과 반영은 같은 잠금으로 직렬화한다.
- `_deepstream_face_context.py`: 얼굴 추론은 잠금 밖에서 수행하고, cache/snapshot 반영 전에 세대를 확인한다.
- `ai/_appearance_pipeline.py`: 속성 추출은 잠금 밖에서 수행하고, smoothing·crop/DB·조건 이벤트 반영을 보호한다. 카메라별 history/대표 crop 참조/cooldown 제거와 I26 카메라 간 cooldown 분리를 추가한다.
- `deepstream_processor.py`: guard를 얼굴 함수로 전달하고 native reset·카메라 제거·파이프라인 생성/재생성에 공통 초기화를 연결한다.

외부 API/DB 스키마/모델 설정은 변경하지 않았다. guard가 없는 기존 동기 호출은
기존 경로를 사용하며, 의상 cooldown의 카메라 분리는 공통 AppearancePipeline 호출에 적용된다.
이전 세대에서 이미 반영한 과거 DB·이미지는 삭제하지 않는다. 이번 차단 대상은
reset 이후 반영하려는 미완료 구작업이며, 저장된 정상 이벤트/발행 큐를 비우지 않는다.

### 시험과 한계

신규 9개 포함 **211 PASS / 2 SKIP**. 선택한 묶음은 context worker/face context/
appearance pipeline/processor/pipeline builder/model metrics/Pose tracks다.
대기·얼굴 추론·의상 추론 중 reset과 다른 카메라 reset을 6가지로 시험했다.
실제 Thread/Event로 추론을 대기시키고 reset이 먼저 완료되는지 확인했다.
같은 카메라의 이전 결과는 cache·DB insert·enqueue에 반영되지 않고 새 작업은 정상 실행된다.
다른 카메라의 작업도 유지된다. 나머지 3개는 카메라별 초기화 및 로그·조건 쿨다운 분리 시험이다.

추론 출력과 DB insert는 시험 대역이며 실제 모델 정확도/실영상/SQLite 부하 검증이 아니다.
저장 I/O와 enqueue도 commit 잠금 범위에 포함되므로 **느린 저장 중 reset 대기 시간은
미측정**이다. 추론 대기와 저장 대기를 구분하며 무병목/30FPS 합격으로 확대하지 않는다.
입력 세대는 작업 접수 시점 기준이다. 미리보기 프레임과 탐지 메타의 시각 대응은 별도 종료 점검 대상이다.

- [JUnit](evidence/context_generation_20260928.xml) · [범위·해시·운영 확인](evidence/context_generation_20260928.json)
- 수정 전 보존: `data/runtime/p51_context_20260928_v041/before/`, 변경 파일 해시 `manifest.json`.
- Ruff·컴파일 통과. 운영 started_at·restart 0·healthy 확인, 운영 재시작/설정 변경 없음.
- I26 FIXED_LOCAL, I20/I07 OPEN. P5.1 종료 점검: 프레임·시각 대응 → 실제 모델 완료 지표 → 남은 단일 입력 계약 확인.
- 정확도/임계값은 P5.2~P5.5, 전체 모델 1→2→4채널 부하는 P6.2, 지속 운전/느린 저장 중 복구 지연은 P6.3 범위로 유지한다. 현재 P5.1은 IN_PROGRESS다.


## 13. v0.42 — 프레임 출처·PTS와 종료 판정 (2026-09-28)

### 점검 결과와 보완

`get_camera_frame(cam2)`가 없던 cam2 대신 cam1 이미지를 반환하는 동작과,
reset 시 기존 이미지가 남는 동작을 2개 실패 테스트로 재현했다.
`deepstream_processor.py`의 다른 카메라 fallback을 제거하고 reset에서
`PreviewFrameStore.clear_camera`를 호출한다. 삭제하는 것은 해당 시점의 캐시이며
이미 큐에 들어온 이전 샘플까지 폐기했다고 해석하지 않는다(I27).

`_preview_frame_store.py`는 실제 native batch 메타에서 단일 프레임·batch_id 0·
source ID 매핑을 확인한 뒤 이미지를 저장한다. 식별 불가/다중 프레임은 거절한다.
카메라·frame_num·source_pts_ns·buffer_pts_ns·수신 monotonic/wall time을 별도 보존한다.
PTS 결측은 None으로 남기며 wall time으로 대신하지 않는다. PTS는 검증된 촬영 UTC가 아니다.

`processor_metrics.py`의 `cctv_processor_preview_frames_stored_total`과
`cctv_processor_preview_identity_rejected_total`로 저장/식별 거절을 구분한다.
새 샘플을 실제 저장한 callback에서만 보조 clip 버퍼로 전달해 throttled 샘플의
이전 이미지 재추가를 막았다. 아직 카메라별로 분리되지 않은 clip 버퍼는 단일 입력에서만 사용한다.
이 변경은 다중 입력 clip 기능을 완성한 것이 아니라 혼합 방지 경계다.

### 실제 후보 실행 및 테스트

| 범위 | 결과 | 판정 |
|---|---|---|
| batch2 + preview, 실제 CLI | nvvideoconvert transform 오류 1회, preview 0 | 실패·I28 OPEN |
| 단일 입력, 수정 전 | preview 메타 표본 14개; source PTS와 buffer PTS 모두 다름 | 시간축 구분 근거 |
| 단일 입력, 수정 후 최종 | 새 frame callback 65건; 마지막 scrape 저장 62·식별 거절 0; 변환 오류 0 | 단일 입력 출처 저장 확인 |
| 최종 preview 메타 표본 | 14개, source PTS와 buffer PTS가 모두 다름 | 변환 buffer PTS를 원본 frame PTS로 대체하지 않음 |
| 이번 회귀 | 신규 17개 포함 172 PASS / 2 SKIP | 미설치 환경 전용 skip 2, Ruff·컴파일 통과 |
| 운영 모델 지표 | Primary/model-label 0, 처리 누적 28,730,127 | I07 OPEN, 전체 모델 완료 계측 합격 아님 |

초기 관찰기는 호출마다 만든 class가 Gst.Sample 참조를 오래 보유할 가능성이 있었다.
고정 class로 보완한 최종 실행에서는 meta pool 경고가 0이었다(이전 125/137회).
이는 관찰 도구 보완이며 제품 tracker pool 설정 변경이나 장시간 메모리 안정성 합격이 아니다.
최종 저장 수 62와 callback 수 65는 마지막 scrape와 종료까지 관찰 시점 차이다.
단일 입력 fixture는 입력 1개를 설정했지만 두 번째 publisher도 실행한다. 이를 2입력 처리로 세지 않는다.

- [JUnit](evidence/frame_identity_20260928.xml) · [실제 실행·범위](evidence/frame_identity_20260928.json) · [운영 metrics](evidence/model_exit_metrics_20260928.prom)
- 격리 자료 `data/runtime/p51_frames_20260928_v042/`: 2입력 실패, 단일 입력 전후, 최종 output, 소스/관찰기 해시 보존.
- 네트워크 none·실물 출력 미연결·운영 재시작/설정 변경 없음. 운영 started_at 유지·restart 0·healthy, 시험 컨테이너 잔여 0.

### P5.1 종료 판정: IN_PROGRESS

출처 저장과 정확한 후처리 결합은 서로 다른 조건이다. worker가 최신 preview를 가져오는
경로는 남아 있어 탐지와 같은 frame_num/source PTS인지 보장하지 못한다. 다음 작업은
I27 동일 프레임 결합과 reset 이전 대기 샘플의 세대 구분이다. I28 다중 preview 분리,
I07 실제 활성 모델별 완료 계측도 열린 항목이며 이번 결과로 DONE 판정하지 않는다.

## 14. v0.43 — I27 동일 프레임 결합 (2026-09-28)

### 변경과 책임 범위

얼굴·의상 context worker에서 최신 이미지 조회를 제거했다. 메타 probe가 각 프레임에
프로세스 내에서 재사용하지 않는 순번과 카메라 세대를 붙인다. 원본 camera/source ID,
frame_num, source_pts_ns, 세대, 순번이 모두 같은 CPU preview만 후처리 큐에 전달한다.
변환 후 buffer_pts_ns는 원본 PTS와 다를 수 있으므로 결합 키에 사용하지 않는다.

태그는 `NvDsFrameMeta.misc_frame_info[0:3]`의 marker `0x43435456`, generation,
frame_token에 저장한다. [NVIDIA 문서의 사용자 메타 필드](https://docs.nvidia.com/metropolis/deepstream/7.1/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html)를 사용하며
reserved 및 네 번째 슬롯은 변경하지 않는다. 다른 사용자의 값이 있으면 덮어쓰지 않고
해당 후처리를 제외한다. 설치된 pyds 배열의 쓰기와 실제 CPU 변환 후 보존을 확인했다.
frame_token은 해당 processor 실행 안에서만 유효하며 공통 event_id를 대체하지 않는다.

| 파일 | 변경 |
|---|---|
| `src/core/_deepstream_context_worker.py` | 토큰 부여·동일 프레임 대기 메타·영상 결합·이전 세대 메타 제외 |
| `src/core/_deepstream_event_factory.py` | tensor/object 이벤트에 native 식별 전달, 변환 전 태그 callback |
| `src/core/_preview_frame_store.py` | 변환 후 태그 읽기·영상과 식별자를 같은 callback으로 전달 |
| `src/core/deepstream_processor.py` | crop Pose 경로 포함 연결·진단 계수 노출 |

외부 필수 payload·DB 스키마·모델/운영 설정은 유지한다. 이벤트의 확장 metadata에
`frame_identity`를 추가한다. 태그 또는 PTS가 없으면 동일 프레임 후처리를 수행하지 않는다.
얼굴/의상 보조 작업을 제외해도 기존 감지 이벤트의 저장·전달을 중단하지 않는다.

카메라당 대기 **메타**는 32프레임으로 제한한다. 오래된 항목부터 제외하고, 다음 submit/accept
접근에서 2초 초과 항목을 만료시킨다. 타이머가 주기적으로 삭제하는 방식은 아니다.
대기 map에 영상을 보관하지 않으며 완성된 작업은 기존 크기 제한 큐를 사용한다.
영상이 도착할 때까지 스레드를 대기시키지 않는다. 저장 I/O의 잠금 지연까지 없는 것은 아니다.
`get_stats()`의 `context_*` 계수로 결합·불일치·만료·용량 초과·큐 드롭을 구분한다.

reset은 해당 카메라 대기 메타를 지우고 세대를 증가시킨다. 이전 preview가 늦게 도착하거나
frame_num/PTS가 다시 사용돼도 예전 토큰은 새 메타와 결합되지 않는다. 이미 큐에 있던 작업은
v0.41의 commit guard로 차단한다. 주변 객체 캐시에도 오래된 세대의 메타를 반영하지 않는다.
같은 native 프레임의 tensor/object 중복 전달은 첫 person 작업 하나로 처리한다.

### 검증과 판정

- **신규 13개 포함 235 PASS / 2 SKIP**. 제외 2개는 DeepStream 미설치 전용이며 이 장비에는 설치돼 있다.
- 카메라/source/frame/PTS/세대/토큰 불일치, 동일 번호·PTS 재사용, PTS 결측, 혼합 메타,
  대기 제한/만료, 큐 포화, 중복 제출, 다른 카메라 유지 및 기존 비동기 reset 회귀를 확인했다.
- 최종 실제 CLI: RTSP 입력 1개·Primary Pose·CPU preview, 발행 중지 20초 후 재개.
  약 99.2초 관찰에서 preview 204건, 후처리 큐 소비 128건의 식별자와 영상 SHA-256이 모두 일치했다.
  native reset 2회, 후처리 세대 1의 20건 → 세대 3의 108건. 마지막 reset 이후 구세대 작업 관측 0건.
- 변환 실패·관측 대상 meta pool 경고·Python traceback 0건. 첫 실행 120건 일치도 별도 보존했다.
- 얼굴 backend는 disabled다. 실제 Pose 탐지 → 메타 → CPU 이미지 → context worker 결합을
  검증했으며 얼굴/의상 모델 정확도·crop 좌표 보정·DB/실물 출력 합격으로 확대하지 않는다.
- 두 번째 publisher는 실행되지만 앱 입력은 1개다. 다중 입력 합격으로 세지 않는다.
  외부 네트워크 none, MQTT 목적지는 격리 loopback의 닫힌 포트다. 운영 AI는 기존 시작 시각,
  restart 0·healthy를 유지했고 시험 컨테이너는 종료됐다. 운영 적용을 수행하지 않았다.
- Ruff·컴파일 통과. 전체 트리 diff 검사에서 기존 Compose 공백 오류가 있어 보존했으며
  이번 변경 경로는 별도 검사한다.

[JUnit](evidence/frame_matching_20260928.xml) · [실행·해시·범위](evidence/frame_matching_20260928.json).
격리 소스/관찰기 347개 해시와 전후 diff는 `data/runtime/p51_matching_20260928_v043/`에 보존했다.

**I27 FIXED_LOCAL, P5.1 IN_PROGRESS.** 다음은 단일 입력의 모델별 실제 실행·완료 지표 최종 대조다.
I07 운영 활성화 확인, I28 다중 preview/P6.2, 모델 품질 P5.2~P5.5,
느린 저장·장시간 복구 지연 P6.3의 잔여 범위는 유지한다.


## 15. v0.44 — 모델별 실제 출력·ROI 경계 대조 (2026-09-28)

[모델 계측 문서](MODEL_INFERENCE_METRICS.md)의 v0.44에 실제 CLI 5회 대조를 기록했다.
SGIE 입력 등록과 tracker 탐지 완료 표시를 보완해 I29를 FIXED_LOCAL로 옮겼다.
동일 원본/처리 해상도에서 Primary tensor 325·의상/헬멧 각각 266프레임 관측,
201 PASS/2 SKIP이다. 색상은 229호출/228완료 표본이며 RF는 요청 미관측, TCN/얼굴은 비활성이다.

기본 4K 원본→1080p mux 구성에서는 원본 ROI 좌표가 처리 표면과 맞지 않아 변환 오류가
발생했다(I30 OPEN). 다음은 resize/padding·ROI 경계·preview/event 역대응 보정이다.
운영 설정은 그대로이며 P5.1은 IN_PROGRESS다. 동일 해상도에서의 tensor 관측을
기본 구성 합격·실제 모델 품질·전체 AI 완료율로 해석하지 않는다.

## 16. v0.45 — mux 좌표·ROI 경계 보정

**I30 FIXED_LOCAL, P5.1 IN_PROGRESS, 운영 미반영.**

- 처리 기준: nvinfer가 입력받는 **패딩 포함 mux surface**의 크기(`DS_STREAM_WIDTH/HEIGHT`)로 YOLO bbox/keypoint를 복원한다. 원본 `source_frame_width/height`는 이 좌표의 기준으로 사용하지 않는다.
- native 객체 ROI·tracker 매칭·OSD·Crop-Pose 전체 좌표와 이벤트 `metadata.frame_width/height`도 같은 기준이다. 기존 preview resize 시 의상 후처리의 비율 변환을 재사용한다. 기존 `frame_identity`의 원본 PTS/카메라/프레임/세대는 유지한다.
- mux 설정과 좌표 복원이 `streammux_dimensions()`를 공유한다. 경계 밖 ROI는 surface 교차 영역으로 자르고, 완전히 벗어나거나 폭/높이가 0 이하이면 등록하지 않는다.
- 원본 영상 위의 좌표를 제공하는 새 API는 추가하지 않았다. 원본 위 overlay가 필요한 소비자는 mux padding/resize 역변환이 별도로 필요하며 현재 이벤트 좌표를 원본 좌표로 해석하면 안 된다. DB schema·운영 env·모델 파일 변경 없음.
- [NVIDIA mux 문서](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html)는 mux가 지정 크기로 축소/패딩하며 메타의 입력 해상도는 원본임을 설명한다. 설치 SDK `gstnvinfer.cpp:1568`에서도 Primary crop이 실제 surface 크기를 사용한다.

| 검증 | 결과 | 한계 |
|---|---|---|
| 좌표 회귀 | 신규 17개 포함 230 PASS/2 SKIP; 수정 전 13 FAIL/1 PASS | 모델 정확도 시험 아님 |
| 4K→1080p 동영상 재시험 | 833프레임, Primary 417·의상/헬멧 각 326프레임 tensor, 변환 오류 0 | 최초 입력 미수신 150초 실패 별도 보존 |
| 4K→1280×960 padding 동영상 | 877프레임, Primary 439·의상/헬멧 각 275프레임 tensor, 변환 오류 0 | 첫 ROI/preview 시각적 위치 불일치 I32 |
| 고정 첫 프레임→padding | 163프레임, 변환 오류 0, 인물 부근 ROI 복원 | 일부 화면 손상 I32, 정지 영상으로 동영상 시각 대응 증명 불가 |
| 후처리 토큰/크기 | 126/112/84샘플의 식별자·크기 불일치 0 | 영상 내용 대응과 구분 |

원본 증거는 [JSON](evidence/coordinates_20260928.json)·[JUnit](evidence/coordinates_20260928.xml),
fixture `data/runtime/p51_coordinates_20260928_v045/`에 있다. 특히 padding의
`preview.jpg`/`preview_geometry.json`은 화면 인물과 ROI가 어긋난 증거이므로 합격 이미지로 사용하지 않는다.
`padded_static/preview_0.jpg`의 하단 손상도 함께 보존한다.

I31(의상 DB 추적 ID 범위 초과)은 P5.4/저장 회귀, I32(영상 내용 대응)는 P5.1 종료 점검에 배정한다.
형식상 토큰·해상도 일치와 모델 tensor 관측을 전체 영상/AI/관제/장치 합격으로 확대하지 않는다.
운영 시작 시각·재시작 0·healthy는 전후 동일하며 시험 컨테이너는 제거했다.

## 17. v0.46 — RTSP 영상 손상·추적 실행 대조

v0.46: [추적·입력 대조](TRACKER_VALIDATION.md). RTSP 지연 패킷 폐기 선택 설정 추가(기본 유지), **172 PASS / 2 SKIP**. 해제 시 영상 표본 개선이나 4K/59.94fps 처리 지연은 잔여(I32 OPEN). NvDCF 첫 시험 10프레임 정지·재시험 Primary 644프레임/35초 완료, 영상 추출 유지. I20/I31 OPEN·IOU 유지·운영 미반영·P5.1 IN_PROGRESS.

재시험의 공개 미리보기 flag는 꺼졌지만 의상 분석이 CPU 프레임을 요구해 영상 추출은 유지됐다. 미리보기 제거가 정지 원인 해결이라고 해석하지 않는다.

<a id="p51-closeout-v075"></a>

## 18. v0.75 — GPU/VIC 대조와 P5.1 종료 점검 (2026-09-30)

**판정:** 기존 GPU 전처리 유지. VIC 후보는 실행되지만 같은 모델 결과를 보장하지 않아 승격하지 않는다.
CUDA700 원인 미확정(I34)을 유지하며 10-01 종료 검토를 목표로 한다. 오늘 DONE 근거는 부족하다.
앞의1~17절과 과거 인수 표는 당시 기록이며 현재 상태는 이 절과 WBS를 따른다.

### 동일 입력 대조 결과

실CCTV 녹화에서 재인코딩 없이120프레임을 고정했다. 입력 SHA·frame_num·원본 PTS·출력 PTS를
대조하고 GPU→VIC→VIC→GPU 순으로 각120프레임 실행했다. 네 실행 모두120개 tensor·EOS·종료0.
동일 장치의 두 실행은56×8400 raw tensor가 모두 정확히 같았고, GPU/VIC 간 결과는 달랐다.

| 비교 항목 | GPU↔VIC 결과 |
|---|---|
| 인물 검출 수 | 두 경로 모두 매 프레임1명,120쌍 |
| bbox IoU | 최소0.961·중앙값0.981 |
| 인물 신뢰도 차이 | 최대0.0461 |
| 관절 좌표 차이 | 양쪽 신뢰도≥0.25인899관측: 중앙값3.78px·P95 18.075px·최대81px |
| 관절 신뢰도0.25 경계 변경 | 전체2040관절 관측 중119개 |
| 낙상 bool·사유 |120쌍 모두 동일, 모두 비낙상·다리 부족 |

화면 가장자리에 부분적으로 보이는1명뿐이다. 낙상 정답 라벨이 없고 유효 관절도 부족해
정확도 악화/유지·현장 합격을 판정하지 않는다. 제외된1141관절 관측도 분모에 보존한다.
0.25는 차이를 나누는 기존 점수 기준이며 새 운영 인수 임계값으로 합의한 값이 아니다.

입력→출력 probe 경계 중앙값(처음10프레임 제외)은 GPU53.28/52.96ms,
VIC45.85/42.82ms다. 전처리·추론·큐/스케줄링 포함으로 순수 CUDA 커널 시간이 아니며
동시에 운영 AI가 GPU를 사용했다. 성능 개선율/실시간 지연 상한으로 사용하지 않는다.
원자료 pipeline_seconds에는 최종 tensor 파일 저장도 포함되어 성능 비교에서 제외한다.
Primary-only·interval0·batch1 조건으로 전체 AI 통합/운영 프로파일과 구분한다.

### 현재 GPU 후보 고정과 복구

`data/runtime/p51_tcn_loaded_20260930_v074/candidate/`359파일이 저장 manifest와 모두 일치한다.
TCN 재사용·640 입력·인물별 후처리0.5초 후보를 그대로 사용하여 격리 입력25초 단절을 시험했다.
송신 재개 후8.065초에 프레임 재개, 세부 후처리/TCN·세대·출력 검증은 아래 실행 근거에 기록한다.
단절 전후2영상 디코드 종료0·stderr0, 파이프라인 재시작/큐 폐기/후처리 실패/CUDA 서명0.
다만 복구 대기66.439초 시점의 미리보기 결합 누락1건은 I32 잔여로 유지한다.
재개 판정 이후 미리보기208건은 모두세대3으로 이전 세대 재사용은 관측하지 않았다.
TCN은 주기 계측5회/종료까지 별도 관찰6회로 표본 시점 차이가 있으며 timeout0,
Pose 부족1건을 성공적인 낙상 판정으로 세지 않는다.
이 시험의 외부 MQTT는 loopback port9로 차단했다. 공통 이벤트 전달·관제 ACK 인수 시험은 아니다.
운영 컨테이너와 물리 출력 장비에는 변경이 없다.

### 종료 판정표 — 범위를 줄이지 않고 남은 검증을 한정

10-02 v0.97 현재 판정은 아래 표를 갱신한 내용이다. 같은 고정 이미지의 통합 성공과
추가 종료 실패를 함께 반영하며, [상세 근거](MODEL_INFERENCE_METRICS.md#startup-delay-v097)를 따른다.

| 조건 | 현재 근거/상태 | 남은 실행 |
|---|---|---|
| 전처리·좌표·인물/세대·시각 계약 | I24/I25/I27/I30/I31 근거 유지. v0.97 이미지0c11b835의 src206파일이 현재 코드와 일치 | 기존 로컬 근거와 운영 반영 구분 유지 |
| 실제 모델별 완료/실패 구분 | 같은 후보 Primary/의상/헬멧 tensor 프레임2750개씩. TCN18반환=유효17/Pose 부족1; 별도 색상252호출/252반환·얼굴161호출/무결과161 | RF verify0/출처 잔여·Python 의상 Null161회를 성공 모델로 세지 않음. 실제 낙상 정확도는 후속 범위 |
| 단일 입력 단절·세대 격리 | 같은 후보 입력25초 단절→재개8.156초, CPU1450개 식별자 전부 등록 메타 대응·세대1/3 각각735/715 | 장시간·다중 부하/운영 반영은 별도 |
| 출력 정합·I32 | GPU OSD 최초10.866초→CPU0.017642초; 후보 CPU 누락0/후처리324완료·정상 제한1126. 영상3278패킷 디코드/PTS·DTS 정상 | 과거 누락 원인 소급 확정 불가. 최대 영상 간격32.794초는 입력 단절 포함; 연속30FPS/절대 지연 미합격 |
| 종료·I40 | 통합10.345초 정상 종료 뒤 추가 관찰 일반30초/진단60초 초과2회. RTSP sink NULL 후 전체 NULL→NVIDIA 버퍼 풀/드라이버 조건 대기 스택 확보 | OPEN 재개. 이후 정상4회는 해결 근거 아님; 인코더/디코더 위치·유발 조건 확인 필요 |
| CUDA700·I34 |120프레임 memcheck/반복 시작에서 미재현, 근본 원인 미확정. VIC 교체도 동등성 미충족 | 열린 결함 유지·최초 장애 원인/안정화 근거 추가. 미재현을 해결로 바꾸지 않음 |
| 운영 적용·복구·I07 | v0.92 격리 전환/데이터 보존 근거 유지. 새 후보/CPU OSD 검사85 PASS·실제 병합 검사 PASS; 운영 미반영 | I40/I34 등 필수 문제 해결 전 배포 보류. I45 DB/I46 구성 정합 잔여·별도 Jetson은 기능 완성 후 |

새 모델 정확도·다인/가림 품질 평가는 P5.2~P5.5, 다중 부하 P6.2, 장시간 운전 P6.3이다.
원래 배정된 후속 범위이며 I34나 단일 입력 결함을 숨기기 위한 이동은 아니다.

### 오늘·내일 실행 계획

- 09-30 완료: GPU/VIC480프레임 비교·GPU 유지 결정·359파일 고정 확인·현재 후보 단절 복구.
- 09-30~10-01: 최종 모델별 상태/품질 부족 및 출력 정합 대조·필요한 최소 보완/회귀3~5시간 추정.
- 10-01: 증거·적용/복구 자료·종료 판정1~2시간 추정. 신규 결함·CUDA 원인 규명·운영 적용 대기는 포함하지 않는다.

하루6시간 이상이면 위 검토 묶음을 오늘·내일 배치할 수 있지만 **DONE 날짜는 확약하지 않는다**.
I34 또는 필수 조건이 남으면 IN_PROGRESS를 유지하고 완료한 범위와 막힌 조건을 정확히 보고한다.
[실행·입력/소스 해시·상태표 근거](evidence/preprocess_comparison_20260930.json).
