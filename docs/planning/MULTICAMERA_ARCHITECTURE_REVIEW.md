# 공유 GPT 대화의 다중 카메라 설계 적용 검토

2026-09-21 · 계획 v0.12 보충 검토 · **설계 검토 완료 / 구현·성능 개선 판정 아님**

대상: [사용자 공유 대화](https://chatgpt.com/share/6ab07846-ee1c-83ee-8f83-39240909ee3f).
질문 7종과 답변을 읽고 실제 코드·컨테이너·공식 문서를 대조했다. 공유 페이지는 웹 열기에서
실패했으나 공개 HTML의 대화 데이터를 가져와 확인했다. 본문 전체를 저장소에 복제하지 않았다.

**권고: 공유 추론을 유지하고 카메라/출력 장애 복구를 단계적으로 보완한다.**
이미 구현된 pyds·TensorRT·제한된 출력 큐를 다시 도입할 필요는 없다. 모델 성능 개선과
장애 격리는 별도 시험으로 판단한다. 전체를 새 프로세스 구조로 교체하는 결정은 아직 이르다.

## 1. 공유 답변의 전제와 우리 프로젝트의 차이

| 주제 | 실제 확인 | 적용 판단 |
|---|---|---|
| pyds 미구현 여부 | 코드에서 pyds import·tensor/frame/object 메타 접근. 실행 컨테이너에서도 1.2.0 import 성공 | 기존 구현 유지. 앞선 5 SKIP는 로컬 .venv의 설치 조건이며 프로젝트 미구현이 아님 |
| DeepStream + YOLO 효율 | nvinfer TensorRT 엔진 실행 뒤 Python/NumPy로 출력 해석 | 전체 구조는 유효. 모델 실행·tensor 후처리·프레임 복사 비용을 구분해 측정 |
| 모델별 입력 크기 | 로컬 설정: Pose/Person 640×640, Helmet 320×320, PP-Human/PA100K 256×192 | 이미 서로 다름. 하나로 통일하지 않으며 엔진 shape·전처리·좌표 복원·품질을 함께 확인 |
| 카메라별 복구 | source attach/detach·backoff가 있으나 소스 ERROR/EOS는 전체 stop/start 경로로 연결 | 개선 우선 후보 I20. 정상 카메라 지속 처리 여부를 격리 장애 시험으로 검증 |
| 제한된 출력 큐 | preview와 output 큐에 leaky=2, 최대 2버퍼. appsink 최대 1버퍼/drop | 새 도입 과제가 아니라 기존 정책·내부 플러그인 정체·복구 범위 검증 과제 |
| 원문의 출력 방식 | 원문은 DGX Spark·DS 9.1·외부 Triton·카메라별 RTMP. 우리는 DS 7.1·로컬 nvinfer·RTSP publish 설정 | demux→N개 RTMP 설계를 그대로 이식할 근거 없음. 필요한 출력 토폴로지부터 고정 |
| 추론 주기 줄이기 | 컨테이너 환경은 crop-Pose=0, Primary interval=1, Helmet interval=1, PP-Human interval=4/reinfer=15 | 모든 운영 AI의 매 프레임 30FPS 목표와 구분. 프레임 생략을 목표 달성 수단으로 자동 채택하지 않음 |

위 interval은 실행 컨테이너의 환경값과 로컬 설정 적용 코드 대조다. 실행 중 요소의
실제 속성·모델별 완료 FPS를 새로 측정한 결과는 아니다. 플래그별 활성 모델과 엔진도 P5.1에서
확인해야 한다. 현재 환경을 임의로 interval=0으로 바꾸거나 운영 부하를 높이지 않았다.

## 2. 확인 근거

| 근거 | 확인한 내용 |
|---|---|
| [deepstream_processor.py](../../src/core/deepstream_processor.py) | pyds lazy import, tensor→NumPy 후처리, 배치 메타 probe, 파이프라인 stop/start 복구 |
| [pipeline builder](../../src/core/_deepstream_pipeline_builder.py) | 공유 nvinfer/SGIE·OSD·preview 분기·RTSP 출력 조립. 현재 조립 함수에는 카메라별 nvstreamdemux 출력 경로 없음 |
| [source health](../../src/core/_deepstream_source_health.py) | 식별 가능한 소스 ERROR/EOS→전체 재시작 요청; 소스 미상 ERROR→runtime 중단 |
| [source manager](../../src/core/_deepstream_source_manager.py) | 카메라 source 연결/제거를 재사용할 기반 존재 |
| [element config](../../src/core/_deepstream_element_config.py) | 추론 간격·SGIE 배치 설정, output 큐 제한 |
| [Primary 설정](../../config/deepstream/config_infer_primary.txt)·[Helmet 설정](../../config/deepstream/config_infer_helmet.txt)·[PP-Human 설정](../../config/deepstream/config_infer_pphuman.txt) | TensorRT 엔진·입력 shape·output-tensor-meta. 파일 설정과 실제 엔진/활성 구성을 구분 |
| [기존 성능 판단](PERFORMANCE_DECISION.md) | Primary-only 4채널의 최저 카메라 tensor FPS 평균 약 14, 약 30초 구간 적체 증가 약 16초. 전체 AI/현재 성능으로 일반화 금지 |
| [기존 인수 기준](../testing/MULTICAMERA_30FPS_ACCEPTANCE.md) | 실물 4대·전체 운영 AI·카메라별 매 프레임 30FPS. 프레임 생략으로 달성 표시 금지 |

2026-09-21 `cctv-ai-engine` 읽기 전용 확인:

- Python 3.10.12, pyds 1.2.0 import 성공.
- nvinfer/nvurisrcbin 플러그인 7.1.0, SDK 디렉터리 deepstream-7.1.
- USE_DEEPSTREAM=1, DS_OUTPUT_MODE=rtsp-publish, DS_PREVIEW_ENABLED=1.
- nvunixfdsink/nvunixfdsrc 등록됨: plugin nvunixfd 0.0.1, 시스템 GStreamer 경로.
- interpipesink는 해당 registry 조회에서 발견하지 못함.
- deepstream-app 명령은 PATH에 없어 버전 명령이 실패했다. SDK 디렉터리·플러그인 버전으로
  확인을 보완했다. 파이프라인을 생성하거나 GPU 추론을 새로 실행하지 않았다.

[검토 증거·소스 해시](evidence/multicamera_architecture_review_20260921.json)에 기록한다.
그래프 검색으로 진입점을 찾았으나 일부 snippet 줄 범위가 현재 파일과 어긋나 실제 파일로 보완했다.

## 3. 바로 적용할 원칙과 조건부 후보

**카메라별 복구를 우선 검증한다.** 카메라별 마지막 프레임 시각·입력/실제 추론/출력 FPS·
큐 대기·재연결/복구 시간을 함께 관찰한다. 연결 상태만으로 정상 처리라고 판단하지 않는다.
한 카메라의 장애 복구가 정상 카메라의 프레임/이벤트 진행을 멈추는지 별도로 확인한다.

**화면 프레임과 이벤트 보존을 구분한다.** 최신 미리보기의 오래된 프레임 폐기는 허용할 수 있지만,
센서 관측·낙상/헬멧 이벤트·명령 기록에는 같은 폐기 정책을 적용하지 않는다. 또한 현재 preview
프레임 캐시는 일부 CPU 외형/얼굴/낙상 보조 처리에도 쓰인다. 큐를 더 줄이거나 preview를 끄기 전
해당 모델의 프레임 공급과 시간창 영향을 먼저 확인한다.

**원본 접수 성공의 의미를 유지한다.** P2.2 선택 경로의 영상 T1 원본 commit은 생산자 스레드에서
동기 실행한다. 단순히 모든 DB 작업을 메모리 큐 뒤로 옮기면 접수 뒤 종료 시 보존 보장이 약해진다.
전송·외부 저장 대기는 분리하되 T1 지연·잠금·디스크 실패는 계측하고 명시적 실패/역압 정책으로
다룬다. [현재 영상 접수 경계](VIDEO_INGEST_RECOVERY.md)를 함께 적용해야 한다.

**추론 자체의 병목은 따로 줄인다.** 실제 엔진·정밀도·shape/profile을 먼저 확인하고 엔진 단독과
전체 파이프라인을 비교한다. 동일 엔진 조건의 batch 1/2/4, 후처리·프레임 복사·인코딩 비용을
분리한다. 사람 crop 방식은 사람 수에 따라 SGIE 처리량이 늘어날 수 있으므로 한산/혼잡 장면을
모두 본다. FP16/INT8·입력 축소·작은 모델은 같은 평가셋에서 품질과 함께 비교할 후보다.

다음은 조건부로 보류한다.

- GstInterpipe/입력·출력 전면 프로세스 분리: 외부 플러그인·IPC·감시·종료/복구 부담이 증가한다.
  먼저 장애가 번지는 지점을 재현하고 필요한 출력 경계부터 비교한다.
- nvunixfd: 현재 registry에 있으므로 DS 7.1이라는 이유만으로 없다고 단정하지 않는다.
  최신 NVIDIA 문서는 DS 8.0 이후 기능과 비배치 NvBufSurface 제약을 설명한다. 현재 0.0.1 플러그인과
  API·메타데이터 전달·버퍼 소유권/반환·죽은 소비자 처리의 동등성을 검증해야 한다. [공식 nvunixfd 문서](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvunixfd.html).
- PyServiceMaker 전환: NVIDIA는 기존 Python bindings를 deprecated로 안내하고 신규 개발에
  PyServiceMaker를 권고한다. 기존 7.1/pyds 코드의 즉시 재작성이나 성능 개선을 뜻하지 않는다.
  SDK/JetPack·모델/메타·운영 회귀 비용을 별도 산정한 후 결정한다. [공식 Python bindings 안내](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).
- 검은 프레임 대체: 기본 정책으로 채택하지 않는다. 필요하면 합성 프레임을 따로 식별해 실제
  관측/FPS·추론 완료·경보 근거에서 제외하고, 끊김 전후 인물 추적/낙상 시간창도 재설정한다.

`nvstreamdemux`는 복사 없는 프레임 참조를 만들며 해당 배치의 모든 분리 버퍼가 반환돼야
원래 배치를 돌려준다. 따라서 분기 큐만으로 버퍼 보유·네이티브 오류까지 격리되지는 않는다.
프로세스를 나눠도 같은 GPU·장비 장애와 IPC 버퍼 수명까지 자동 해결되지는 않는다.
[공식 demux의 버퍼 반환 설명](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreamdemux.html)을 기준으로 한 설계 판단이다.

## 4. 작은 검증 단위와 채택 기준

| 순서 | 시험/산출물 | 판단 기준 | 기존 WBS |
|---|---|---|---|
| 1 | 현재 실행 모델/엔진·계측·pyds/SDK·interval 대응표 | 파일/환경 설정과 실행 결과 구분, 모델별 실제 완료를 셀 수 있음 | P5.1, I04/I07 |
| 2 | 격리 1→2→4채널 정상 기준선 | 카메라·모델별 완료/누락·p95 지연·적체·메모리/온도, 동일 입력 반복 비교 | P5/P6.2 |
| 3 | 입력 한 개 끊기·복구 | 정상 입력의 프레임/이벤트가 계속 진행, 카메라 ID 혼선 없음, 복구 시간 기록 | P5.1/P6.3, I20 |
| 4 | 출력 수신처 지연/단절과 인코딩 경계 시험 | 추론/영속 기록 진행, 제한된 메모리·큐, 복구 후 오래된 경보 재실행 없음 | P6.3/P2.4 |
| 5 | 영향이 남는 최소 입력/출력 경계만 분리 | 개선된 복구/지연의 근거가 있고 정확도·처리량·메모리·코드 복잡도 악화가 수용 가능 | P5.1/P6.3 후속 |

새 수치 상한은 임의로 합격 기준에 넣지 않는다. 지연·복구 시간 목표는 D04, 모델 품질은 D07이다.
단절된 카메라의 장애 구간을 정상 30FPS 처리로 계산하지 않으며, 정상 카메라에 전파된 영향은
계속 포함해 보고한다. 디코딩/추론/전송은 경계를 한 단계씩 바꿔 원인을 분리한다.

## 5. 외부 출처와 현재 판단의 한계

- [NVIDIA 포럼 원문](https://forums.developer.nvidia.com/t/recommended-deepstream-architecture-for-fault-isolated-multi-stream-alpr-with-per-source-rtmp-outputs/383413): 2번은 경험 기반 제안이다. 09-20 후속에는 네트워크 출력을 제거한 7~8스트림에서도 정체가 보고돼 원인 조사 중이다. 완성된 검증 결과나 우리 장비의 성능 수치로 인용하지 않는다.
- [DeepStream 7.1 nvinfer](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvinfer.html): TensorRT 실행·모델별 전처리·batch/interval의 의미.
- [GStreamer queue](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html): 큐 상한과 leaky 방향의 정의. 플러그인 내부의 정체를 없애는 보장은 아님.
- [NVIDIA nvstreamdemux](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreamdemux.html): 분리 프레임 참조와 원 배치 반환 시점.
- [NVIDIA nvunixfd](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvunixfd.html): 같은 시스템/디바이스의 비배치 버퍼 전달·메타 직렬화 설정.
- [NVIDIA Python bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps): deprecated 및 PyServiceMaker 권고. 기존 7.1 배포의 즉시 비호환을 의미하지 않음.
- [TensorRT 성능 측정](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html): 엔진 측정·배치·정밀도 후보를 전체 애플리케이션과 함께 비교할 근거.

이번에는 문서·소스·설치/환경값을 읽었다. 실제 카메라 단절·송출 장애·GPU 벤치마크·모델
품질 시험은 실행하지 않아 성능 향상 수치나 장애 격리 합격을 주장하지 않는다.
코드/설정/DB/서비스 재시작은 없다. WBS 상태·연말 목표·P1.4 다음 작업 순서는 유지한다.
검토 결과를 P5.1/P6.3의 입력으로 남기고, 다음 개발 전에 참고할 수 있도록 진행 기록에 연결한다.
