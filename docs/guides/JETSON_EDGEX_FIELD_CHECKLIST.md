# Jetson EdgeX 현장 점검 체크리스트

이 문서는 Jetson Orin 배포 후, EdgeX 연동과 출력 디바이스 전송 경로를 현장 테스트 전에 점검하기 위한 운영 체크리스트입니다.

## 현재 구조 평가

현재 저장소는 현장 테스트를 진행할 수 있는 수준의 배포 자산을 이미 갖추고 있습니다.

- Jetson 전용 배포 파일: [docker-compose.jetson.yml](../../docker-compose.jetson.yml)
- Jetson 전용 컨테이너 이미지: [Dockerfile.jetson](../../Dockerfile.jetson)
- TLV 파서: [Dockerfile](../../Dockerfile)의 `parser` build target
- EdgeX 어댑터: [runners/run_edgex_adapter.py](../../runners/run_edgex_adapter.py)
- 센서 규칙 브리지: [runners/run_sensor_rule_bridge.py](../../runners/run_sensor_rule_bridge.py)
- Action Layer: [runners/run_action_bridge.py](../../runners/run_action_bridge.py)
- Alert API: [runners/run_alert_api.py](../../runners/run_alert_api.py)
- TLV Parser DB 환경값: `.env.jetson`

즉, 방향은 맞습니다.  
다만 현장 안정성을 위해서는 배포 성공 여부보다 아래 3가지를 확인해야 합니다.

1. `AI -> MQTT -> EdgeX Adapter -> EdgeX` 경로가 실제로 이어지는지
2. `TLV -> aiot-parser -> aiot/sensors/# -> sensor-rule-bridge -> aiot/rules/sensor/#` 경로가 실제로 이어지는지
3. `Rule/Alert -> Action Layer -> 스피커/전광판/사이렌` 경로가 실제 장비까지 닿는지
4. 장애 시 `outbox` 와 재시도가 기대대로 동작하는지

<a id="orin-nx16-compatibility"></a>
## Orin NX 16GB 대응 기준 — 2026-10-19

사용자는 Jetson Orin NX 16GB로 전환할 가능성을 알렸다. 공통 소스와 이벤트/API/DB 계약을 유지하고 보드별 배포 설정·TensorRT 엔진을 분리한다. NX의 JetPack·캐리어보드·저장장치·전력/냉각 조건은 미확인이고, 실제 장비 전환·설치 합격·성능 합격은 아직 아니다.

| 항목 | 현재 확인·NX 대응 |
| --- | --- |
| 소프트웨어 | NVIDIA는 DeepStream7.1의 AGX Orin/Orin NX 지원을 명시한다. 프로젝트 현재 호스트는 AGX Orin64GB·L4T36.5.2이며 CUDA12.6/TensorRT10.3/DeepStream7.1 조합이다. NX도 우선 같은 컴퓨트 스택을 후보로 삼고 캐리어보드 BSP 지원과 실제 로드를 확인한다. DeepStream7.1 공식 기준표의 JetPack은6.1이며, 현재 호스트 조합의 관측을 NX 인증으로 확대하지 않는다. |
| 모델 | 학습 가중치/ONNX·전처리·후처리를 공유한다. AGX의 `.engine` 및 TensorRT 캐시를 NX 합격품으로 복사하지 않고, NX에서 같은 ONNX·입력 크기·정밀도·batch 조건으로 재생성하여 별도 보존한다. JetPack의 TensorRT 하드웨어 호환 모드를 이 절차의 대안으로 가정하지 않는다. |
| 런타임 의존성 | 현재 Docker 이미지는 호스트 DeepStream/NVIDIA 라이브러리와 `.venv-jetson-train`의 GPU PyTorch에 의존한다. Docker 이미지 복사만으로 이식이 완료되지 않는다. 아래 경로 변수로 NX의 실제 설치 위치를 선택하며 모델·설정·GPU venv 누락 시 빈 디렉터리 생성 대신 기동을 거부한다. |
| 자원 | 16GB는 CPU/GPU가 공유한다. 모델·디코드/인코드 버퍼·DB/Redis·운영체제의 사용량과 최대 부하를 함께 측정한다. AGX의 Docker 단발 관측에서 AI2.832GiB·Redis932.2MiB였지만 이것은 NX의 전체/최대 메모리 예산이 아니다. 먼저 기존 큐/보관/모델 상주 설정을 활용하며 임의 RAM 제한·swap으로 성능 합격을 만들지 않는다. |
| 배포·전력 | 앱·Compose를 공통으로 유지하고 NX용 env에는 실제 차이만 둔다. 현재 장비의 절대 경로·모델 캐시·전력모드 숫자·CPU affinity·부팅 unit을 그대로 복사하지 않는다. NX 모델/캐리어보드에 맞는 설정을 확인한 뒤 적용한다. |
| 성능·품질 | 1080p·기존 신뢰도·실물4카메라 목표와 내부3초목표/5초상한을 자동 축소하지 않는다. NX에서1→2→4채널의 실제 추론율·경보 지연·정답 회귀·메모리/온도·30분/장시간·재시작을 따로 판정한다. 현재 AGX에서도 후보의 지연변동 기준이 미달이므로 NX의 동일 성능을 보장하지 않는다. |

10-19 사용자 진행 지시에 따라 **보드별 경로 선택·NX env 예시·엔진 로드 사전 검사**를 구현했다. 기존 AGX 경로가 기본값이며 실행 중인 컨테이너·모델·전력은 교체하지 않았다. 공통 추론/API/DB 코드는 그대로다. 기존 P5.2 보완→다중 검증→운영 준비 순서를 유지하며, 실제 NX 설치/엔진 생성/성능·복구 합격은 P6.2~P6.4와 I07의 잔여 증거다.

### NX 배포 설정 준비

[.env.jetson.nx16.example](../../.env.jetson.nx16.example)은 **기존 후보의 `.env.jetson`에 병합할 경로 4개만** 담는다. 단독 env로 실행하거나 운영 env 전체를 덮어쓰지 않는다. 실제 NX의 프로젝트 경로에서 아래 값을 적용한다.

| 변수 | NX 예시 | 미지정 시 기존 AGX 값 |
| --- | --- | --- |
| `MODELS_HOST_PATH` | `./data/deploy/orin-nx16/models` | `./models` |
| `DEEPSTREAM_CONFIG_HOST_PATH` | `./data/deploy/orin-nx16/config/deepstream` | `./config/deepstream` |
| `JETSON_TORCH_VENV_HOST_PATH` | `./.venv-jetson-train` (NX에서 새로 준비) | `./.venv-jetson-train` |
| `TRT_CACHE_VOLUME` | `edgex-jetson-nx16_trt-cache` | `edgex-jetson_trt-cache` |

컨테이너의 `/app/models`, `/app/config/deepstream`, `/app/.venv-jetson-train` 경로는 같다. 경로 변수는 반드시 `--env-file .env.jetson`으로 Compose에도 전달한다. 동일 호스트에서 AGX/NX 스택을 동시에 띄우는 기능은 아니며 컨테이너 이름·포트·DB 볼륨은 기존 구성을 사용한다.

1. NX의 JetPack/BSP를 확인하고 해당 ARM64/Python3.10/CUDA 환경에 맞는 GPU PyTorch venv와 SDK를 준비한다. AGX venv의 복사는 설치 재현으로 인정하지 않는다. 기존 `DEEPSTREAM_ROOT`/`NVIDIA_LIB_ROOT`, 데이터셋·카메라·출력 장치·X11·systemd 경로도 대상 장비에서 확인한다.
2. 고정 후보의 모델 트리에서 ONNX/가중치/RF·TCN·라벨을 NX 모델 경로에 배치하고 SHA256을 대조한다. 절대 경로의 심볼릭 링크가 AGX 파일을 가리키지 않게 한다. **기존 `.engine`/캐시는 가져오지 않는다.** NX에서 같은 ONNX·입력 이름/크기·정밀도·batch/profile로 엔진을 생성하고 명령·버전·해시를 기록한다. 아래 단일 모델 변환기로 명시한 조건을 먼저 검증한다.
3. 고정 후보의 실제 활성 nvinfer 설정·tracker·라벨 파일을 NX 설정 경로에 배치한다. 상대 모델 경로는 컨테이너의 공통 구조를 기준으로 유지한다. 예시 기본 config를 현재 활성 모델 목록으로 간주하지 않는다.
4. NX에서 별도 캐시 볼륨을 생성하고 Compose 해석을 검사한다. 기존 다른 external 볼륨·DB/비밀값 준비는 기존 설치·복구 절차를 따른다.

```bash
rtk proxy docker volume create edgex-jetson-nx16_trt-cache
rtk proxy docker compose --env-file .env.jetson -f docker-compose.jetson.yml config -q
```

대상 NX에서 먼저 읽기 전용으로 확인할 명령:

```bash
rtk proxy cat /proc/device-tree/model
rtk proxy cat /etc/nv_tegra_release
rtk proxy nvpmodel -q
rtk proxy free -h
rtk proxy python3 scripts/health/check_deepstream_env.py --native-only --timeout 15 --json
```

이미지·호스트 의존성·엔진 준비 후 **대상 NX에서** 다음을 실행한다. `run --no-deps`는 앱 대신 점검 명령만 실행하지만 GPU 메모리를 사용하므로 운영 부하 시험과 겹치지 않게 한다. 아래 `--config`는 실제 활성 목록으로 지정한다. 새 이미지가 없으면 대상 NX에서 먼저 `docker compose --env-file .env.jetson -f docker-compose.jetson.yml build cctv-ai-engine`을 수행한다.

```bash
rtk proxy docker compose --env-file .env.jetson -f docker-compose.jetson.yml run --rm --no-deps --entrypoint python3 cctv-ai-engine /app/scripts/health/check_deepstream_env.py --load-engines --batch-size 1 --timeout 15 --config /app/config/deepstream/config_infer_primary.txt --json
rtk proxy docker compose --env-file .env.jetson -f docker-compose.jetson.yml run --rm --no-deps --entrypoint python3 cctv-ai-engine /app/scripts/health/check_torch_cuda.py
```

`--load-engines`는 신뢰하는 후보 엔진만 순차 역직렬화하며 빈 파일/디렉터리·TensorRT 누락·플러그인 초기화/로드 실패를 종료 코드1로 반환한다. 기본 검사와 systemd의 `--native-only`는 GPU 엔진을 로드하지 않는다. 로드 통과는 NX에서 빌드했다는 증명이나 실제 추론/동시 메모리/성능 합격이 아니다. 별도 custom plugin이 있는 모델은 해당 라이브러리를 포함한 실제 파이프라인에서도 검증한다. [NVIDIA TensorRT 로드 API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html).

배치 사전점검은 단일 4차원 영상 입력 엔진의 **profile 0 min/opt/max**와 요청 배치를 대조한다. `--load-engines`만 지정하면 설정 파일의 `batch-size`(누락 시 1)를 검사한다. 앱은 Primary 배치를 카메라 수로 덮어쓰므로, 위 단일 입력 예시의 `--batch-size 1`을 실제 입력 수 2 또는 4로 바꿔 검사한다. 설정 파일의 참고값 8을 실제 요청값으로 오인하지 않는다. `--batch-size`는 양수이며 `--load-engines`와 단일 `--config`를 함께 지정해야 한다. SGIE는 카메라 수와 독립적인 실제 ROI 배치를 별도로 검사한다. 배치 불일치·지원하지 않는 입력 형상·profile 조회 실패는 종료 코드1이며 엔진을 자동 생성하거나 교체하지 않는다. 이 검사는 전체 입력 크기·출력 형상·추론 품질·처리율 또는 카메라별 미리보기/RTSP 분리까지 보증하지 않는다. [NVIDIA 다중 입력 배치 설정 기준](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html).

최종 4카메라용 Primary 후보는 **min1/opt4/max4**로 준비하고, 같은 엔진으로 실행 배치 1→2→4를 검증한다. nano640 원본은 동적 입력이며 별도 후보의 GPU 생성(52.14초)·batch1/2/4 raw 추론 검증을 통과했다. 한 영상의 서로 다른4프레임에서 후보 단건/묶음 출력차0을 확인했으나, 기존 엔진 대비 수치 차이가 있어 정답 기반 정확도·공유4입력 앱·메모리/추론율/지연 합격은 별도다. 기존 운영 엔진은 profile1/1/1이며 자동 교체하지 않는다. [max4 후보·병렬 작업 인계](../planning/handoffs/engine_batch_20261023.md).

### 장비 준비 전 로컬 결과 — 10-19

사용자가 **NX 실장비 준비 전·로컬 준비 계속**을 확인했다. 기존 검사기에 `--files-only`를 추가해 SDK/GPU 없이 설정·엔진·라벨·tracker 파일을 점검한다. 실제 앱과 같은 `config_tracker_iou.yml`을 검사하며, nvinfer 경로는 `[property]`에서만 읽는다. 이전에는 다른 섹션의 값이 모델 경로를 덮을 수 있었다. 수정 전5시험 실패 재현 후 **관련32시험·Ruff PASS**, 현재 기본 설정8개의 파일 검사도 통과했다. 파일 검사는 엔진 호환성·정확도·메모리 합격을 뜻하지 않는다.

현재 개발 장비의 [NX 모델 입력 목록](../../data/deploy/orin-nx16/build_inputs.json)에 **22파일/255,872,413bytes(약244MiB)**를 준비하고 원본/복사본 SHA256을 대조했다. 최초 ONNX6개(Helmet·Person·Pose nano/medium·PA100K·PPHuman), nvinfer 설정8개, 라벨2개, IOU tracker·streammux의18파일에 후속 RF2개·TCN/Pose PT2개를 추가했다. 복사본은 `data/deploy/`의 로컬 산출물로 Git에서 제외하며, 최초 파일별 해시·입력 이름/shape는 [진행 데이터](../planning/figma_exports/progress_v138.json)의 `orin_nx16_compatibility.local_preparation`, 추가 파일은 `dependency_preparation`에 보존한다.

이 묶음은 **모델 원본·설정 준비본**이다. AGX 엔진/캐시/venv·배포 이미지·얼굴 등록 데이터·DB·비밀값을 포함하지 않는다. 실제 후보가 확정되면 이미지·실행 환경·카메라/출력장치 설정과 데이터 이행까지 확인해야 한다. 기본 Primary nano640 설정과 P5.2 medium960 후보를 혼동하지 않는다. medium960 원본 SHA는 기존 빌드 기록과 일치했으며, manifest의 `fall_candidate_960.build_argv_on_target`은 FP16·input1×3×960×960·workspace1024·최적화0의 **NX용 실행 초안**이다. NX에서 실행하지 않았고 batch1 조건을 다중 카메라에 사용하지 않는다. 후보 설정을 기본 설정에 자동 적용하지 않았다.

```bash
# 현재 프로젝트 기본 설정의 파일 구성만 확인한다.
rtk proxy python3 scripts/health/check_deepstream_env.py --files-only --json
# NX 준비 폴더 검사: 엔진을 제외했으므로 현재는 model-engine-file 누락으로 종료1이 정상이다.
rtk proxy python3 scripts/health/check_deepstream_env.py --files-only --root data/deploy/orin-nx16 --json
```

준비본의 설정8개를 모두 지정한 검사에서 설정/라벨/tracker는 통과하고 엔진8참조만 누락됐다(고유 엔진5개). `--config`는 여러 번 지정할 수 있다. NX 엔진 생성 후 해당 검사를 통과시키고 `--load-engines`·실제 파이프라인 시험으로 이어간다. `--files-only`와 `--native-only`/`--load-engines`는 동시에 사용할 수 없다.

### 모델별 엔진 생성 명령 준비 — 10-19 후속

기존 변환기의 모든 YOLO 기본320·누락 원본 건너뛰기·엔진 덮어쓰기를 제거했다. [변환기](../../scripts/convert/convert_onnx_to_engine.py)는 **원본/대상 경로·SHA256·입력 이름·min/opt/max shape**를 필수로 받는다. 기본은 GPU 없는 dry-run이며, 대상 NX에서 `--build`를 추가할 때만 생성한다. 프로젝트의 기존 ONNX 변환용 패키지를 사용하며 새 의존성은 추가하지 않았다. NX의 변환용 Python에도 `onnx` 패키지가 필요하다. 입력 한 개의 NCHW 단일 ONNX만 지원하며 별도 가중치 파일을 참조하는 모델은 거부한다.

관련 **23시험·Ruff PASS**, 실제 준비 ONNX6개 dry-run PASS·18파일 SHA 재대조 PASS를 확인했다. 빌더 실패·빈 엔진·생성 중 원본 변경은 성공으로 처리하지 않으며 임시 파일을 정리한다. 기존 파일/디렉터리/끊어진 링크와 검사 후 생긴 경쟁 출력도 보존한다. 실제 생성은 모의 빌더로 검증했으므로 TensorRT 파싱/생성 합격과 구분한다.

| 준비 모델 | 입력 이름 | H×W | 생성 계획 min/opt/max batch |
| --- | --- | --- | --- |
| Helmet | `input` | 320×320 | 1/8/8 |
| Person·Pose nano | `input` | 640×640 | 1/8/8 |
| PA100K | `images` | 256×192 | 1/8/8 |
| PPHuman | `x` | 256×192 | 1/8/8 |
| Pose medium 참고 후보 | `input` | 960×960 | 1/1/1 |

기본 설정의 batch8에 맞춘 **생성 계획**이며 실제 모델 연산자의 batch 지원/메모리나 처리율은 아직 검증하지 않았다. medium960은 참고 후보 조건이며 기본 Primary 설정에 자동 연결하지 않는다. 프로젝트 루트에서 실행할 예:

```bash
rtk proxy .venv/bin/python scripts/convert/convert_onnx_to_engine.py \
  --onnx data/deploy/orin-nx16/models/fall/yolov8m-pose.onnx \
  --engine data/deploy/orin-nx16/models/fall/medium_960_fp16.engine \
  --sha256 53e9514c3fd900b32702f95d696689d031feb4603141941ca45fbf194b3cce40 \
  --input-name input --min-shape 1x3x960x960 \
  --opt-shape 1x3x960x960 --max-shape 1x3x960x960
```

대상 NX의 SDK/버전을 기록하고 같은 명령에 `--build`를 추가한다. 기본 FP16·workspace1024MiB·최적화0·avgTiming1 조건이며 **workspace는 프로세스 전체 메모리 제한이 아니다**. TensorRT10.x `trtexec` 위치는 `--trtexec` 또는 `TRTEXEC`으로 지정한다. [NVIDIA trtexec 10.x 옵션](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-trtexec.html). 생성 성공 JSON의 `BUILT_NOT_VALIDATED`와 엔진 SHA는 생성 사실만 뜻한다. 이어서 `--files-only` → `--load-engines` → 실제 품질·메모리·성능/복구를 확인한다. 무인자 일괄 변환과 `TRT_INPUT_NAME`/`TRT_*_IMGSZ` 환경변수는 폐지했으며 참조하던 PPHuman 안내도 갱신했다.

실제 6명령은 로컬 manifest의 `conversion_plans`와 진행 데이터의 `orin_nx16_compatibility.engine_preparation`에 보존했다. 운영 컨테이너를 읽기 전용 대조한 결과, 활성 설정은 기본 설정 폴더가 아닌 `data/runtime/live_refresh_20261008/frozen/config/deepstream`을 사용한다. 최초에는 RF2개·TCN 가중치·Pose PT를 누락 목록으로 기록했고, 아래 후속 작업에서 복사·대조를 마쳤다. 얼굴 설정 `opencv`는 현재 소스에서 Haar 경로를 선택하므로 YuNet/SFace2개를 필수로 간주하지 않았다. 해당 선택 모델의 원본 존재/해시도 별도로 기록했다. 얼굴 갤러리·이미지·DB 내용과 비밀값은 수집/복사하지 않았다.

### 보조 모델·운영 설정·설치 조건 대조 — 10-19 후속

운영 보존 폴더의 **nvinfer8개·라벨2개·IOU tracker·streammux 총12파일은 준비본과 SHA까지 동일**했다. 폴더 위치가 다른 것을 설정 내용 차이로 해석하지 않는다. 구형 tracker2개와 crop 예시는 현재 준비 경로에서 사용하지 않아 추가하지 않았다. 고정 후보의 환경변수·후처리 소스·모델 선택이 모두 동일하다는 판정은 아니다.

| 추가 원본 | 준비 경로 (`models/` 기준) | 용도·선택 조건 |
| --- | --- | --- |
| Pose RF | `experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl` | 관측된 인라인 RF 비교 모델 |
| TCN | `fall/fall_temporal_same_split_resume_20260908.pt` | 관측된 TCN 비교 모델 |
| Pose PT | `fall/yolov8n-pose.pt` | 보조 TCN의 Pose 추출 원본 |
| 구형 MediaPipe RF | `legacy/falldata_mediapipe/FNF_RF_SMOTE_CAM_1.pkl` | 선택형 레거시 재현용; 현재 인라인 RF 분기의 필수 모델로 세지 않음 |

4파일을 원본 해시 확인 후 **일반 파일로 복사**했고, 기존18개를 포함한22개를 전부 대조했다. `.engine`·심볼릭 링크는0개이며 pickle/PT 로드·추론은 하지 않았다. 운영은 `FALLDATA_AUX_INLINE_POSE_RF=true`, `FALLDATA_AUX_MODE=shadow`이며 현재 소스의 `verify()`는 인라인 RF 분기에서 반환한다. TCN·보조 기능의 실제 NX 실행은 별도 확인한다.

Compose의 `FALLDATA_AUX_MODEL_PATH` 기본값을 `/app/models/legacy/falldata_mediapipe/FNF_RF_SMOTE_CAM_1.pkl`, `FALLDATA_AUX_TEMPORAL_POSE_MODEL_PATH`를 `/app/models/fall/yolov8n-pose.pt`로 맞추고 env 예시도 갱신했다. 기존 외부 데이터셋 경로와 모델 루트 별칭에 의존하지 않게 하기 위함이다. 명시한 환경변수는 계속 우선한다. **관련11시험 PASS·예시/현재 AGX/NX 경로/명시적 덮어쓰기의 Compose 해석4경우 PASS**를 확인했다. 운영 env·컨테이너는 적용하지 않았다. 파일 검사22항목은14통과·엔진8참조(고유5개) 누락으로 예상대로 종료1이다.

설치 조건은 기존 패키지의 **메타데이터 관측값**으로 기록했으며 NX 설치 성공이나 GPU 로드 검증이 아니다.

| 실행 경로 | AGX에서 관측한 패키지·NX 준비 조건 |
| --- | --- |
| AI 컨테이너 인라인 RF | NumPy1.26.4·SciPy1.15.3·scikit-learn1.7.2·joblib1.5.3. 후보 이미지/requirements를 고정한 뒤 NX에서 같은 RF 원본 로드·`predict_proba` 확인 |
| `.venv-jetson-train` | torch2.8.0·torchvision0.23.0·Ultralytics8.4.104·NumPy1.26.4 관측. NX의 Python/CUDA에 맞게 새로 설치하고 `check_torch_cuda.py`와 실제 TCN/Pose 시험 필요 |
| 선택형 `.venv-falldata` | 기존 검사에서 **scikit-learn1.7.2로 실패**(요구1.3.2 이상·1.4 미만). `requirements/falldata-model.txt`는1.3.2 고정. 레거시를 선택할 때 별도 환경을 재현하며 현재 인라인 RF 환경을 다운그레이드하지 않음 |
| 선택형 `.venv-mediapipe` | MediaPipe0.10.14·NumPy2.2.6 관측. 레거시 RF와 NumPy 요구가 다르므로 기존 분리 구조 유지; NX 설치/실행 재검증 |

운영 컨테이너 ID·이미지·시작시각이 대조 전후 동일했고 가동 상태를 유지했다. **GPU 실행·운영 재시작/교체0·P6.4 PLANNED·I07 OPEN**이다. 다음은 고정 후보의 이미지·환경·데이터 이행/복구 준비이며, NX 도착 후 SDK/Python 설치 → 엔진 생성/로드 → 1→2→4채널 품질·16GB 메모리·성능/복구를 판정한다.

### 후보 이미지 선택·장치 이력 복구 — 10-19 후속

기존 `docker-compose.jetson-candidate.yml`의 이미지·GPU venv·화면 세션이 AGX 값으로 고정돼 있었다. 이미지에는 `CCTV_CANDIDATE_IMAGE_ID`, GPU venv에는 공통 `JETSON_TORCH_VENV_HOST_PATH`, 화면에는 공통 `DS_DISPLAY`를 적용했다. 미지정 시 기존 AGX 후보 기본값을 유지한다. 모델은 후보 전용 `CCTV_CANDIDATE_MODEL_ROOT`를 명시하며, NX 준비본의 빈 `models/trt_cache/` 연결 디렉터리도 마련했다. 이 디렉터리에는 AGX 엔진이나 캐시를 복사하지 않았다.

후보 검사기의 `--expected-image-id`는 **검증 기록에서 선택한 전체 `sha256:` 이미지 ID**와 병합 구성을 대조한다. 태그·잘린 ID를 거부하고 기존 DB 영속 경로·읽기 전용 모델·소스 덮기 금지 검사를 유지한다. 이미지 ID를 구성에서 자동 추출해 기대값으로 사용하지 않는다. 대상 장비에서 사용할 이미지 ID와 모델 경로를 셸 변수로 설정하고, NX 경로를 병합한 env 파일을 준비한 뒤 다음처럼 비밀값을 파일에 저장하지 않고 검사한다.

```bash
rtk proxy docker compose --env-file .env.jetson \
  -f docker-compose.jetson.yml -f docker-compose.jetson-candidate.yml config --format json | \
  rtk proxy .venv/bin/python scripts/health/check_compose_runtime_assumptions.py \
    --candidate-compose-json - \
    --expected-image-id "${CCTV_CANDIDATE_IMAGE_ID:?검증 기록의 전체 이미지 ID 필요}" --json
```

`CCTV_CANDIDATE_IMAGE_ID`·`CCTV_CANDIDATE_MODEL_ROOT`는 Compose에도 전달되도록 내보낸 환경변수여야 한다. `.env.jetson.nx16.example`은 경로 병합 예시이며 완전한 후보 환경파일이 아니다. 후보 overlay의 기존 판정 설정은 과거 AGX 시험용이므로 현재 낙상 후보 r4의 설정으로 간주하지 않는다. 새 이미지의 소스·모델·설정·호스트 SDK 조합을 고정하고 별도 실행 검증을 통과해야 한다.

**후보/Compose 관련78시험·Ruff PASS**, 실제 Compose 해석과 검사기 CLI의 AGX 기본값/NX 경로 두 경우도 통과했다. NX 경우에는 **가상 이미지 ID와 임시 빈 venv 디렉터리**를 사용해 선택·연결 규칙만 확인했다. 이미지 존재·빌드·Python 실행·엔진 로드는 검사하지 않았으며 NX 기동 합격이 아니다.

다음 작업인 장치 이력 복구도 기존 백업 시험에 연결했다. 전광판·스피커·사이렌 서비스와 모의 장치로 생성한 DB를 기존 백업 스크립트로 보존한 뒤 새 경로에 복원했다. 완료 명령 재접수 시 출력0회, 결과 저장 실패로 미확정인 명령 재출력 차단, 복원 후 신규 명령 저장·재시작 뒤 중복 차단, 원본/스냅샷 보존을 확인했다. **백업 시험14개 PASS(새 서비스 복구3경우 포함)**다. 실제 출력·운영 DB 복원·다른 이미지/버전으로의 복귀는 실행하지 않았다. [복구 범위와 제약](../handover/BACKUP_RESTORE_RUNBOOK.md#nx-device-restore-20261019).

합계 **고유92시험 PASS**, 운영 ID·이미지·시작시각 동일, GPU 실행·이미지 빌드·컨테이너 교체0이다. 다음은 실제 대상 SDK에 맞는 후보 이미지/설치 환경 고정과 전체 데이터·설정 이행이며, NX 실장비에서는 엔진 생성/로드 → 1→2→4채널 품질·16GB 메모리·성능/복구를 확인한다. P6.4 PLANNED·I07 OPEN을 유지한다.

근거: [DeepStream7.1 지원표](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html#platform-and-os-compatibility), [JetPack6.2.3 구성](https://developer.nvidia.com/embedded/jetpack-sdk-623), [TensorRT 엔진 호환 제약](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html#hardware-compatibility), [Orin 제품군의 공유 아키텍처·메모리 사양](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/).

## 지금 꼭 확인해야 하는 항목

### 1. JetPack / L4T 버전 정합성

[Dockerfile.jetson](../../Dockerfile.jetson)의 베이스 이미지는 `l4t-jetpack:r36.4.0`이며,
2026-10-02 확인한 운영 호스트는 JetPack 6.2.3/L4T 36.5.2·CUDA 12.6·TensorRT 10.3·cuDNN 9.3·DeepStream 7.1입니다.
호스트와 이미지 태그를 같은 버전이라고 해석하지 않습니다. 현재 검증한 조합을 기준으로 유지하고,
새 장비에서는 아래 설치·실행 검사를 다시 통과해야 합니다.
[NVIDIA JetPack 6.2.3 지원 정보](https://developer.nvidia.com/embedded/jetpack-sdk-623).

### 1-1. 호스트와 컨테이너 의존성 준비

2026-10-02 사용자 결정: 별도 Jetson 설치 재현은 프로젝트 기능 완성 후·정식 배포 전에 수행합니다.
현재 장비 점검을 새 장비 설치 합격으로 표시하지 않습니다. 같은 날 RF 활성화/모델 종류는
현재 운영값이 재생성 후에도 유지되도록 `.env.jetson`에 명시하고 Compose 해석으로 확인했습니다.
외형 모델 경로와 기존 배포 검사 기준의 불일치는 [I46](../planning/STATUS.md)에 남아 있습니다.

현재 Compose는 호스트의 `/opt/nvidia/deepstream`과 NVIDIA 라이브러리를 연결합니다.
Docker 이미지 배포만으로 호스트 준비가 끝나지 않습니다. 대상 Jetson에는 먼저 지원하는
L4T/드라이버·Docker/NVIDIA Container Toolkit·DeepStream 7.1 SDK를 준비해야 합니다.
SDK 배포 파일은 NVIDIA의 Jetson용 패키지를 사용하고 임의로 버전을 올리지 않습니다.
[공식 설치 절차](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html#jetson-setup).

호스트 SDK 설치 후에는 다음 도구로 필수 Ubuntu 런타임 패키지를 준비합니다.
기본 실행은 모의 설치이며 `--apply`에서만 실제 설치합니다. SDK/JetPack 교체 및 패키지 제거는 하지 않습니다.

```bash
rtk proxy bash scripts/ops/install_jetson_deepstream_dependencies.sh --dry-run
rtk proxy sudo bash scripts/ops/install_jetson_deepstream_dependencies.sh --apply
rtk proxy python3 scripts/health/check_deepstream_env.py --native-only --timeout 15 --json
rtk proxy docker exec cctv-ai-engine python /app/scripts/health/check_deepstream_env.py --native-only --timeout 15 --json
```

검사 항목은 실제 `deepstream-app --version-all` 실행/SDK 7.1 버전과
`nvstreammux`, `nvinfer`, `nvtracker`, `nvv4l2decoder`, `nvvideoconvert` 로드입니다.
하나라도 실패하면 종료 코드 1로 배포 준비 실패를 표시합니다.
컨테이너의 `libgstrtspserver-1.0-0` 등은 Dockerfile 설치 목록에 포함돼 있습니다.
모델/Python 바인딩 검사에는 `--native-only`를 빼고 실제 사용하는 `--config`를 지정합니다.
기본 config 목록을 현재 활성 모델 목록으로 간주하지 않습니다.

2026-10-02에는 호스트의 `libgstrtspserver-1.0.so.0` 누락을 재현했고,
사용자가 도구로 설치한 뒤 실제 호스트·컨테이너 점검을 수행했습니다.
실제 GPU/디스플레이 접근이 제한된 샌드박스의 timeout은 호스트 설치 실패와 구분합니다.

### 1-2. X11 준비와 자동 시작

현재 EGL 경로는 로그인 세션의 X11 접근을 필요로 합니다.
`deploy/systemd/cctv-jetson-stack.service`의 계정·작업 경로·`DISPLAY`·`XAUTHORITY`를 대상 장비에 맞춥니다.
`prepare_jetson_display.sh`는 인증 성공까지 재시도하고 로컬 root만 허용합니다.
서비스는 X11 준비 → 호스트 SDK 검사 → Compose 시작 → 컨테이너 SDK 검사 순서입니다.
SDK 검사 통과만으로 실제 영상 처리나 재부팅 복구까지 합격한 것은 아닙니다.

```bash
rtk proxy systemd-analyze verify deploy/systemd/cctv-jetson-stack.service
rtk proxy sudo install -m 0644 deploy/systemd/cctv-jetson-stack.service /etc/systemd/system/cctv-jetson-stack.service
rtk proxy sudo systemctl daemon-reload
```

설치 후 서비스 활성화 여부를 확인합니다. 위 명령은 실행 중인 스택을 재시작하지 않습니다.
이 서비스의 `start/restart`는 AI·Public API·UI를 재생성하므로 운영 복구 시험 시간에 수행합니다.
새 로그인 세션/재부팅 후 X11 인증, 자동 시작, 실제 프레임 증가를 확인해야 합니다.

### 2. 민감 정보 외부 분리

`.env.jetson`에 스피커 비밀번호가 평문으로 들어갈 수 있습니다.

- 현장 배포 전에는 `.env.jetson`을 실운영용 비밀 파일로 분리하는 것을 권장합니다.
- 저장소에는 예시 파일만 남기고, 실제 장비 비밀번호는 배포 서버의 별도 env 파일이나 비밀 저장소로 옮기는 편이 안전합니다.

### 3. AI 엔진 / 어댑터 헬스체크 강도

현재 `cctv-ai-engine`은 Zone API `8765/health`를 확인하는 HTTP healthcheck를 사용합니다. `cctv-edgex-adapter`는 아직 `kill -0 1` 기반 프로세스 생존 확인만 사용합니다.

- DeepStream의 AI health는 실행 상태와 실제 처리 프레임의 나이를 확인합니다.
  첫 프레임이 없거나 30초 넘게 멈추거나 통계 조회에 실패하면 HTTP 503입니다.
  전체 파이프라인 진행 검사이며 카메라별 영상 품질·모델 정확도·MQTT 전달 합격을 뜻하지 않습니다.
- 어댑터는 프로세스가 살아 있어도 MQTT 또는 EdgeX 연결이 끊긴 상태일 수 있습니다.
- 현장에서는 반드시 별도 점검 스크립트로 `MQTT`, `Core Metadata`, `Core Data`, `Action Layer`를 함께 확인해야 합니다.

## 추가한 점검 스크립트

현장 점검용 스크립트를 추가했습니다.

- [scripts/health/check_jetson_edgex_stack.py](../../scripts/health/check_jetson_edgex_stack.py)

기본 점검 항목:

- MQTT 브로커 `1883`
- Redis `6379`
- AIoT Parser DB `5432`
- EdgeX Core Metadata `/api/v3/ping`
- EdgeX Core Data `/api/v3/ping`
- AIoT Parser `/health`
- Alert API `/health`
- Action Layer `/health`
- Public API `/api/v1/health`
- 선택적으로 Public API `/api/v1/appearances/status`
- 선택적으로 스피커 / 전광판 / 사이렌 TCP 연결

예시:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py --host 127.0.0.1
```

외형 검색 상태까지 포함:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py \
  --host 127.0.0.1 \
  --check-appearance-status \
  --public-api-key <PUBLIC_API_KEY>
```

`/api/v1/appearances/status`의 응답 해석 기준은
[../features/APPEARANCES_STATUS_API.md](../features/APPEARANCES_STATUS_API.md)
문서를 기준으로 확인하는 것을 권장합니다.

출력 장비까지 포함:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py \
  --host 127.0.0.1 \
  --speaker-host 192.168.88.92 --speaker-port 80 \
  --signboard-host 192.168.88.91 --signboard-port 5000
```

JSON 출력:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py --json
```

DeepStream Python 모듈까지 확인하려면 `--deepstream`을 추가합니다. 이 검사는 명령을 실행한 호스트 Python 환경의 `gi`, `pyds` import를 확인하므로, 컨테이너 내부 런타임을 검증하려면 컨테이너 안에서 실행하거나 `check_deepstream_env.py`를 함께 사용합니다.

## 권장 현장 점검 순서

1. Jetson 호스트에서 GPU 런타임과 Docker를 확인합니다.
2. `docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build`로 스택을 올립니다.
3. `docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps`로 컨테이너 상태를 확인합니다.
4. AI 엔진만 재시작할 때는 `docker compose --env-file .env.jetson -f docker-compose.jetson.yml restart cctv-ai-engine`을 사용합니다.
5. `.venv/bin/python scripts/health/check_jetson_edgex_stack.py ...`로 인프라 상태를 확인합니다.
6. 외형 검색을 운영할 경우 `--check-appearance-status` 옵션으로 `appearances/status`도 함께 확인합니다.
7. `cctv-ai-engine` 로그에서 카메라 입력과 MQTT 발행 여부를 확인합니다.
8. `cctv-edgex-adapter` 로그에서 카메라 등록, Core Metadata/Core Data 연결 여부를 확인합니다.
9. `cctv-action-layer` 로그에서 MQTT 구독, 장비 제어 성공 여부를 확인합니다.
10. 실제 이벤트 1건을 발생시켜 스피커/전광판/사이렌 반응을 확인합니다.
11. MQTT 또는 EdgeX를 잠시 끊었다가 복구해 outbox 재전송이 동작하는지 확인합니다.

## 현장 테스트 전 최종 권장사항

반드시 필요한 항목:

- Jetson JetPack 버전과 컨테이너 기준 버전 재확인
- `.env.jetson`의 민감 정보 외부 분리
- 카메라 RTSP 주소 고정 확인
- 스피커 / 전광판 / 사이렌 IP 고정 확인
- 현장 네트워크에서 `1883`, `8000`, `8080` 접근 정책 확인
- 장애 복구 테스트 1회 수행

있으면 더 좋은 항목:

- `cctv-ai-engine`, `cctv-edgex-adapter`에 더 강한 readiness 체크 추가
- 현장용 `docker compose logs` 수집 스크립트 추가
- 테스트 이벤트 자동 발생 스크립트 추가

## 한 줄 결론

현재 저장소는 Jetson + EdgeX + 출력 디바이스 현장 테스트를 진행할 수 있는 구조입니다.  
다만 배포 전에 `버전 정합성`, `비밀값 분리`, `실제 네트워크/장비 연결 점검 자동화` 세 가지는 꼭 챙기는 것이 좋습니다.
