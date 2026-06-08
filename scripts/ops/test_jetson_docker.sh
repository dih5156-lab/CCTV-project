#!/usr/bin/env bash
# ==============================================================================
# test_jetson_docker.sh — Jetson Docker 환경 검증 스크립트
#
# 대상: JetPack 6.2 (L4T R36.4.x) / CUDA 12.6 / TensorRT 10.3.0
#
# 테스트 항목:
#   1. 호스트 Jetson 환경 정보 출력
#   2. NVIDIA Container Runtime 확인
#   3. cctv-ai-engine 이미지 빌드 (없으면 스킵)
#   4. Docker 컨테이너 내부 CUDA 접근 테스트
#   5. Docker 컨테이너 내부 TensorRT import 테스트
#   6. Docker 컨테이너 내부 PyTorch + GPU 테스트
#   7. Docker 컨테이너 내부 Ultralytics YOLO import 테스트
#   8. Docker 컨테이너 내부 OpenCV 헤드리스 테스트
#
# 사용법:
#   cd /path/to/CCTV-project
#   bash scripts/ops/test_jetson_docker.sh [--build] [--full-compose]
#
# 옵션:
#   --build         : cctv-ai-engine 이미지 재빌드 후 테스트
#   --full-compose  : 전체 docker-compose 스택 기동 + 서비스 헬스체크
# ==============================================================================

set -euo pipefail

# ── 색상 출력 헬퍼 ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
BOLD='\033[1m'

pass() { echo -e "${GREEN}✓ PASS${NC} $1"; }
fail() { echo -e "${RED}✗ FAIL${NC} $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }
warn() { echo -e "${YELLOW}⚠ WARN${NC} $1"; }
info() { echo -e "${BLUE}ℹ INFO${NC} $1"; }
section() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }

FAIL_COUNT=0
DO_BUILD=false
DO_FULL_COMPOSE=false
COMPOSE_FILE="docker-compose.jetson.yml"
ENV_FILE=".env.jetson"
AI_IMAGE="edgex-jetson-cctv-ai-engine"  # docker compose 기본 이미지명
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 인수 파싱
for arg in "$@"; do
    case $arg in
        --build)          DO_BUILD=true ;;
        --full-compose)   DO_FULL_COMPOSE=true ;;
    esac
done

cd "$PROJECT_ROOT"
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Jetson Docker 환경 검증 (JetPack 6.2 / CUDA 12.6)  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"

# ============================================================================
# [1] 호스트 Jetson 환경 정보
# ============================================================================
section "1. 호스트 Jetson 환경 정보"

if [[ -f /etc/nv_tegra_release ]]; then
    L4T_REV=$(grep -oP 'REVISION: \K[\d.]+' /etc/nv_tegra_release)
    L4T_MAJOR=$(grep -oP 'R\K\d+' /etc/nv_tegra_release | head -1)
    info "L4T Release : R${L4T_MAJOR}.${L4T_REV}"
    if [[ "$L4T_MAJOR" == "36" ]]; then
        pass "L4T R36.x (JetPack 6.x) 확인"
    else
        warn "예상 버전: R36.x (JetPack 6.2), 현재: R${L4T_MAJOR}.${L4T_REV}"
    fi
else
    fail "/etc/nv_tegra_release 없음 — Jetson 장치가 아닐 수 있습니다"
fi

# CUDA 버전
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[\d.]+')
    info "CUDA        : ${CUDA_VER}"
    if [[ "$CUDA_VER" == 12.6* ]]; then
        pass "CUDA 12.6 확인"
    else
        warn "예상 CUDA 12.6, 현재: ${CUDA_VER}"
    fi
else
    warn "nvcc 미발견 (CUDA 경로 확인 필요)"
fi

# TensorRT 버전
TRT_VER=$(python3 -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null || echo "N/A")
info "TensorRT    : ${TRT_VER}"
if [[ "$TRT_VER" == 10.3* ]]; then
    pass "TensorRT 10.3.x 확인"
elif [[ "$TRT_VER" != "N/A" ]]; then
    warn "예상 TensorRT 10.3.x, 현재: ${TRT_VER}"
else
    warn "TensorRT Python 바인딩 미설치 (호스트) — 컨테이너에서 검증 예정"
fi

# PyTorch (호스트)
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
info "PyTorch     : ${TORCH_VER} (cuda_available=${CUDA_AVAIL})"

# ============================================================================
# [2] NVIDIA Container Runtime 확인
# ============================================================================
section "2. NVIDIA Container Runtime 확인"

if command -v docker &>/dev/null; then
    pass "Docker 설치됨: $(docker --version | grep -oP '\d+\.\d+\.\d+')"
else
    fail "Docker 미설치"
    exit 1
fi

if docker info 2>/dev/null | grep -q "nvidia"; then
    pass "NVIDIA runtime 등록됨"
elif [[ -f /etc/docker/daemon.json ]] && grep -q "nvidia" /etc/docker/daemon.json; then
    pass "NVIDIA runtime daemon.json 확인됨"
else
    warn "NVIDIA runtime 미확인 — /etc/docker/daemon.json 검토 필요"
fi

# nvidia-container-runtime 실행 가능 여부
if command -v nvidia-container-runtime &>/dev/null; then
    pass "nvidia-container-runtime 사용 가능"
else
    fail "nvidia-container-runtime 미설치 (sudo apt install nvidia-container-toolkit)"
fi

# ============================================================================
# [3] 이미지 빌드 (--build 플래그 시)
# ============================================================================
section "3. cctv-ai-engine 이미지 빌드"

if [[ "$DO_BUILD" == "true" ]]; then
    info "docker compose 빌드 시작 (소요 시간: 5~15분)..."
    if docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build cctv-ai-engine 2>&1 | tee /tmp/jetson_build.log; then
        pass "cctv-ai-engine 이미지 빌드 성공"
    else
        fail "cctv-ai-engine 이미지 빌드 실패 — /tmp/jetson_build.log 확인"
        echo "마지막 20줄:"
        tail -20 /tmp/jetson_build.log
        exit 1
    fi
else
    # 이미지 존재 여부 확인
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "cctv-ai-engine\|edgex-jetson"; then
        pass "cctv-ai-engine 이미지 이미 존재 (빌드 스킵)"
        docker images | grep -E "cctv-ai-engine|edgex-jetson" | head -3
    else
        warn "cctv-ai-engine 이미지 없음 — --build 플래그로 재실행하거나 직접 빌드:"
        warn "  docker compose --env-file .env.jetson -f docker-compose.jetson.yml build cctv-ai-engine"
        info "빌드 없이 나머지 테스트를 계속합니다..."
    fi
fi

# ============================================================================
# [4] 컨테이너 내부 CUDA 접근 테스트 (빠른 검증: l4t-jetpack 이미지 직접 사용)
# ============================================================================
section "4. 컨테이너 내부 CUDA 접근 테스트"

info "nvcr.io/nvidia/l4t-jetpack:r36.4.0 기반 GPU 접근 테스트..."

CUDA_TEST=$(docker run --rm --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
    bash -c "nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not found'" 2>/dev/null || echo "DOCKER_RUN_FAILED")

if echo "$CUDA_TEST" | grep -q "release 12"; then
    pass "컨테이너 내부 CUDA 12.x 접근 성공: $(echo "$CUDA_TEST" | grep -oP 'release \K[\d.]+')"
elif echo "$CUDA_TEST" | grep -q "DOCKER_RUN_FAILED"; then
    fail "docker run 실패 — NVIDIA runtime 또는 이미지 문제"
else
    warn "CUDA 버전 확인 불가: $CUDA_TEST"
fi

# deviceQuery 대신 nvidia-smi 또는 간단한 GPU 존재 확인
GPU_TEST=$(docker run --rm --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
    bash -c "ls /dev/nvhost-ctrl* 2>/dev/null | head -1 || ls /dev/dri/card* 2>/dev/null | head -1 || echo 'no_gpu_dev'" 2>/dev/null || echo "FAILED")

if echo "$GPU_TEST" | grep -qvE "no_gpu_dev|FAILED"; then
    pass "GPU 디바이스 접근 확인: $GPU_TEST"
else
    warn "GPU 디바이스 직접 확인 불가 (Container Runtime이 자동 마운트하는 경우 정상)"
fi

# ============================================================================
# [5] 컨테이너 내부 TensorRT import 테스트
# ============================================================================
section "5. 컨테이너 내부 TensorRT 테스트"

TRT_TEST=$(docker run --rm --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
    python3 -c "import tensorrt as trt; print('TRT:', trt.__version__)" 2>/dev/null || echo "IMPORT_FAILED")

if echo "$TRT_TEST" | grep -q "TRT: 10"; then
    pass "컨테이너 TensorRT 10.x import 성공: $TRT_TEST"
elif echo "$TRT_TEST" | grep -q "TRT:"; then
    pass "컨테이너 TensorRT import 성공: $TRT_TEST"
else
    fail "컨테이너 TensorRT import 실패: $TRT_TEST"
fi

# ============================================================================
# [6] cctv-ai-engine 컨테이너 PyTorch + GPU 테스트
# ============================================================================
section "6. cctv-ai-engine PyTorch + CUDA 테스트"

# 빌드된 이미지가 있는지 확인
AI_IMAGE_ID=$(docker images -q "edgex-jetson-cctv-ai-engine" 2>/dev/null \
    || docker images -q "cctv-ai-engine" 2>/dev/null \
    || docker images --format "{{.ID}} {{.Repository}}" | grep -i "cctv-ai\|edgex-jetson" | awk '{print $1}' | head -1)

if [[ -n "$AI_IMAGE_ID" ]]; then
    PYTORCH_TEST=$(docker run --rm --runtime nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
        "$AI_IMAGE_ID" \
        python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    print(f'GPU matmul OK, result shape: {y.shape}')
" 2>/dev/null || echo "PYTORCH_FAILED")

    if echo "$PYTORCH_TEST" | grep -q "CUDA available: True"; then
        pass "cctv-ai-engine PyTorch GPU 연산 성공"
        echo "$PYTORCH_TEST" | while read -r line; do info "  $line"; done
    elif echo "$PYTORCH_TEST" | grep -q "PyTorch:"; then
        warn "PyTorch 로드됨 (CPU 모드): $(echo "$PYTORCH_TEST" | head -2)"
    else
        fail "cctv-ai-engine PyTorch 테스트 실패: $PYTORCH_TEST"
    fi
else
    warn "cctv-ai-engine 이미지 없음 — PyTorch 테스트 스킵 (--build 플래그 사용)"
fi

# ============================================================================
# [7] cctv-ai-engine Ultralytics YOLO import 테스트
# ============================================================================
section "7. Ultralytics YOLO + TensorRT 테스트"

if [[ -n "$AI_IMAGE_ID" ]]; then
    YOLO_TEST=$(docker run --rm --runtime nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
        "$AI_IMAGE_ID" \
        python3 -c "
from ultralytics import YOLO
import torch
import tensorrt as trt
print(f'ultralytics OK')
print(f'torch: {torch.__version__}')
print(f'tensorrt: {trt.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
" 2>/dev/null || echo "YOLO_FAILED")

    if echo "$YOLO_TEST" | grep -q "ultralytics OK"; then
        pass "YOLO + TensorRT import 성공"
        echo "$YOLO_TEST" | while read -r line; do info "  $line"; done
    else
        fail "YOLO import 실패: $YOLO_TEST"
    fi

    # 실제 YOLO 모델 추론 테스트 (모델 파일 있는 경우)
    MODEL_PATH=""
    for m in models/helmet_model_ver0.5.pt models/helmet_model.pt models/yolov8n.pt; do
        if [[ -f "$m" ]]; then MODEL_PATH="$m"; break; fi
    done

    if [[ -n "$MODEL_PATH" ]]; then
        info "YOLO 모델 추론 테스트: $MODEL_PATH"
        INFER_TEST=$(docker run --rm --runtime nvidia \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
            -v "$PROJECT_ROOT/models:/app/models:ro" \
            "$AI_IMAGE_ID" \
            python3 -c "
import torch
from ultralytics import YOLO
import numpy as np
model = YOLO('/app/$(basename $MODEL_PATH)')
# 더미 이미지로 추론 (640x640 RGB)
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
results = model.predict(dummy, device='cuda' if torch.cuda.is_available() else 'cpu',
                        verbose=False, conf=0.5)
print(f'추론 성공: {len(results)} 결과')
" 2>/dev/null || echo "INFER_FAILED")

        if echo "$INFER_TEST" | grep -q "추론 성공"; then
            pass "YOLO GPU 추론 성공: $INFER_TEST"
        else
            warn "YOLO 추론 실패 (모델 파일 경로 또는 의존성 확인): $INFER_TEST"
        fi
    else
        warn "모델 파일 없음 — 추론 테스트 스킵 (models/ 디렉토리에 .pt 파일 배치 필요)"
    fi
else
    warn "cctv-ai-engine 이미지 없음 — YOLO 테스트 스킵"
fi

# ============================================================================
# [8] OpenCV 헤드리스 테스트
# ============================================================================
section "8. OpenCV 헤드리스 + GStreamer 테스트"

if [[ -n "$AI_IMAGE_ID" ]]; then
    CV_TEST=$(docker run --rm --runtime nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        "$AI_IMAGE_ID" \
        python3 -c "
import cv2
print(f'OpenCV: {cv2.__version__}')
# GStreamer 지원 확인
build_info = cv2.getBuildInformation()
gst_ok = 'GStreamer' in build_info and 'YES' in build_info[build_info.find('GStreamer'):build_info.find('GStreamer')+30]
print(f'GStreamer support: {gst_ok}')
# 더미 이미지 처리
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'이미지 처리 OK: {gray.shape}')
" 2>/dev/null || echo "CV_FAILED")

    if echo "$CV_TEST" | grep -q "이미지 처리 OK"; then
        pass "OpenCV 헤드리스 동작 확인"
        echo "$CV_TEST" | while read -r line; do info "  $line"; done
    else
        fail "OpenCV 테스트 실패: $CV_TEST"
    fi
else
    warn "cctv-ai-engine 이미지 없음 — OpenCV 테스트 스킵"
fi

# ============================================================================
# [전체 Compose 스택 테스트] — --full-compose 플래그 시
# ============================================================================
if [[ "$DO_FULL_COMPOSE" == "true" ]]; then
    section "9. 전체 docker-compose 스택 기동 테스트"
    warn "전체 스택 기동 — 약 2~3분 소요될 수 있습니다"

    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d \
        edgex-mqtt-broker consul database 2>&1 | tail -5

    sleep 10
    info "핵심 서비스 헬스체크..."

    # MQTT 브로커
    if docker exec edgex-mqtt-broker mosquitto_pub -t test -m ping 2>/dev/null; then
        pass "MQTT 브로커 응답 확인"
    else
        warn "MQTT 브로커 응답 없음"
    fi

    # Redis
    if docker exec edgex-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        pass "Redis 응답 확인"
    else
        warn "Redis 응답 없음"
    fi

    # Consul
    if curl -sf "http://localhost:8500/v1/status/leader" &>/dev/null; then
        pass "Consul 응답 확인"
    else
        warn "Consul 응답 없음"
    fi

    warn "테스트 후 스택 정리: docker compose -f $COMPOSE_FILE down"
fi

# ============================================================================
# 결과 요약
# ============================================================================
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}테스트 결과 요약${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Jetson 환경: JetPack 6.2 / L4T R36.4.7 / CUDA 12.6 / TRT 10.3.0"
echo -e "  베이스 이미지: nvcr.io/nvidia/l4t-jetpack:r36.4.0"
echo ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✓ 모든 테스트 통과! Docker Compose 배포 준비 완료.${NC}"
    echo ""
    echo -e "배포 명령:"
    echo -e "  ${BLUE}docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build${NC}"
else
    echo -e "${RED}${BOLD}✗ ${FAIL_COUNT}개 테스트 실패. 위 오류 메시지를 확인하세요.${NC}"
fi
echo ""
exit $FAIL_COUNT
