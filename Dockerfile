# CCTV Device Service Dockerfile
# EdgeX Foundry v3 호환

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /build

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# OpenCV 시스템 의존성 설치 (headless 버전 - GUI 없음)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 보안 설정
RUN groupadd -r cctv && useradd -r -g cctv cctv

# 작업 디렉터리
WORKDIR /app

# Builder에서 Python 패키지 복사
COPY --from=builder /root/.local /home/cctv/.local
RUN chown -R cctv:cctv /home/cctv/.local

# 환경 변수 설정
ENV PATH=/home/cctv/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 애플리케이션 코드 복사 (cameras.json 제외)
COPY --chown=cctv:cctv src /app/src
COPY --chown=cctv:cctv main.py /app/main.py
COPY --chown=cctv:cctv requirements.txt /app/
COPY --chown=cctv:cctv .dockerignore /app/

# 모델 디렉터리 생성
RUN mkdir -p /app/models /app/event_backup && chown -R cctv:cctv /app

# 사용자 변경
USER cctv

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:59999/health', timeout=5)" || exit 1

# 진입점 (display 옵션 제거 - EdgeX에서는 GUI 불필요)
CMD ["python", "main.py", "--edgex"]

# 포트 노출
EXPOSE 59999

# 메타데이터
LABEL org.edgex.service="cctv-device-service" \
      org.edgex.version="1.0.0" \
      org.edgex.description="CCTV Helmet Detection and Fall Detection Service for EdgeX Foundry"
