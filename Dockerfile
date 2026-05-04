# CCTV Pipeline Runtime Dockerfile

# Stage 1: Builder
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim-bookworm

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

RUN mkdir -p /app/models /app/event_backup && chown -R cctv:cctv /app

# Builder에서 Python 패키지 복사
COPY --from=builder /root/.local /home/cctv/.local

# 외부 speaker-edgex 모듈 런타임 의존성 보강
RUN pip install --no-cache-dir "pydantic>=2.5.0" \
    && chown -R cctv:cctv /home/cctv/.local

# 환경 변수 설정
ENV PATH=/home/cctv/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 애플리케이션 코드 복사
COPY --chown=cctv:cctv src /app/src
COPY --chown=cctv:cctv main.py /app/main.py
COPY --chown=cctv:cctv runners /app/runners
COPY --chown=cctv:cctv kuiper /app/kuiper
COPY --chown=cctv:cctv models/model_manifest.json /app/models/model_manifest.json
COPY --chown=cctv:cctv requirements.txt /app/

# 런타임 쓰기 디렉터리 권한 보정
RUN chown -R cctv:cctv /app

USER cctv

# 기본 진입점 (서비스별로 docker-compose에서 command override)
CMD ["python", "main.py", "--help"]
