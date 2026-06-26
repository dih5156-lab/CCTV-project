# CCTV Pipeline Runtime Dockerfile
#
# Build targets:
#   default/runtime: AI engine and utility services
#   action:         Public API, Alert API, Action Layer, EdgeX adapter
#   parser:         AIoT TLV parser

# Stage 1: Builder
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements/base.txt requirements/ai.txt ./
RUN pip install --user --no-cache-dir -r ai.txt

# Stage 1b: Lightweight action/API services
FROM python:3.10-slim-bookworm AS action

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

RUN groupadd -r -g 2001 cctv && useradd -r -u 2002 -g cctv cctv

WORKDIR /app

RUN mkdir -p /app/models /app/event_backup && chown -R cctv:cctv /app

COPY --chown=cctv:cctv src         /app/src
COPY --chown=cctv:cctv runners     /app/runners
COPY --chown=cctv:cctv kuiper      /app/kuiper
COPY --chown=cctv:cctv models/model_manifest.json /app/models/model_manifest.json
COPY --chown=cctv:cctv requirements/base.txt /app/requirements-base.txt

RUN chown -R cctv:cctv /app

USER cctv

CMD ["python", "--help"]

# Stage 1c: AIoT TLV parser service
FROM python:3.10-slim-bookworm AS parser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/parser.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY parser-python/ .

RUN groupadd -r -g 2001 cctv && useradd -r -g cctv -u 2003 cctv \
    && mkdir -p /data && chown -R cctv:cctv /app /data \
    && chmod 2775 /data

USER cctv

ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV PYTHONUNBUFFERED=1

EXPOSE 4000

CMD ["python", "main.py"]

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
COPY --chown=cctv:cctv app /app/app
COPY --chown=cctv:cctv main.py /app/main.py
COPY --chown=cctv:cctv run_external_ingest.py /app/run_external_ingest.py
COPY --chown=cctv:cctv runners /app/runners
COPY --chown=cctv:cctv kuiper /app/kuiper
COPY --chown=cctv:cctv models/model_manifest.json /app/models/model_manifest.json
COPY --chown=cctv:cctv requirements/ai.txt /app/requirements-ai.txt

# 런타임 쓰기 디렉터리 권한 보정
RUN chown -R cctv:cctv /app

USER cctv

# 기본 진입점 (서비스별로 docker-compose에서 command override)
CMD ["python", "main.py", "--help"]
