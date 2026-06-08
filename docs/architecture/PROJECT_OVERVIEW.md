# 📄 CCTV AI 엣지 관제 시스템 기술 요약 및 운용 가이드

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 CCTV 영상과 AIoT 센서 데이터를 활용하여 산업 현장의 안전 이벤트(헬멧 미착용, 낙상, 위험 구역 침입 등)를 실시간으로 감지하고, 이를 외부 시스템 및 현장 장비(스피커, 경광등)와 연동하는 **엣지 기반 AI 관제 시스템**입니다.
Windows(개발/PoC) 및 NVIDIA Jetson Orin(실제 운영) 환경을 모두 완벽하게 지원하도록 설계되었습니다.

## 2. 핵심 설계 의도 및 아키텍처 (Design Intentions)
이 프로젝트의 가장 큰 장점은 **"장애 고립(Fault Isolation)"**과 **"데이터 유실 방지"**를 위한 철저한 분산 설계입니다.

### 1) 마이크로서비스 아키텍처 (MSA) 및 책임 분리
하나의 거대한 프로그램이 모든 것을 처리하지 않고 4개의 독립된 계층으로 나뉘어 있습니다.
* **AI Engine (`main.py`)**: 영상 수신 및 객체 감지, 순수 이벤트(JSON) 생성 후 MQTT로 발행.
* **EdgeX Adapter**: AI 이벤트를 EdgeX Foundry 표준 메타데이터로 정규화.
* **Rule Engine (eKuiper)**: 알람의 조건(Confidence, 지속 시간 등)을 필터링.
* **Action Layer (`speaker.py`, `action_bridge.py`)**: 실제 스피커(InterM) 알람 송출 및 외부 API(관제 서버) 전송.
* **설계 의도**: 특정 서비스(예: 스피커 네트워크 단절)에 장애가 발생해도 AI 영상 분석이나 메인 관제 서버로의 데이터 전송은 멈추지 않도록 유연하게 설계되었습니다.

### 2) 하이브리드 데이터 저장 전략 (EdgeX + SQLite)
* **EdgeX**는 데이터의 라우팅과 외부 연계(HTTP/MQTT Export)에 집중합니다.
* **SQLite**는 엣지 디바이스 내부의 로컬 영속 버퍼(Outbox)로 사용됩니다. 네트워크가 끊어져도 SQLite에 이벤트를 임시 저장(`pending`)하고, 복구 시 재전송하여 데이터 유실을 막는 **Store-and-Forward** 패턴을 구현했습니다.

### 3) 엣지 가속 및 안정성 (Jetson & RTSP)
* **영상 수신 안정성 (`camera_input.py`)**: 카메라의 RTSP 스트림이 끊어지면 리소스를 낭비하지 않도록 '지수 백오프(Exponential Backoff)' 방식으로 안전하게 재연결을 시도합니다.
* **하드웨어 가속**: Jetson 운영 환경을 위해 GStreamer 하드웨어 디코딩과 TensorRT(`.engine`) 기반의 추론 파이프라인(`deepstream_processor.py`)을 구성하여 성능을 극대화했습니다.

---

## 3. 주요 모듈 및 기능 요약

* **AI 파이프라인 (`events.py`, `visualizer.py`)**
  * 표준화된 `DetectionEvent` 데이터 클래스를 사용하여 BBox, 신뢰도, 객체 ID를 관리합니다.
  * YOLOv8 기반의 커스텀 헬멧 모델, YOLOv8-pose 기반의 사람/낙상 스켈레톤 추출 모델을 결합하여 분석합니다.
* **디바이스 제어 (`speaker.py`)**
  * InterM 스피커 장비와 통신하기 위해 HTTP Digest Auth 기반의 클라이언트를 구현했습니다.
  * 텍스트를 받아 TTS(음성) 파일을 생성하고, 이를 BGM으로 변환하여 송출하는 복잡한 API 연동을 완벽히 추상화했습니다.
* **운영 및 관리 API (`api/`)**
  * **Port 9000 (Public API)**: 웹 대시보드나 중앙 관제 서버가 엣지 시스템의 상태(Health), 이벤트 기록, 카메라 목록을 조회할 수 있는 REST API입니다.
  * **Port 8769 (Stream API)**: 관제 화면에 뿌려줄 MJPEG 실시간 스트리밍을 제공합니다.

---

## 4. 운용 및 배포 방법 (Operations)

이 프로젝트는 Docker Compose를 기반으로 유연한 배포를 지원합니다.

* **환경 설정**: `.env.example`을 복사하여 `.env`를 생성하고 스피커 IP, 패스워드 등을 설정합니다.
* **Windows (개발/PC 환경)**
  ```bash
  docker compose up -d --build
  ```
* **Jetson Orin (현장 엣지 장비)**
  Jetson의 GPU와 ARM 아키텍처에 맞게 오버라이드된 Compose 파일을 사용합니다.
  ```bash
  COMPOSE_FILE=docker-compose.jetson.yml ./docker-build.sh
  ```
* **시연 및 UI 확인**
  저장소 내 `web/public-demo.html` 파일을 브라우저로 열면, Localhost로 연결된 Public API, Stream API, Grafana 상태를 한눈에 모니터링하고 알람 테스트를 할 수 있습니다.

---

## 5. 지금까지의 주요 성과 (Achievements)

1. **AI 모델 고도화**: YOLOv8 커스텀 헬멧 감지 모델이 학습 데이터 7,000여 장을 바탕으로 Recall(재현율) 94%, Precision 95% 달성.
2. **견고한 코드 검증 인프라**: 700개가 넘는 자동화 테스트(Pytest)가 구현되어 있으며, 배포, 데이터 흐름, 민감 정보 유출을 점검하는 자동화 스크립트 구축 완료.
3. **EdgeX v3 연동 완성**: AI 이벤트가 단순히 콘솔에 찍히는 것을 넘어 EdgeX 표준 규격으로 변환되는 어댑터 개발 완료.
4. **Action Layer 구현**: 실제 물리 장비(InterM 스피커)와 연동하여 TTS 방송을 자동 송출하고 오래된 음원 파일을 자동 정리하는 로직 완성.

---

## 6. 추후 방향성 및 남은 과제 (Future Directions)

프로젝트 뼈대와 실무 로직은 90% 이상 완성되었으며, 상용 런칭 및 현장 도입을 위해 다음 작업들이 필요합니다.

1. **현장 튜닝 및 실제 장비 테스트**
   * 실제 산업 현장의 조명, 카메라 각도에 따른 '오탐지(False Positive)' 튜닝.
   * 현장 네트워크망에 InterM 스피커 및 경광등을 실제 물리적으로 연결하여 딜레이와 음량 테스트.
2. **보안(Security) 하드닝**
   * API 접근 시 `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN` 필수 인증 강제.
   * Grafana, MQTT 브로커 등 인프라 컨테이너의 기본 비밀번호 변경 및 노출 포트 최소화.
3. **데이터베이스 식별자 및 중복 방지 강화**
   * 이벤트 유실을 막기 위해 모든 AI 이벤트에 고유 `event_id`를 생성하고 SQLite DB에 Unique 인덱스를 반영하는 작업.
4. **Jetson 성능 최적화 프로파일링**
   * 카메라 4대 이상이 동시에 돌아갈 때의 GPU 메모리 점유율, 발열 여부를 측정하고 `config_streammux.txt` 배치 사이즈 튜닝 마무리.

### 💡 총평
현재 프로젝트 체계는 뛰어난 MSA 구조, DevOps, MLOps 관리가 돋보입니다. 설계된 가이드라인에 따라 **보안 하드닝**과 **물리 장비 통합 테스트**만 거치면 매우 훌륭한 B2B 산업용 상용 솔루션으로 손색이 없을 것입니다.