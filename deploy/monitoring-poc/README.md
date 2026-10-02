# CCTV 노트북 관제 수신 POC

프로젝트 소스 전체 없이 실행하는 최소 수신 서버다. 실제 관제 화면은 포함하지 않는다.
Jetson 전용 시험 DB의 합성 센서·영상 기록을 저장하고, HTTPS로 저장 ACK를 조회한다.
기존 nginx 18080은 별도이며 이 패키지는 MQTT/TLS 18883, HTTPS 18443을 사용한다.

## 노트북의 일반 Ubuntu에서

Docker Compose v2, Python 3, OpenSSL이 필요하다. 최초 실행에는 이미지·패키지 다운로드가 필요하다.
압축 파일을 Ubuntu 홈 디렉터리로 옮긴 뒤 실행한다. Linux 파일시스템에 풀고 `/mnt/c`에서 DB를 실행하지 않는다.

```bash
tar -xzf cctv-monitoring-poc.tar.gz
cd cctv-monitoring-poc
python3 init_poc.py --host 192.168.2.79 --lan
docker compose up -d --build
docker compose ps
docker compose logs --tail=40 receiver broker
```

`init_poc.py`는 한 번만 실행한다. 이후 재시작은 `docker compose up -d`다.
서버 인증서, MQTT 계정, API token은 노트북에서 생성한다. 유효기간은 365일이며 POC용이다.
`--lan`은 두 포트를 LAN에 공개한다. Windows 방화벽에서는 Jetson `192.168.2.242`만 허용한다.
기존 nginx 연결 방식이 포트 프록시라면 **18883·18443에도 같은 WSL 포트 전달 설정**이 필요하다.
WSL mirrored/Docker Desktop/portproxy 중 기존 구성을 확인한 뒤 적용하며 무조건 중복 설정하지 않는다.
Wi-Fi IP가 바뀌면 인증서와 송신 endpoint도 관련되므로 시험 도중 임의로 바꾸지 않는다.

## Jetson으로 전달

생성된 `client/` 폴더만 Jetson 프로젝트의 `data/runtime/monitoring_remote_20260921/client/`로 복사한다.
이 폴더에는 CA 공개 인증서·송신 MQTT 비밀번호·ACK 조회 token·설정만 들어 있다.
`state/`와 서버 개인키는 노트북에 남긴다. `client/`도 공개 공유나 Git에 올리지 않는다.

Jetson 프로젝트 폴더에서 실행한다(`rtk` 설치 환경):

```bash
rtk proxy .venv/bin/python runners/run_monitoring_remote_poc.py \
  --source-db data/runtime/monitoring_remote_20260921/source.db \
  --client-dir data/runtime/monitoring_remote_20260921/client --demo --run-seconds 60
```

첫 시험의 합격은 `total: 2`, `stored: 2`, `states: {stored: 2}`와 종료 코드 0이다.
이는 센서 모형 1건+영상 모형 1건의 DB 저장 ACK 확인이다. 실센서·AI 탐지·화면·물리 출력 합격은 아니다.
같은 명령을 반복해도 같은 두 업무 기록을 사용한다. 통신 오류 때는 영속 대기 상태를 보존한다.
`uncertain`은 복구 후 재실행 가능하고, `blocked`는 인증/대상 오류 확인 후 별도 해제가 필요하다.
키·token을 보내지 말고 오류 종류와 위 집계 결과만 공유한다.

## 보존·종료

DB는 `state/data/receiver.db`, Jetson DB는 Jetson의 전용 시험 경로에 각각 남는다.
`docker compose down`은 이 POC 컨테이너만 종료하며 bind mount DB는 보존한다.
`state/`, `client/`, `.env` 삭제나 `init_poc.py` 재생성은 기존 시험 신원·ACK를 끊을 수 있으므로 하지 않는다.
계속 쌓이는 서비스로 운영하기 전 보관량·디스크 부족·백업 정책 시험이 별도로 필요하다.
