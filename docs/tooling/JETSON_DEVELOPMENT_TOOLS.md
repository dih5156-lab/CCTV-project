# Jetson 개발·검증 도구 적용표

확인: 2026-10-13 KST. 현재 aarch64 / Python3.10.12 / L4T36.5.2 / CUDA12.6 / DeepStream7.1.
설치는 시험 실행 가능 상태이며 실제 모델의 성능·장기 운전 합격과 구분한다.

| 도구 | 현재 상태와 결정 | 적용 업무·제약 |
|---|---|---|
| NVIDIA Nsight Systems | **2024.5.4.34 Jetson용 CLI 로컬 설치·보고서 생성 확인** | P5.1/P6.2. CPU/GPU 복사·대기·스레드 흐름. 짧은 격리 재생에서 수집하고 운영 부하와 병행하지 않는다. GPU 추적·GUI는 이번 미검증 |
| DeepStream Latency Measurement | **SDK7.1 내장, 별도 설치 불필요** | P5.1/P6.2. 프레임/컴포넌트 지연과 기존 앱 계측을 대조. 현재 커스텀 Python 파이프라인의 API/probe 연결과 로그 출력은 별도 확인; 환경변수만으로 연결 완료를 주장하지 않는다 |
| Compute Sanitizer | **CUDA12.6의 2024.3.0.0 설치됨** | I34 CUDA700 재현 때 memcheck 재사용. 이전 검사 이력은 TRACKER_VALIDATION 참고. 검사 시 메모리/실행시간이 늘며 결과가 없다고 모든 CUDA 오류를 배제하지 않는다 |
| pytest + pytest-benchmark | **pytest8.4.2 유지, benchmark5.2.3·py-cpuinfo9.0.0 추가 설치** | 개발용 `.venv`, `requirements/dev.txt`에 고정. 이번 우선순위 회귀·측정 JSON 생성. 고정 입력·전력/온도/다른 부하 조건을 맞춘 비교에 사용 |
| Ruff + basedpyright | **Ruff0.15.16·기존 mypy1.20.1 재사용, basedpyright 보류** | Ruff는 타입 검사기가 아니다. 기존 mypy 설정을 우선 적용해 중복 도입을 피한다. basedpyright는 향후 IDE 진단 비교가 필요할 때 별도 개발 환경에서 검토 |
| Memray | **도입 보류** | 공식 Linux aarch64 wheel 지원 확인. P6.3에서 Python/네이티브 CPU 메모리 증가가 재현되면 사용. GPU 메모리 전체를 설명하는 도구로 간주하지 않음 |
| GitHub Actions | **기존 `.github/workflows/ci.yml` 재사용** | pytest/Ruff/Compose/이미지 작업이 이미 있음. ARM64 이미지 빌드와 실제 Jetson CUDA 실행은 구분. 원격 CI 실행·새 러너 등록은 이번 미실행; 실제 Jetson 검증과 필수 체크의 배포 차단 연결은 P6.4 |
| Grafana + Prometheus | **10.4.2 / 2.51.2 실행 중, 재설치 불필요** | Grafana DB health=ok. 실제 Prometheus5대상 중4 up/1 down: EdgeX 어댑터9105 연결 거부(I64). Jetson 설정은 `monitoring/prometheus.jetson.yml`, `/prometheus` 영속 볼륨 있음. P6.3에서 누락 수집·보관량·재생성 후 보존·알림 전달 검증 |

## 설치 위치와 재사용 명령

Nsight 시스템 설치는 `sudo -n`이 암호를 요구해 수행되지 않았다. NVIDIA의 현재 Jetson APT 저장소에서
패키지를 다운로드하고 SHA256 검증 후 프로젝트 SSD에 풀었다. 시스템 패키지·드라이버·서비스는 변경하지 않았다.

- 패키지: `nsight-systems-2024.5.4_2024.5.4.34-245434855735v0_arm64.deb`
- SHA256: `404b1d921366d94f60a027298523e295c5484ee6f6e3b8a27da304ad4fd92bad`
- CLI: `data/tooling/nsight-systems-2024.5.4/root/opt/nvidia/nsight-systems/2024.5.4/bin/nsys`
- 시험: GPU 수집을 끈 독립 `/bin/sleep 0.05` 실행에서 `data/runtime/tools_priority_20261013/nsys_smoke.nsys-rep` 생성.
- 압축 패키지와 로컬 설치는 개발 산출물이며 제품 이미지에 포함하지 않는다. GUI 의존성 추가 설치는 하지 않았다.

프로젝트 루트에서 실행한다.

```bash
rtk proxy data/tooling/nsight-systems-2024.5.4/root/opt/nvidia/nsight-systems/2024.5.4/bin/nsys --version
rtk proxy /usr/local/cuda-12.6/bin/compute-sanitizer --version
rtk proxy .venv/bin/python -m pytest tests/test_priority_risk_order.py tests/test_event_priority.py -q
rtk proxy .venv/bin/python -m pytest tests/benchmarks/test_event_priority_benchmark.py --benchmark-json=/tmp/cctv-priority-benchmark.json
rtk proxy .venv/bin/ruff check src/config/event_type_map.py
rtk proxy .venv/bin/python -m mypy --follow-imports=silent src/config/event_type_map.py
```

DeepStream 격리 시험에는 `NVDS_ENABLE_LATENCY_MEASUREMENT=1`과
`NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1`을 해당 시험 프로세스에만 설정한다.
측정 전후 처리율·출력 로그·동일 프레임 ID를 대조하고 계측 비용을 함께 기록한다.

benchmark는 기능 회귀를 먼저 통과한 뒤 비교한다. 이번 단회 평균 약8.45→7.78µs,
중앙값 약5.81→6.68µs로 방향도 다르므로 **성능 향상 판정은 하지 않는다**.
GPU FPS·카메라 지연을 Python 함수 시험으로 대신하지 않는다.

## WBS 적용 순서

1. P3.1 우선순위·만료/안전 재개: pytest/Ruff/mypy. 함수 성능이 문제가 될 때 benchmark 비교.
2. P5.1/P6.2 병목: 기존 DeepStream 계측 → 짧은 Nsight 수집. CUDA 오류가 재현되면 Compute Sanitizer.
3. P6.3 장기 운전: 기존 Prometheus/Grafana 수집 복구·큐/디스크/온도 알림. 메모리 증가 원인이 불명확할 때 Memray.
4. P6.4 후보 고정: 기존 Actions 검증과 Jetson 실검증 증거 연결. 운영 Jetson에 일반 PR을 자동 실행하는 러너는 두지 않는다.

## 공식 자료

- [Nsight 설치: Tegra는 JetPack Embedded Platforms Edition](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- [DeepStream7.1 지연 계측](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_troubleshooting.html)
- [Compute Sanitizer 검사 범위·메모리 비용](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [pytest-benchmark 사용·결과 비교](https://pytest-benchmark.readthedocs.io/en/latest/usage.html)
- [Ruff는 린터이며 타입 검사기는 아님](https://docs.astral.sh/ruff/faq/), [mypy](https://mypy.readthedocs.io/en/stable/), [basedpyright 설치](https://docs.basedpyright.com/latest/installation/command-line-and-language-server/)
- [Memray 플랫폼·네이티브 추적 한계](https://bloomberg.github.io/memray/supported_environments.html)
- [GitHub 자체 호스팅 러너](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [Prometheus 보관 정책](https://prometheus.io/docs/prometheus/latest/storage/), [Grafana 영속 저장](https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/)
