# Windows 노트북 임시 관제 서버 준비

작성·갱신: 2026-09-21 KST · 관련: P0.2 / S01 / P2.4 · 상태: **수신기 중지 중 대기 보존·재기동 후 총 4건 저장 ACK 대조 완료**

최신 사용자 확인: 일반 Ubuntu에 프로젝트 소스가 없다. [별도 실행 패키지·설치 명령](MONITORING_REMOTE.md)을 준비했다.
후속 사용자 확인으로 broker/receiver가 모두 실행됐다. 아래 네트워크 진단 기록은 과거 근거로 보존하며,
client 전달 후 16:19 KST 인증·모형 2건 저장 ACK 확인을 완료했다.
수신기 중지 중 새 2건 보존에 이어 16:32 KST 재기동 후 총 4건 저장 ACK를 확인했다.
후속 [격리 SQLite 용량 부족 검증](MONITORING_CAPACITY.md)은 71 PASS, [오래된 센서값 내부 조회](MONITORING_FRESHNESS.md)는 131 PASS, 09-22 [영상 최초 잠금 재시도·활성 쓰기](VIDEO_INTAKE_CONTENTION.md)는 200 PASS다. 조회 변경은 노트북에 미배포이며 실제 노트북 디스크 부족 시험도 미수행이다. 다음은 혼합 적체 누적·재시작 복구/저장량 계측이다. 정식 서버 이전 상세는 관제 대상 결정 시 검토하고 현재 시험 서버를 계속 사용한다.

09-21 연결 진단 보류 후, 사용자가 Jetson curl의 **HTTP/1.1 200 OK** 결과를 제공했다.
응답 Date는 14:48:25 KST이며 [제공 결과](evidence/windows_lan_recovered_20260921.json)를 보존했다.
현재는 HTTP 연결이 확인된 상태다. I21의 간헐 원인·지속성은 미확정이고 아래 실패 진단은 과거 기록이다.
후속 실제 노트북 TLS 인증·모형 전달 근거는 [원격 ACK 실측](MONITORING_REMOTE.md)에 기록했다. 단순 nginx 응답과 실제 저장 ACK 검증을 구분한다.

## 1. 확인된 조건과 권장 구성

사용자가 Windows 11, RAM 32GB, Jetson과 같은 공유기 사용을 확인했다.
최신 사용자 확인으로 SDK Manager 배포판 대신 **일반 Ubuntu를 Docker에 연결**했다.
기존 SDK Manager 배포판은 보존하며 현재 시험 대상으로 사용하지 않는다. 일반 Ubuntu의
세부 OS/WSL 버전과 Docker/Compose 버전 출력은 아직 받지 않았다.
노트북 Wi-Fi IPv4는 `192.168.2.79`, Windows의 vEthernet(WSL)은 `172.23.176.1`이다.
현재 Jetson의 `ip route get 192.168.2.79` 결과는 `dev wlP1p1s0 src 192.168.2.242`다.
당시 확인은 경로/출발 주소까지였으며, 후속 사용자 결과로 HTTP 접속 성공도 확인했다.
후속 사용자 확인: Windows `localhost:18080` nginx 화면 정상, 노트북 서브넷
`255.255.255.0`, 기본 게이트웨이 `192.168.2.254`. 초기 실패 후 HTTP 연결이 회복됐다.
CPU·디스크 여유·Windows 세부 버전·처리량은 미검증이다.

| 위치 | 맡을 역할 | 데이터 보관 |
|---|---|---|
| Jetson | 기존 영상 분석·센서 수집·출력 장치 제어, 관제 전달 대기·재시도 | 원본·공통 기록·전달 대기 DB |
| 노트북의 WSL2 Ubuntu | 시험용 MQTT broker·수신 서비스·저장 결과 조회 | 노트북 로컬 수신 SQLite DB |
| 향후 실제 관제 | 확정한 외부 API·인증·화면 연동 | D06 선정 후 결정 |

권장 시험 흐름은 `Jetson → MQTT → 노트북 수신 DB commit → 저장 결과 조회 → Jetson ACK 영속화`다.
broker의 PUBACK는 관제 DB 저장 ACK와 구분한다. 연결 장애 시 Jetson의 전달 대기를
보존하고 복구 후 ID·hash를 대조한다. 후순위 기록도 보존하며 물리 경보 재생은 별도 정책이다.

기존 WSL2·Docker를 활용해 전용 작업 디렉터리와 시험 서비스만 준비한다. 전체 Jetson
Compose·AI 모델·CUDA 의존성을 노트북으로 복사해 실행하는 구성은 아니다.
DB는 WSL Linux 파일시스템의 전용 디렉터리에 두고 두 장비가 같은 SQLite 파일을
네트워크 공유로 열지 않는다. DB 공유 대신 메시지와 저장 결과 API로 연결한다.

## 2. 노트북에서 먼저 할 일

일반 Ubuntu와 Docker 연결은 사용자 확인으로 기록했다. 설치를 반복하지 않는다.
아래 접속 시험은 **Windows Docker Desktop에 일반 Ubuntu를 연동한 구성**을 기준으로 한다.
Docker Desktop은 `-p`로 공개한 포트를 Windows 호스트/LAN에 전달한다.
[Docker Desktop 네트워크 안내](https://docs.docker.com/desktop/features/networking/).

노트북의 **일반 Ubuntu 터미널**에서 Docker 실행 위치를 확인할 수 있다.

```bash
docker info --format '{{.OperatingSystem}}'
```

`Docker Desktop`이면 아래 절차를 적용한다. Ubuntu 내부에 별도 Docker Engine을
직접 설치한 구성이라면 Windows LAN까지의 전달 방식이 다를 수 있다. 이 경우 이 출력과
PowerShell의 `wsl --version`, `wsl --list --verbose`를 확인한 후 NAT 전달 또는 mirrored
방식을 정한다. 기존 Engine 제거·배포판 교체는 하지 않는다.

노트북의 전원·절전·디스크 여유도 시험 전에 확인한다. Compose는 관제 서비스 패키징
단계에서 버전을 확인한다. 다음 단순 HTTP 시험에는 Compose가 필요하지 않다.

## 3. 두 장비의 연결 준비

| 주소 | 의미·사용 위치 |
|---|---|
| `192.168.2.79` | 노트북 Wi-Fi 주소. Jetson에서 접속할 대상 |
| `192.168.2.242` | 현재 Jetson의 노트북 방향 출발 주소. 방화벽 허용 대상 |
| `172.23.176.1` | Windows의 WSL 가상 어댑터 주소. Jetson의 접속 대상으로 사용하지 않음 |

주소는 이번 확인 시점 기준이다. 재연결로 주소가 바뀌면 접속 URL·방화벽 범위를 함께
갱신한다. 공유기 외부 포트 포워딩은 필요 없다.

1. 노트북 **일반 Ubuntu 터미널**에서 내용이 없는 기본 nginx 시험 컨테이너를 실행한다.
   이미지가 없으면 최초 다운로드가 발생한다. 호스트 파일이나 운영 DB는 연결하지 않는다.

   ```bash
   docker run -d --rm --name cctv-link-test -p 0.0.0.0:18080:80 nginx:alpine
   ```

   `18080`은 이번 임시 연결 시험 포트이며 관제 API나 MQTT 포트가 아니다.
   이름/포트 사용 중 오류가 나면 기존 대상을 먼저 확인하고 임의로 종료하지 않는다.

2. **Windows 브라우저**에서 `http://localhost:18080`을 연다. `Welcome to nginx!`가
   보이면 컨테이너와 Windows 공개 포트까지 확인한 것이다. 실패하면 다음 LAN 시험 전에
   Ubuntu의 `docker ps --filter name=cctv-link-test`, `docker logs cctv-link-test`를 확인한다.

3. **관리자 PowerShell**에서 Jetson 출발 주소에 한정한 임시 인바운드 규칙을 추가한다.
   이 규칙은 아래 IP/포트의 허용을 추가하며 기존의 다른 허용/차단 규칙을 변경하지 않는다.

   ```powershell
   New-NetFirewallRule -Name "CCTV-POC-Link-Test" -DisplayName "CCTV POC link test" -Direction Inbound -Action Allow -Protocol TCP -LocalAddress 192.168.2.79 -LocalPort 18080 -RemoteAddress 192.168.2.242 -Profile Any
   ```

   같은 이름의 규칙이 이미 있으면 `Get-NetFirewallRule -Name "CCTV-POC-Link-Test"`로
   기존 설정을 확인한다. [Microsoft 방화벽 규칙 안내](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule).

4. **Jetson 터미널**에서 시험한다.

   ```bash
   rtk proxy curl --noproxy '*' --connect-timeout 3 --max-time 5 -I http://192.168.2.79:18080/
   ```

   `HTTP/1.1 200 OK`이면 Jetson→노트북→Docker HTTP 연결에 성공한 것이다. 이는 관제
   데이터 저장·ACK 검증과 별개다. localhost는 성공하고 Jetson 요청이 timeout이면 Windows
   방화벽의 실제 적용 정책·현재 두 IP·공유기 단말 격리/게스트 Wi-Fi 순으로 확인한다.
   ping 실패만으로 HTTP 접속 불가를 확정하지 않는다.

시험을 마친 뒤 **노트북 Ubuntu**의 `docker stop cctv-link-test`로 이 컨테이너만 종료한다.
`--rm`으로 종료 시 시험 컨테이너가 제거된다. **관리자 PowerShell**에서
`Remove-NetFirewallRule -Name "CCTV-POC-Link-Test"`로 이번 규칙을 정리한다.
실제 관제 서비스의 포트·인증은 후속 어댑터 구현에 맞춰 별도로 설정한다.

WSL 내부 서비스에 직접 접근하는 별도 구성의 근거는
[Microsoft WSL 네트워크 안내](https://learn.microsoft.com/en-us/windows/wsl/networking)를 따른다.
Docker Desktop 공개 포트 시험 전에 전역 WSL 네트워크 모드 변경을 요구하지 않는다.

### 3.1 초기 연결 실패 기록 — I21 간헐 원인 OPEN

[진단 근거](evidence/windows_lan_failure_20260921.json): 사용자 HTTP 오류를 Jetson에서
재현했다. `curl` exit 7, 약 3.077초 후 `호스트로 갈 루트가 없음`이다. 실제 route는
존재하며 Jetson은 `192.168.2.242/24`, 기본 공유기는 `192.168.2.254`다.

| 확인 항목 | 결과 | 해석 |
|---|---|---|
| Jetson→공유기 ping | 2회 중 2회 응답, 이웃 상태 REACHABLE | 공유기까지의 통신 확인 |
| Jetson→노트북 ping | 2회 중 0회 응답 | 단독으로 ICMP 차단/장비 부재를 구분하지 못함 |
| Jetson의 노트북 이웃 상태 | FAILED, 재조회도 동일 | ARP 이웃 확인 실패, 장비 간 구간을 우선 진단 |
| Windows localhost:18080 | 사용자 확인: nginx 화면 정상 | 노트북 내부 테스트 서버 확인 |
| 노트북 IPv4/마스크/공유기 | 사용자 재확인: 192.168.2.79 / 255.255.255.0 / 192.168.2.254 | 양쪽 주소 대역·공유기 일치, IP 변경 후보의 우선순위 하향 |
| Jetson Wi-Fi 연결 이름 | SAW_DV_AX_5G | 사용자: 같은 네트워크, 회사 공용 AP 사용 |
| Windows의 Jetson 이웃 상태 | 사용자 확인: Unreachable, 같은 네트워크 사용 | 양방향 이웃 확인 실패. Windows ping 결과 본문은 미제공 |

`FAILED`는 이웃 확인 시도가 성공하지 못했다는 뜻이다.
[iproute2 이웃 상태 정의](https://man7.org/linux/man-pages/man8/ip-neighbour.8.html).
현재 우선 후보는 공유기/AP의 단말 격리·게스트/분리 무선망·다른 링크 필터링이며
원인은 아직 확정하지 않았다. 서버 내부 화면 성공만으로 LAN 인바운드 방화벽 통과도
증명된 것은 아니다. 네트워크 장비를 찾는 단계가 복구되면 TCP/방화벽을 다시 확인한다.

노트북의 **Windows PowerShell**에서 반대 방향을 확인한다.

```powershell
ping -n 2 192.168.2.242
Get-NetNeighbor -IPAddress 192.168.2.242 | Format-Table InterfaceAlias,IPAddress,State
```

사용자는 같은 네트워크 사용과 Windows의 `Unreachable`, 회사 공용 AP 사용을 확인했다.
네트워크 관리자와 `AP Isolation`, `Client Isolation`, 무선 단말 격리·내부망 접근 정책을
대조한다. AP 격리는 같은 무선망의 단말 간 통신을
차단할 수 있다. [제조사 기능 설명](https://www.tp-link.com/us/support/faq/2089/).
이 링크는 동작 설명이며 현재 장비가 TP-Link라는 뜻은 아니다. 공유기 모델은 미확인이다.

회사 공용 AP의 전체 격리를 임의로 해제하지 않는다. 네트워크 관리자에게 두 시험 장비의
통신 정책 확인과 필요한 예외 또는 승인된 개발망을 요청할 자료를 아래에 준비했다.
승인된 유선 개발망에서 비교 시험이 가능하면 새 IPv4와 접속 대상·시험
방화벽의 LocalAddress를 함께 갱신해 비교할 수 있다. 유선 성공을 사전 보장하지 않는다.
설정 확인 전 운영 Jetson의 무선 연결이나 공유기를 재시작하지 않는다.
네트워크·방화벽·컨테이너 설정은 이번 진단에서 변경하지 않았다. 실제 HTTP 200 재시험이
통과하기 전에는 이슈를 해결 또는 관제 연결 완료로 표시하지 않는다.

후속 사용자 설명: 이전에는 접속됐고 가끔 단절되며 약 한 시간 뒤 다시 연결되기도 한다.
이에 따라 항상 적용되는 AP 격리만을 원인으로 보지 않는다. 아래 간헐 장애 진단을 추가했다.

### 3.2 네트워크 관리자에게 전달할 내용

아래는 사용자에게 제공하는 요청 초안이다. 에이전트가 담당자에게 메시지를 발송하지 않았다.

> CCTV 프로젝트의 Jetson과 노트북 간 관제 수신 시험을 준비하고 있습니다.
> Jetson: 192.168.2.242/24, 노트북: 192.168.2.79/24, 기본 게이트웨이: 192.168.2.254입니다.
> Jetson SSID는 SAW_DV_AX_5G이며 사용자는 두 장비가 같은 회사 공용 무선망이라고 확인했습니다.
> Jetson→노트북 ARP는 FAILED, Windows→Jetson 이웃 상태는 Unreachable입니다.
> Jetson→게이트웨이는 정상 응답하고 Windows의 localhost:18080 nginx 서버도 정상입니다.
> 무선 단말 격리·VLAN/ACL 등 장비 간 통신 제한 여부를 확인 부탁드립니다.
> 추가로 이 증상은 간헐적이며 약 한 시간 뒤 회복되기도 합니다. 장애 시점의 로밍·클라이언트 정책·ARP/단말 추적 기록도 확인 부탁드립니다.
> 시험 장비 간 필요한 통신을 허용하거나 승인된 개발용 SSID/LAN을 제공할 수 있을까요?
> 첫 시험은 Jetson→노트북 TCP 18080입니다. 같은 서브넷의 이웃 탐색이 가능한지도 함께 확인 부탁드립니다.

이 주소는 현재 DHCP 할당값이다. 예외 정책에 MAC 등이 필요하면 현재 활성 인터페이스의
정보를 추가 확인하고 관리자의 주소 관리 방식에 맞춘다. 관제 MQTT/API의 실제 포트·인증은
후속 구현에서 확정한다. 네트워크 준비 대기는 해당 연결 시험의 차단이며 P2.4 내부 저장/복구
시험 전체가 기술적으로 불가능하다는 뜻은 아니다. I21은 새 간헐 장애 정보로 로컬 진단을
재개해 OPEN으로 갱신했으며, 회사 AP 로그 확인은 계속 관리자 협조가 필요하다.

### 3.3 간헐 단절·약 한 시간 후 회복의 추가 진단

[추가 원자료](evidence/windows_lan_intermittent_20260921.json): 13:54 KST 읽기 확인에서
Jetson은 BSSID `7c:4f:7d:00:7a:b0`, 신호 `-47 dBm`, 주파수 `5745 MHz`, Wi-Fi 절전 ON이다.
최근 두 시간의 NetworkManager 조회 결과는 DHCP 갱신 시 같은 `192.168.2.242`를 유지했다.
13:35:47~49의 CONNECTED_SITE→CONNECTED_GLOBAL 변화는 무선 연결 해제를 의미하지 않는다.
조회한 로그에 명시적 Wi-Fi 단절은 없지만 이 일부 기록으로 로밍/드라이버 문제를 배제하지 않는다.

우선 후보는 AP/controller의 간헐 전달·단말 추적 상태 또는 AP별 정책/로밍 차이,
단말 절전/드라이버, 주소 충돌이다. 절전 ON 자체나 약 한 시간이라는 간격만으로
원인을 정하지 않는다. 현재 신호값도 과거 단절 순간의 신호 품질을 증명하지 않는다.

노트북이 정상일 때와 장애일 때 다음을 각각 기록하고, Jetson의 동시점 BSSID/이웃/HTTP
결과와 비교한다. BSSID는 같은 SSID 이름 아래에서 실제 연결된 AP를 구분하는 주소다.

```powershell
Get-Date
netsh wlan show interfaces
Get-NetNeighbor -IPAddress 192.168.2.242 | Format-Table InterfaceAlias,IPAddress,State
curl.exe --noproxy "*" --max-time 5 -I http://localhost:18080/
```

사용자는 실패 중에도 노트북 인터넷과 localhost가 정상이라고 확인했다. 노트북 전체
연결 해제나 서버 중단보다 단말 간 전달 경로를 우선 비교한다. 절전/이동과의 연관은
별도로 답변되지 않아 미확인으로 남긴다. 같은 SSID라도 다른 AP에 연결될 수 있으므로
장애/정상 시점의 양쪽 BSSID와 정책/단말 추적 기록을 대조한다.
Windows가 위치 권한 등으로 무선 정보를 제한하면 그 결과를 기록하고 정책을 임의 변경하지 않는다.
장애 시각과 WLAN/AP 로그를 맞춰 확인하는 절차는
[Microsoft 무선 진단 안내](https://learn.microsoft.com/en-us/troubleshoot/windows-client/networking/wireless-network-connectivity-issues-troubleshooting)를 따른다.
연결 정상 시점의 캡처 없이 ARP 삭제·DHCP 재할당·Wi-Fi 재시작·절전 설정 변경을 한꺼번에
실행하지 않는다. 상태를 먼저 보존한 뒤 확인할 변수를 하나씩 정한다.

## 4. 현재 코드와 남은 실측

[P2.3 로컬 서비스](MONITORING_SERVICE.md)는 Unix MQTT socket과 같은 호스트의
SQLite ACK 조회를 사용한다. `runners/run_monitoring_local.py`를 두 PC에서 실행하는
것만으로 원격 연결되지는 않는다. 후속 [별도 패키지](MONITORING_REMOTE.md)에서
MQTT/TLS 수신과 인증된 HTTPS 저장 ACK 조회, Jetson 송신 CLI를 구현·격리 검증했다.

| 순서 | 작은 작업 단위 | 확인할 결과 |
|---|---|---|
| 1 | 일반 Ubuntu·Docker 및 노트북 LAN 접속 확인 | 사용자 확인 완료: Jetson curl HTTP 200 OK. 지속 연결 안정성은 별도 |
| 2 | TCP/TLS 수신·인증 HTTPS 조회·독립 패키지 준비 | 구현/격리 검증 완료: 회귀 73 PASS·실제 TLS/재시작 각 1 PASS |
| 3 | 합성 센서/영상 각 1건을 Jetson 시험 DB에서 노트북으로 전달 | 16:19 KST 완료: CA/토큰 인증·ready·total=2/stored=2, source/remote ACK ID/hash 일치 |
| 4 | 시험 수신기 종료·네트워크 단절·재시작 | receiver 중지 중 새 2건 보존·재기동 후 총 4건 ACK 대조 완료. 네트워크 구간 단절·장시간 장애는 별도 |
| 5 | D04/D06 수치 합의 후 누적량·디스크·오래된 데이터 시험 | P2.4 완료 기준에 따른 근거 |

2단계 격리 검증과 3단계 정상 전달에 이어 4단계의 짧은 수신기 중지/재기동 복구까지 확인했다. 새 원격 수신처는
별도의 consumer/destination 설정과 시험 DB로 준비했다. 이미 전송한 고정 목적지를
덮어쓰지 않는다. 두 장비 간 검증 후 운영 적용 범위를 정한다.
실제 관제 화면·운영 자동 시작·모델 추론 이관은 이 준비 작업의 완료 범위가 아니다.

## 5. 현재 판정

- 사용자 확인: Windows 11 / RAM 32GB / 같은 공유기 / 일반 Ubuntu·Docker 연결 / Wi-Fi `192.168.2.79` / vEthernet(WSL) `172.23.176.1`.
- 실제 조회: Jetson→노트북 출발 경로 `wlP1p1s0 / 192.168.2.242`. 샌드박스 netlink 제한 후 승인된 읽기 전용 조회로 확인했다.
- 문서 준비: Docker Desktop 기준 임시 HTTP 포트·방화벽·접속 판정·시험 종료 절차.
- HTTP 상태: 사용자 제공 Jetson curl에서 200 OK 확인(응답 Date 14:48:25 KST). 실패/정상 시점 BSSID 비교와 원인은 미확정이며 I21 OPEN 유지. 추가 AP 진단·설정 변경은 하지 않았다.
- 검증: 별도 패키지 빌드·실제 TLS 모형 2건 저장 ACK·중복·인증 오류·수신기 재시작을 Jetson 내부 격리 환경에서 확인했다.
- 노트북 기동: 사용자 확인으로 broker/receiver 정상 실행. 16:07 KST Jetson에서 18883/18443 TCP 접속 성공, [근거](evidence/monitoring_laptop_startup_20260921.json).
- 실제 전달: client 4개 파일/binding·TLS CA/토큰 인증·ready·합성 2건의 source/remote 저장 ACK ID/hash를 직접 확인했다. [16:19 실측](evidence/monitoring_laptop_delivery_20260921.json).
- 미검증: 노트북 이미지 버전/로그·DB 파일 직접 조회, 부하·용량·네트워크 구간 단절/장시간 장애·관제 화면·상시 운영 자동 시작.
- 중지 중 보존: 기존 stored 2건+새 uncertain 2건, 새 레코드는 각각 4회 재시도·ACK 없음·ID/hash 보존. [근거](evidence/monitoring_laptop_outage_20260921.json).
- 재기동 후 복구: 기존 송신 CLI를 같은 DB로 재실행해 새 접수 없이 총 4건 stored와 source/remote ACK 일치를 확인했다. [16:32 복구 근거](evidence/monitoring_laptop_recovery_20260921.json).
- P2.4는 IN_PROGRESS. 수신기 중지/재기동·격리 SQLite 용량 부족·오래된 센서값 내부 조회·영상 BUSY 재시도 내부 단위는 완료, 다음은 혼합 적체 누적·재시작 복구/저장량 계측이다. I22 운영 부하·실센서 유효시간·실제 보관량 및 P0.2/D06의 관제·현장 조건은 대기 유지.
- 이번 연결 성공 제보 반영에서는 네트워크·노트북 앱·운영 설정을 변경하지 않았다. 별도 내부 코드 수정은 [경합 개선 보고서](MONITORING_CONTENTION.md)에 기록했다.

진행 상태는 [WBS](WBS.md), 사용자 확인과 후속 이력은 [진행 기록](PROGRESS.md),
전체 시험 범위는 [POC 자원·시험표](POC_SCOPE.md)에 연결한다.
