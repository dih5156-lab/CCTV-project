# EdgeX 중심 장치 제어 전환 체크리스트

- [x] 1단계: Command 계약 초안과 단위 테스트 추가
- [x] 1단계: 계약 모듈 함수 설명 주석을 한글 기준으로 적용
- [x] 2단계: Action Layer용 EdgeX shadow Command 발행기 추가
- [ ] 3단계: shadow 모드로 direct 호출과 결과 비교
- [x] 4단계: 스피커 Device Service 변환 경계와 Jetson MQTT 러너 추가
- [x] 5단계: 사이렌 Device Service 변환 경계와 Jetson MQTT 러너 추가
- [x] 6단계: 전광판 MQTT Device Service 변환 경계와 Jetson 러너 추가
- [x] 7단계: 세 장치 결과 토픽 공통 수집 및 SQLite 감사 저장 추가
- [x] 8단계: Public API에 EdgeX Command 결과 조회 엔드포인트 추가
- [x] 9단계: EdgeX 장치 결과를 Action Layer 명령 이력에 반영
- [x] 10단계: 스피커 EdgeX Core Command 호환 HTTP 경계 추가
- [x] 11단계: 사이렌 EdgeX Core Command 호환 HTTP 경계 추가
- [x] 12단계: 전광판 EdgeX Core Command 호환 HTTP 경계 추가
- [x] 13-a단계: 실제 장치 없이 세 장치 Core Command 계약 자동 점검 추가
- [x] 13-a단계: 현장 UAT 실행 스크립트와 물리 제어 확인 보호장치 추가
- [x] 14단계: 출력 장치 Device Profile·Metadata 등록 스크립트 추가
- [x] 14-a단계: 실행 중인 EdgeX Metadata에 세 출력 장치 등록 검증
- [x] 15단계: 다중 장치 레지스트리와 device_id 기반 shadow fan-out 연결
- [ ] 13-b단계: 장치별 현장 UAT 및 롤백 검증
- [x] 16단계: 다중 장치 Device Service 요청별 클라이언트 선택 구현
- [~] 16-a단계: 러너의 레지스트리 기반 다중 클라이언트 생성 및 현장 장애 격리 검증

  - 완료: `EDGEX_DEVICE_REGISTRY_PATH`가 설정되면 러너가 장치별 클라이언트 풀을 생성한다.
  - 완료: HTTP 경로와 MQTT 하위 토픽에서 등록된 `device_id`를 검증하고 장치별로 실행한다.
  - 완료: 물리 장치 없이 2대 레지스트리 기반 세 러너의 클라이언트 풀 생성 테스트를 자동화했다.
  - 남음: 실제 장치를 2대 이상 등록한 현장 환경에서 한 장치 장애가 다른 장치에 영향을 주지 않는지 확인한다.

> 2026-09-04 운영 현황: 코드·인프라 준비도는 약 70%지만 `.env.jetson`에서 Shadow와 장치 레지스트리가 비활성화되어 실제 운영 제어는 아직 direct 경로다. 상세 내용은 `docs/handover/EDGEX_MIGRATION_STATUS.md`를 참조한다.
