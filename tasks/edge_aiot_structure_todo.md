# Edge AIoT 구조화 정리 체크리스트

## 현재 확인

- [x] 코드 그래프 인덱스 갱신
- [x] `ActionBridge._on_message`부터 `_ActionExecutor.execute`까지 핵심 경로 확인
- [x] 직접 장치 호출과 EdgeX Shadow 발행이 같은 실행 경계에 있음을 확인
- [x] 현재 운영 경로와 목표 경로 문서 작성
- [x] EdgeX 장치 제어 ADR 작성
- [x] `direct / shadow / edgex` 명령 모드와 기존 Shadow 설정의 하위 호환 처리

## 다음 작업

- [x] `direct/shadow/edgex` 모드의 실제 환경변수와 Compose 연결 상태 표준화
- [ ] 대표 낙상 이벤트 한 건의 end-to-end 추적 로그 확인
- [ ] Action Layer에서 경보 정책과 장치 전송 정책 분리
- [x] 스피커 1대 기준 shadow 결과 비교 기록 구조
- [ ] 스피커 1대 기준 실제 장치 shadow 결과 비교
- [ ] 실제 장치 연결 후 스피커 UAT
- [x] 전광판 EdgeX Core Command 실제 UAT
- [ ] 사이렌·전광판 순서로 동일한 수직 전환
- [ ] 다중 장치 장애 격리 및 롤백 검증

## 완료 기준

- [ ] 운영 경로가 direct인지 EdgeX인지 설정과 문서에서 일치한다.
- [ ] 모든 장치 명령이 `event_id`, `request_id`, `device_id`로 추적된다.
- [ ] EdgeX 장애와 물리 장치 장애가 AI 이벤트 처리 전체를 중단시키지 않는다.
- [ ] 한 장치 장애가 다른 장치 명령에 전파되지 않는다.
- [ ] 실제 장치 UAT와 롤백 결과가 기록된다.
