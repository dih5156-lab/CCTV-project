# 보안·개인정보 운영

## 비밀값

API key, MQTT 계정, EdgeX 인증, 장치 Digest 계정, DB 비밀번호는 `.env` 또는 secret manager에만 둔다. `.env.example`, 문서, 로그, 커밋에 실제 값이 없어야 한다.

## 영상·얼굴 데이터

얼굴 이미지, embedding, snapshot, 원본 RTSP, appearance crop은 개인정보가 될 수 있다. 접근 권한, 암호화 저장, 보관 기간, 삭제 요청 처리자를 현장 정책에 맞게 지정한다.

## 운영 점검

- [ ] Public API에 운영 API key 적용
- [ ] 내부 REST에 `INTERNAL_SERVICE_TOKEN` 적용
- [ ] MQTT 외부 노출 차단 및 계정 사용
- [ ] 장치 계정 기본 비밀번호 변경
- [ ] 로그에 password/token/원본 얼굴 정보가 없는지 확인
- [ ] 백업 파일 접근 권한과 암호화 확인
- [ ] 퇴사·담당자 변경 시 secret 회전

보안 사고가 의심되면 장치 재시작보다 먼저 접근 차단·증거 보존·관리자 보고를 수행한다.

