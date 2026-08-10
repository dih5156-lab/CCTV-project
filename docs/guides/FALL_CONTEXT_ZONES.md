# 낙상 장소 맥락 ROI 설계

현재 `cameras.json`의 `zones` polygon을 낙상 맥락 영역으로 확장해 사용할 수 있다. 기존 구역 감지와 호환되도록 `id`, `name`, `polygon`은 유지하고 `context_type`만 추가한다.

```json
{
  "id": "bed_1",
  "name": "침대 영역",
  "context_type": "bed",
  "polygon": [[120, 180], [560, 180], [560, 420], [120, 420]]
}
```

권장 `context_type` 값:

- `bed`: 침대·병상
- `sofa`: 소파·휴게공간
- `floor`: 일반 바닥
- `road`: 도로·통로
- `industrial`: 산업 작업구역

적용 정책은 다음과 같다.

- `bed`/`sofa`: 정적인 눕기만으로 낙상 확정하지 않음
- `floor`/`road`/`industrial`: 급격한 중심 하강 + 연속 temporal 확인 시 낙상 확정
- ROI 밖: 기존 RF+TCN 정책 적용

좌표는 카메라 해상도 기준 pixel polygon이며, 실제 좌표를 넣기 전에는 기능을 활성화하지 않는다. 먼저 shadow 로그에서 ROI별 오탐·미탐을 수집한 후 보정값을 결정한다.
