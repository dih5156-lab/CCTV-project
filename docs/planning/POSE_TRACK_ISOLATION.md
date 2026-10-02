# P5.1 — I24 인물별 자세 시간창·보조 판정 연결

갱신: 2026-09-22 KST · 계획 v0.31 · **I24 FIXED_LOCAL**

서로 다른 추적 ID의 자세가 한 낙상 모델 입력에 섞이는 문제를 로컬에서 수정했다.
카메라별 제한된 버퍼에 인물 ID를 보존하고, 낙상 후보에 연결된 인물의 기록만 모델에 전달한다.
같은 카메라의 다른 인물에게 캐시·연속 확정 횟수를 재사용하지 않는다.
**실제 영상의 ID 추적 정확도·모델 정확도·운영 적용을 완료했다는 뜻은 아니다.**

## 원인과 학습 입력 대조

- 이전에는 프레임마다 최고 fall_score인 사람 하나를 선택해 인물 ID 없이 카메라 버퍼에 넣었다.
  두 사람의 점수가 교대하면 101→202→101의 자세가 한 시간창에 섞였다.
- 보조 모델 접수는 임시 추적 ID 부여보다 앞에 있었다. 또한 기존 임시 ID는 이벤트 유형별로
  생성되어 같은 사람의 PERSON과 FALL_DETECTED가 다른 ID를 받을 수 있다.
- 결과 캐시와 연속 확정 횟수도 카메라 단위여서 인물 경계가 없었다.
- 현재 학습 코드 `_select_tracked_pose_index`는 최초 신뢰도, 이후 신뢰도와 이전 중심점 거리를
  고려한다. 실시간 경로의 프레임별 최고 낙상 점수 선택과 다르다. 학습 코드의 이 방식도
  물리적 인물 동일성을 보장하는 추적기는 아니며, 저장 모델 생성 당시 소스 버전은 미확인이다.
- 로컬 모델 `yolo_pose_fall_cam2_continuous_200_80_640.pkl`의 metadata를 읽어
  **특징 50개·3초 창·48개 샘플·최소 자세 6프레임·임계값 0.7**을 확인했다.
  실제 프로세스에 이 artifact가 로드됐다는 증거와 구분한다.
  [모델 입력 근거](evidence/pose_track_model_contract_20260922.json).

## 변경 파일과 동작

| 파일 | 핵심 변경 |
|---|---|
| `src/core/ai/_falldata_aux.py` | 프레임+인물별 중복 제거, 지정 인물/후보 프레임까지만 조회, 인물별 캐시·확정 횟수, 세대 초기화/검증 |
| `src/core/_deepstream_event_queue.py` | 임시 ID 부여 직후·누적 필터 전 관측 callback 추가 |
| `src/core/deepstream_processor.py` | callback 연결, 카메라 제거/파이프라인 재생성 시 초기화, 오래된 결과 발행 차단 |
| `src/core/_fall_shadow_review.py` | 비동기 작업에서 후보의 자세 참조를 검증기에 전달 |
| `src/core/_fall_aux_policy.py` | 후보와 inline 결과의 자세 참조가 같은 경우만 확정 근거로 허용 |
| `tests/test_falldata_pose_tracks.py` | 신규 20개 회귀; 기존 관련 테스트 4파일도 새 연결/참조 계약에 맞춰 보완 |

`metadata.falldata_pose_scope`는 camera_id·object_id·generation·frame_index를 담는다.
같은 프레임·동일 bbox에 유일하게 연결되는 PERSON의 ID를 참조하므로 FALL의 기존 ID를
바꾸지 않는다. 후보 둘이 겹쳐 유일하게 구분할 수 없으면 다른 인물을 임의 선택하지 않는다.
누락·sentinel·잘못된 ID는 인물 버퍼에 넣지 않으며 `missing_pose_scope` 등으로 남긴다.
이 상태는 확인된 낙상이나 확인된 정상으로 해석하지 않는다.
confirm 경로에서 이 보조 근거만으로 후보를 확정하지 않으므로 실영상에서 판정 보류가
증가하는지 확인해야 한다. 기존 일반 낙상 경로와 모델/임계값은 변경하지 않았다.

검증기의 내부 `verify` 호출에는 선택형 키워드 `pose_scope`가 추가된다.
non-inline 호출은 기존 방식대로 동작하고, inline 호출은 유효한 참조가 필요하다.
보조 결과·선택형 특징 JSONL에도 pose_scope를 추가한다. DB 스키마와 최상위 이벤트 형식은
유지하며 내부 부가 metadata만 늘어난다. 새 라이브러리는 없다.

후보보다 뒤의 프레임·미래 시점의 캐시는 과거 후보 판단에 쓰지 않는다.
캐시 재사용에는 `cache_reused=true`, 원래 계산한 `cached_pose_scope`를 남기고
연속 확정 횟수를 늘리지 않는다. 같은 프레임 재요청도 새 확정 횟수로 세지 않는다.
기존 cooldown 설정은 유지하므로 같은 인물의 확률 재사용 가능 시간 자체를 단축한 것은 아니다.

프레임 역행·카메라 제거·파이프라인 재생성 시 기록/캐시/확정 횟수를 초기화하고 세대를 바꾼다.
모델 계산 도중 초기화되거나 이전 세대의 작업이 늦게 도착하면 결과를 확정에 사용하지 않는다.
세대는 프로세스 내부 값이다. 모든 RTSP 내부 재연결·실제 추적기 ID 교체를 자동 탐지하는
전역 스트림 세대 구현은 남는다. 현재 단일 보조 worker를 기준으로 검증했다.

버퍼 상한은 기존 camera별 `buffer_frames`개 기록으로 유지한다. 이제 여러 인물의 기록이
함께 상한을 사용하므로 사람이 많으면 인물별 확보 시간은 짧아질 수 있다.
최소 프레임 부족을 다른 사람의 기록으로 보충하지 않는다. 버퍼에서 사라진 인물의 캐시도 정리한다.
기존 시간 샘플링·결측/PTS 품질·TCN 장면 입력·추적 ID 충돌/교체 문제까지 해결한 것은 아니다.

## 검증과 근거

**291 PASS / 0 FAIL / 5 SKIP. 신규 I24 20개 포함.**
첫 실행 13.49초, 최종 작업 트리 재확인 34.11초로 같은 결과를 얻었다.
최종 대조에서 이 작업 외의 `src/config/config.py` 변경을 발견해 그대로 보존하고
[같은 범위의 회귀를 다시 실행](evidence/pose_track_isolation_current_tree_20260922.xml)했다.
5 SKIP은 로컬 `.venv`에서 pyds가 필요한 기존 DeepStreamProcessor 생성 시험이다.
운영 컨테이너의 pyds 설치 여부와 다르다.

```bash
rtk proxy .venv/bin/python -m pytest tests/test_falldata_pose_tracks.py tests/test_falldata_aux.py tests/test_fall_aux_policy.py tests/test_fall_shadow_review.py tests/test_train_yolo_pose_fall_rf.py tests/test_deepstream_model_metrics.py tests/test_deepstream_primary_metrics.py tests/test_deepstream_element_config.py tests/test_deepstream_pipeline_builder.py tests/test_deepstream_event_factory.py tests/test_deepstream_processor.py tests/test_video_observation.py tests/test_video_event_ingest.py -q -o addopts=--tb=short -o junit_family=xunit1 --junitxml=docs/planning/evidence/pose_track_isolation_20260922.xml
rtk proxy .venv/bin/python -m docs.planning.evidence.verify_pose_track_isolation_20260922
```

- 교대 점수 2인: 6개 기록 보존, 각 인물 입력 3개씩, bbox 비율 평균 0.5/2.0으로 분리.
- 인물/카메라별 캐시·연속 확정 분리, 동일 프레임 중복, 최소 프레임 부족, 참조 누락/모호함 검증.
- 지연 후보의 미래 프레임 배제, frame rewind·명시 초기화·계산 중 초기화·카메라 제거 검증.
- 실제 Python 후처리 연결에서 PERSON/FALL ID가 달라도 올바른 자세를 참조하고 잘못된 결과는 거절.
- 기존 near-miss 시험의 전체 함수 stub을 제거해 실제 후처리 callback 연결도 통과하도록 보완.

[수정 전 합성 근거](evidence/pose_track_gap_20260922.json)는 이력으로 보존한다.
그때의 `reproduce_pose_track_gap_20260922.py`는 수정 전 코드용 재현이며 현재 재검증에는
위의 새 검증 스크립트를 사용한다.
[수정 후 대조](evidence/pose_track_isolation_example_20260922.json) ·
[JUnit](evidence/pose_track_isolation_20260922.xml) ·
[종합 검증](evidence/pose_track_isolation_validation_20260922.json).

## 운영 반영과 다음 작업

운영 AI 재시작·모델/추론 설정 변경·실장비 명령은 실행하지 않았다.
bind mount의 디스크 변경과 실행 중 코드 반영을 구분한다. I07은 OPEN이다.
I24는 서로 다른 명시적 추적 ID의 혼합/캐시 교차 사용에 대해 **FIXED_LOCAL**로 옮긴다.
실제 다인 영상의 ID 연속성과 미탐/오탐 영향은 P5.2·허용된 운영 점검에서 확인한다.

P5.1은 IN_PROGRESS, WBS DONE은 **9/33(작업 수 기준 27.3%)**로 유지한다.
다음은 **남은 Python 모델 실행/캐시 계측과 입력 품질·단절 경계 검증**이다.
그 후 적용 파일·복구본·재시작 범위를 묶어 한 서비스씩 운영 반영을 검증한다.
