import json
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.datasets.train_yolo_pose_fall_rf import FEATURE_NAMES
from src.core._fall_aux_policy import aux_result_confirms_fall
from src.core._fall_shadow_review import (
    FallShadowReviewConfig,
    FallShadowReviewRecorder,
)
from src.core._synthetic_object_ids import SyntheticObjectIdAssigner
from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier
from src.core.events import DetectionEvent, EventType


def pose(track=101, frame=1, *, score=1.0, width=80, timestamp=None):
    return DetectionEvent(
        event_type=EventType.PERSON,
        object_id=track,
        x=10,
        y=20,
        width=width,
        height=160,
        confidence=0.9,
        timestamp=timestamp if timestamp is not None else frame / 30,
        keypoints=[[float(10 + i), float(20 + i), 0.9] for i in range(17)],
        metadata={
            "frame_num": frame,
            "frame_width": 640,
            "frame_height": 480,
            "fall_score": score,
        },
    )


class RecordingClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self):
        self.inputs = []
        self.on_predict = lambda: None

    def predict_proba(self, features):
        self.inputs.append(features.copy())
        self.on_predict()
        aspect = features[0, FEATURE_NAMES.index("mean_bbox_aspect")]
        return np.array([[0.1, 0.9] if aspect > 1 else [0.9, 0.1]])


def verifier(**config):
    classifier = RecordingClassifier()
    instance = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            inline_pose_rf=True,
            cooldown_seconds=config.pop("cooldown_seconds", 0),
            **config,
        ),
        inline_pose_rf_bundle={
            "model_kind": "deepstream_pose_inline_rf",
            "feature_source": "deepstream_pose_inline",
            "model": classifier,
            "feature_names": FEATURE_NAMES,
            "inference_config": {"candidate_window_seconds": 3, "max_frames": 48},
            "training_config": {"min_pose_frames": 1, "decision_threshold": 0.7},
        },
    )
    return instance, classifier


def verify_event(instance, event, camera="camera_1"):
    return instance.verify(
        camera_name=camera, pose_scope=event.metadata.get("falldata_pose_scope")
    )


@pytest.mark.parametrize("max_frames", [2, 48])
def test_capture_training_reload_preserves_five_second_runtime_window(tmp_path, max_frames):
    import joblib

    from scripts.datasets.train_deepstream_pose_fall_rf import (
        load_capture_datasets,
        train_candidate,
    )
    from scripts.ops.evaluate_sample_deepstream_replay import (
        _label_feature_capture_records,
    )

    # 시간창과 실제 표본 상한이 수집→학습→재로드에서 유지되어야 한다.
    capture_path = tmp_path / "capture.jsonl"
    instance, _ = verifier(inline_feature_capture_path=capture_path, min_pose_frames=1)
    instance._inline_pose_rf_bundle["inference_config"]["candidate_window_seconds"] = 5.0
    instance._inline_pose_rf_bundle["inference_config"]["max_frames"] = max_frames
    expected_samples = min(max_frames, 6)
    for label in (0, 1):
        for group in range(4):
            instance.reset_pose_records("camera_1")
            for frame, timestamp in enumerate(range(7), start=1):
                event = pose(frame=frame, timestamp=timestamp, width=80 + label * 240)
                instance.add_pose_events("camera_1", [event])
            assert verify_event(instance, event)["sampled_frames"] == expected_samples

    rows = [json.loads(line) for line in capture_path.read_text().splitlines()]
    assert len(rows) == 8
    for index, row in enumerate(rows):
        assert row["window_seconds"] == 5.0
        assert row["sampling_contract"] == {
            "policy": "unique_uniform_valid_pose_v1", "max_frames": max_frames,
        }
        # 기존 라벨 도구를 통과해도 새 표본 계약이 학습 입력까지 보존되어야 한다.
        labeled, errors = _label_feature_capture_records([row], {
            "is_fall": bool(index // 4), "group_id": f"g{index}",
            "scene_id": f"s{index}", "video_path": "synthetic.mp4",
            "fall_start_frame": 2, "fall_end_frame": 7,
        })
        assert errors == []
        assert len(labeled) == 1
        rows[index] = labeled[0]
    capture_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = load_capture_datasets([capture_path])
    bundle, _ = train_candidate(dataset, random_state=42, validation_fraction=0.25)
    assert bundle["inference_config"]["max_frames"] == max_frames
    assert bundle["sampling_contract"] == rows[0]["sampling_contract"]
    model_path = tmp_path / "candidate.joblib"
    joblib.dump(bundle, model_path)

    reloaded = FallDataAuxVerifier(
        FallDataAuxConfig(
            enabled=True,
            inline_pose_rf=True,
            cooldown_seconds=0,
            min_pose_frames=1,
            compare_model_path=model_path,
        )
    )
    for frame, timestamp in enumerate(range(7), start=1):
        event = pose(frame=frame, timestamp=timestamp)
        reloaded.add_pose_events("camera_1", [event])
    result = verify_event(reloaded, event)
    assert result["status"] == "ok"
    assert result["sampled_frames"] == expected_samples


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"model_kind": "yolo_pose_summary_rf"},
        {"model_kind": "deepstream_pose_inline_rf", "feature_source": "offline_pose"},
        {"model_kind": "yolo_pose_summary_rf", "feature_source": "deepstream_pose_inline"},
    ],
)
def test_inline_rf_rejects_unknown_or_different_feature_source(metadata):
    instance, classifier = verifier()
    bundle = instance._inline_pose_rf_bundle
    bundle.pop("model_kind")
    bundle.pop("feature_source")
    bundle.update(metadata)
    event = pose(width=320)
    instance.add_pose_events("camera_1", [event])

    result = verify_event(instance, event)

    # 열 이름만 같아도 계산 출처가 다르면 보조 판정에 사용할 수 없다.
    assert result["status"] == "incompatible_feature_contract"
    assert result["confirmed"] is False
    assert classifier.inputs == []
    assert "fall_probability" not in result
    metrics = instance.inline_metrics_snapshot()["cameras"]["camera_1"]
    assert metrics["unavailable_results"] == 1
    assert metrics.get("predict_calls", 0) == 0


@pytest.mark.parametrize("fail_open", [True, False])
def test_incompatible_rf_respects_existing_alarm_fallback_policy(fail_open):
    from src.core.deepstream_processor import DeepStreamProcessor

    instance, classifier = verifier(mode="confirm", fail_open_on_unavailable=fail_open)
    instance._inline_pose_rf_bundle["model_kind"] = "yolo_pose_summary_rf"
    person = pose(width=320)
    fall = replace(person, event_type=EventType.FALL_DETECTED, metadata=dict(person.metadata))
    instance.add_pose_events("camera_1", [person, fall])

    assert instance.annotate_events([fall]) == ([fall] if fail_open else [])
    assert fall.metadata["falldata_aux"]["confirmed"] is False
    assert classifier.inputs == []
    if fail_open:
        assert fall.metadata["falldata_aux_confirm_fallback"] == "incompatible_feature_contract"
    processor = object.__new__(DeepStreamProcessor)
    processor._falldata_aux = instance
    assert processor._should_fail_open_falldata_aux_result(
        fall.metadata["falldata_aux"]
    ) is fail_open


def test_incompatible_camera_model_does_not_disable_other_camera(tmp_path, monkeypatch):
    import joblib

    good_path, bad_path = tmp_path / "good.pkl", tmp_path / "bad.pkl"
    good_path.touch()
    bad_path.touch()
    instance, classifier = verifier(
        compare_model_paths_by_camera={"camera_1": bad_path, "camera_2": good_path}
    )
    good = dict(instance._inline_pose_rf_bundle)
    bad_classifier = RecordingClassifier()
    bad = {**good, "model_kind": "yolo_pose_summary_rf", "model": bad_classifier}
    monkeypatch.setattr(joblib, "load", lambda path: good if path == good_path else bad)
    first, second = pose(width=320), pose(width=320)
    instance.add_pose_events("camera_1", [first])
    instance.add_pose_events("camera_2", [second])

    assert verify_event(instance, first)["status"] == "incompatible_feature_contract"
    assert verify_event(instance, second, camera="camera_2")["status"] == "ok"
    assert bad_classifier.inputs == []
    assert len(classifier.inputs) == 1


def test_three_real_poses_do_not_satisfy_ten_pose_minimum():
    # 최소 개수는 실제 관측으로 검사한다. 반복 표본으로 부족한 관측을 채우지 않는다.
    instance, classifier = verifier(min_pose_frames=10)
    for frame in range(1, 4):
        event = pose(frame=frame)
        instance.add_pose_events("camera_1", [event])

    result = verify_event(instance, event)

    assert result["status"] == "insufficient_pose_records"
    assert result["frames_with_pose"] == 3
    assert result["min_pose_frames"] == 10
    assert classifier.inputs == []


@pytest.mark.parametrize("minimum_probability, captured", [(0.0, True), (0.5, False)])
def test_incompatible_rf_can_capture_unfiltered_training_features(
    tmp_path, minimum_probability, captured
):
    path = tmp_path / "features.jsonl"
    instance, classifier = verifier(
        inline_feature_capture_path=path,
        inline_feature_capture_min_probability=minimum_probability,
    )
    instance._inline_pose_rf_bundle["model_kind"] = "yolo_pose_summary_rf"
    event = pose()
    instance.add_pose_events("camera_1", [event])

    result = verify_event(instance, event)

    # 미검증 모델의 확률로 수집하지 않되, 명시적인 무필터 특징 수집은 유지한다.
    assert result["status"] == "incompatible_feature_contract"
    assert path.exists() is captured
    assert classifier.inputs == []
    if captured:
        assert json.loads(path.read_text())["runtime"] == "deepstream_pose_inline"


@pytest.mark.parametrize("record_count", [3, 60])
def test_inline_sampling_keeps_only_real_pose_records(tmp_path, record_count):
    from scripts.datasets.train_yolo_pose_fall_rf import _summarize_frames

    # 부족한 관측을 반복하면 frames_seen과 상위 점수 평균이 학습 경로와 달라진다.
    capture = tmp_path / "features.jsonl"
    instance, classifier = verifier(inline_feature_capture_path=capture)
    for frame in range(1, record_count + 1):
        event = pose(frame=frame, score=float(frame), timestamp=frame / 30)
        instance.add_pose_events("camera_1", [event])
    result = verify_event(instance, event)
    records = list(instance._pose_records["camera_1"])
    count = min(48, record_count)
    indices = np.linspace(0, record_count - 1, num=count).astype(int)
    expected = _summarize_frames([records[i] for i in indices], count)

    assert result["sampled_frames"] == count
    np.testing.assert_allclose(classifier.inputs[0][0], expected["feature_vector"])
    captured = json.loads(capture.read_text())
    frames = [record["frame_index"] for record in captured["frame_records"]]
    assert len(frames) == len(set(frames)) == count
    assert frames[0] == 1 and frames[-1] == record_count


def test_alternating_high_scores_never_mix_people_in_model_inputs_or_capture(tmp_path):
    capture = tmp_path / "features.jsonl"
    instance, classifier = verifier(
        inline_feature_capture_path=capture, cooldown_seconds=60
    )
    for frame, winner in enumerate([101, 202, 101], start=1):
        first = pose(101, frame, score=4 if winner == 101 else 1)
        second = pose(202, frame, score=4 if winner == 202 else 1, width=320)
        instance.add_pose_events("camera_1", [first, second])
    assert len(instance._pose_records["camera_1"]) == 6
    results = [verify_event(instance, first), verify_event(instance, second)]
    assert [r["confirmed"] for r in results] == [False, True]
    assert [r["frames_with_pose"] for r in results] == [3, 3]
    assert [r["cache_reused"] for r in results] == [False, False]
    index = FEATURE_NAMES.index("mean_bbox_aspect")
    assert [features[0, index] for features in classifier.inputs] == [0.5, 2]
    captures = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [{r["object_id"] for r in row["frame_records"]} for row in captures] == [
        {101},
        {202},
    ]
    assert [row["pose_scope"]["object_id"] for row in captures] == [101, 202]


def test_same_frame_duplicates_are_removed_per_person_not_per_camera():
    instance, _ = verifier()
    instance.add_pose_events("camera_1", [pose(), pose(score=4), pose(202)])
    instance.add_pose_events("camera_1", [pose(202, score=5)])
    assert [
        (r["object_id"], r["fall_score"]) for r in instance._pose_records["camera_1"]
    ] == [(101, 4), (202, 5)]


@pytest.mark.parametrize("track", [None, True, -1, 2**64 - 1, "101"])
def test_unknown_track_is_not_merged_and_cannot_keep_old_scope(track):
    instance, classifier = verifier()
    event = pose(track)
    event.metadata["falldata_pose_scope"] = {"object_id": 999}
    instance.add_pose_events("camera_1", [event])
    assert "falldata_pose_scope" not in event.metadata
    assert verify_event(instance, event)["status"] == "missing_pose_scope"
    assert not instance._pose_records and not classifier.inputs


def test_camera_isolation_and_mismatched_scope_never_reuse_other_result():
    instance, classifier = verifier(cooldown_seconds=60)
    first, second = pose(width=320), pose()
    instance.add_pose_events("camera_1", [first])
    instance.add_pose_events("camera_2", [second])
    assert verify_event(instance, first)["confirmed"] is True
    assert verify_event(instance, second, "camera_2")["confirmed"] is False
    assert verify_event(instance, first, "camera_2")["status"] == "missing_pose_scope"
    assert len(classifier.inputs) == 2


def test_cooldown_and_streak_are_track_local_and_cache_does_not_advance_streak():
    instance, classifier = verifier(min_confirmed_results=2)
    first = pose(width=320)
    instance.add_pose_events("camera_1", [first])
    result = verify_event(instance, first)
    assert result["confirmation_streak"] == 1 and result["confirmed"] is False
    repeated = verify_event(instance, first)
    assert repeated["cache_reused"] is True and repeated["confirmation_streak"] == 1
    second = pose(202, width=320)
    instance.add_pose_events("camera_1", [second])
    assert verify_event(instance, second)["confirmation_streak"] == 1
    first_next = pose(frame=2, width=320)
    instance.add_pose_events("camera_1", [first_next])
    result = verify_event(instance, first_next)
    assert result["confirmation_streak"] == 2 and result["confirmed"] is True
    assert len(classifier.inputs) == 3


def test_cached_result_reports_its_original_scope():
    instance, classifier = verifier(cooldown_seconds=60)
    first = pose(width=320)
    instance.add_pose_events("camera_1", [first])
    verify_event(instance, first)
    second = pose(frame=2, width=320)
    instance.add_pose_events("camera_1", [second])
    result = verify_event(instance, second)
    assert result["cache_reused"] is True
    assert result["pose_scope"] == second.metadata["falldata_pose_scope"]
    assert result["cached_pose_scope"] == first.metadata["falldata_pose_scope"]
    assert len(classifier.inputs) == 1


def test_delayed_candidate_never_reads_future_frames_or_future_cache():
    instance, classifier = verifier(cooldown_seconds=60)
    first, later = pose(), pose(frame=2, width=320)
    instance.add_pose_events("camera_1", [first])
    instance.add_pose_events("camera_1", [later])
    verify_event(instance, later)
    result = verify_event(instance, first)
    assert result["frames_with_pose"] == 1
    assert result["cache_reused"] is False and result["confirmed"] is False
    assert classifier.inputs[-1][0, FEATURE_NAMES.index("mean_bbox_aspect")] == 0.5


@pytest.mark.parametrize("reset", ["rewind", "explicit"])
def test_new_generation_invalidates_buffer_cached_confirmation_and_old_candidates(
    reset,
):
    instance, classifier = verifier(cooldown_seconds=60, min_confirmed_results=2)
    old = pose(frame=42, width=320)
    instance.add_pose_events("camera_1", [old])
    verify_event(instance, old)
    if reset == "explicit":
        instance.reset_pose_records("camera_1")
    new = pose(frame=0, width=320)
    instance.add_pose_events("camera_1", [new])
    assert verify_event(instance, old)["status"] == "stale_pose_scope"
    result = verify_event(instance, new)
    assert result["confirmation_streak"] == 1 and result["cache_reused"] is False
    assert len(instance._pose_records["camera_1"]) == 1
    assert len(classifier.inputs) == 2


def test_reset_during_classifier_execution_discards_old_result():
    instance, classifier = verifier()
    event = pose(width=320)
    instance.add_pose_events("camera_1", [event])
    classifier.on_predict = lambda: instance.reset_pose_records("camera_1")
    result = verify_event(instance, event)
    assert result["status"] == "stale_pose_scope" and result["confirmed"] is False
    assert not instance._inline_results


def test_evicted_tracks_and_old_candidates_do_not_leave_unbounded_cached_results():
    instance, _ = verifier(buffer_frames=2, cooldown_seconds=60)
    old = pose()
    instance.add_pose_events("camera_1", [old])
    verify_event(instance, old)
    for track in range(202, 207):
        event = pose(track, frame=track)
        instance.add_pose_events("camera_1", [event])
        verify_event(instance, event)
    assert len(instance._pose_records["camera_1"]) == 2
    assert len(instance._inline_results) == 2
    assert verify_event(instance, old)["status"] == "no_pose_records"


@pytest.mark.parametrize("reset_at", [None, "queued", "predict"])
def test_queued_rf_keeps_admitted_pose_but_rejects_reset_generation(tmp_path, reset_at):
    instance, classifier = verifier(buffer_frames=4, min_pose_frames=1)
    recorder = FallShadowReviewRecorder(
        FallShadowReviewConfig(tmp_path / "review.jsonl", tmp_path / "clips"),
        falldata_aux=instance,
    )
    queue = Queue(maxsize=2)
    for frame in (1, 2):
        person = pose(frame=frame, width=320)
        instance.add_pose_events("camera_1", [person, pose(202, frame=frame)])
    fall = replace(person, event_type=EventType.FALL_DETECTED)
    original_payload = fall.to_dict()
    assert recorder.submit_shadow_window(queue, "camera_1")
    assert recorder.submit_aux_work(queue, "camera_1", [fall]) is fall

    # 앞선 TCN 작업 동안 새 관측이 들어와도 접수한 사람·과거 창을 보존한다.
    assert queue.get_nowait()[1]["type"] == "fall_shadow_window"
    for frame in range(3, 8):
        instance.add_pose_events("camera_1", [pose(frame=frame, width=80)])
    assert verify_event(instance, fall)["status"] == "no_pose_records"
    if reset_at == "queued":
        instance.reset_pose_records("camera_1")
    elif reset_at == "predict":
        classifier.on_predict = lambda: instance.reset_pose_records("camera_1")

    result, record = recorder.verify_and_write_aux_record(*queue.get_nowait())
    assert fall.to_dict() == original_payload
    assert len(classifier.inputs) == (0 if reset_at == "queued" else 1)
    if reset_at:
        assert result["status"] == "stale_pose_scope" and not result["confirmed"]
        assert not instance._inline_results
    else:
        assert result["status"] == "ok" and result["confirmed"]
        assert result["frames_with_pose"] == 2
        assert result["pose_scope"] == fall.metadata["falldata_pose_scope"]
        assert classifier.inputs[0][0, FEATURE_NAMES.index("mean_bbox_aspect")] == 2.0
    assert "_falldata_pose_records" not in json.dumps(record)


def test_queued_rf_does_not_backfill_missing_pose_from_later_frames(tmp_path):
    instance, classifier = verifier(min_pose_frames=1)
    person = pose()
    instance.add_pose_events("camera_1", [person])
    fall = replace(person, event_type=EventType.FALL_DETECTED, metadata={
        **person.metadata,
        "falldata_pose_scope": {**person.metadata["falldata_pose_scope"], "object_id": 202},
    })
    recorder = FallShadowReviewRecorder(
        FallShadowReviewConfig(tmp_path / "review.jsonl", tmp_path / "clips"),
        falldata_aux=instance,
    )
    queue = Queue(maxsize=1)
    assert recorder.submit_aux_work(queue, "camera_1", [fall]) is fall
    # 접수 때 없었던 관측은 같은 프레임에 나중에 도착해도 후보 입력이 아니다.
    instance.add_pose_events("camera_1", [pose(202, width=320)])
    result, _ = recorder.verify_and_write_aux_record(*queue.get_nowait())
    assert result["status"] == "no_pose_records"
    assert not classifier.inputs


def test_pose_binding_rejects_ambiguous_same_box():
    instance, classifier = verifier()
    fall = replace(pose(), event_type=EventType.FALL_DETECTED)
    instance.add_pose_events("camera_1", [pose(), pose(202), fall])
    assert verify_event(instance, fall)["status"] == "missing_pose_scope"
    assert not classifier.inputs


def test_minimum_pose_count_cannot_be_met_by_adding_other_peoples_frames():
    instance, classifier = verifier(min_pose_frames=6)
    for index in range(1, 4):
        first, second = pose(frame=index), pose(202, frame=index, width=320)
        instance.add_pose_events("camera_1", [first, second])
    result = verify_event(instance, first)
    assert result["status"] == "insufficient_pose_records"
    assert result["frames_with_pose"] == 3 and not classifier.inputs


def test_removed_camera_and_reused_track_cannot_publish_old_inline_result():
    from src.core.deepstream_processor import DeepStreamProcessor

    instance, _ = verifier()
    person = pose(width=320)
    instance.add_pose_events("camera_1", [person])
    result = verify_event(instance, person)
    processor = object.__new__(DeepStreamProcessor)
    processor._falldata_aux = instance
    processor._cameras = {"camera_1": {}}
    processor._camera_ai_flags = {}
    processor.track_manager = Mock()
    processor.violation_filter = Mock()
    processor._debouncer = Mock()
    processor._synthetic_id_assigner = Mock()
    processor._face_identity_cache = {}
    processor._context_event_store = Mock()
    processor._put_event_dict = Mock(return_value=True)
    payload = replace(person, event_type=EventType.FALL_DETECTED).to_dict()
    assert processor._enqueue_aux_confirmed_fall_event("camera_1", payload, result)
    processor._put_event_dict.reset_mock()
    processor.remove_camera("camera_1")
    assert not processor._enqueue_aux_confirmed_fall_event("camera_1", payload, result)
    assert not instance._pose_records
    processor._put_event_dict.assert_not_called()


def test_inline_annotation_uses_bound_pose_identity():
    instance, _ = verifier(mode="confirm")
    person = pose(width=320)
    fall = replace(
        person, event_type=EventType.FALL_DETECTED, metadata=dict(person.metadata)
    )
    instance.add_pose_events("camera_1", [person, fall])
    assert instance.annotate_events([fall]) == [fall]
    assert fall.metadata["falldata_aux"]["confirmed"] is True
    assert (
        fall.metadata["falldata_aux"]["pose_scope"]
        == fall.metadata["falldata_pose_scope"]
    )


def test_processor_assigns_ids_before_pose_buffering_and_binds_fall_to_person():
    from src.core.deepstream_processor import DeepStreamProcessor

    processor = object.__new__(DeepStreamProcessor)
    instance, _ = verifier()
    processor._falldata_aux = instance
    assigner = SyntheticObjectIdAssigner(track_iou=0.3, track_timeout=10)
    processor._assign_synthetic_object_ids = assigner.assign
    processor.track_manager = SimpleNamespace(
        update=lambda camera, events: (events, [])
    )
    processor.violation_filter = SimpleNamespace(filter=lambda camera, events: events)
    processor._submit_face_work = Mock()
    processor.zone_manager = None
    processor._enqueue_zone_events = Mock()
    processor._enqueue_event_or_defer_fall_aux = Mock()
    processor._write_fall_near_miss_review_records = Mock()
    person = pose(None, width=320)
    fall = replace(
        person, event_type=EventType.FALL_DETECTED, metadata=dict(person.metadata)
    )
    processor._apply_existing_event_pipeline("camera_1", [person, fall])
    assert person.object_id != fall.object_id
    assert fall.metadata["falldata_pose_scope"]["object_id"] == person.object_id
    result = verify_event(instance, fall)
    assert result["confirmed"] is True
    assert aux_result_confirms_fall(
        fall.to_dict(),
        result,
        compare_veto_enabled=False,
        compare_veto_min_fall_score=0,
    )
    result["pose_scope"] = {**result["pose_scope"], "object_id": 999}
    assert not aux_result_confirms_fall(
        fall.to_dict(),
        result,
        compare_veto_enabled=False,
        compare_veto_min_fall_score=0,
    )


def test_inline_metrics_separate_predict_cache_and_camera_and_export(monkeypatch):
    from prometheus_client import CollectorRegistry

    from src.services.processor_metrics import ProcessorStatsCollector

    instance, classifier = verifier(cooldown_seconds=60)
    clock = [100.0]
    monkeypatch.setattr("src.core.ai._falldata_aux.time.monotonic", lambda: clock[0])
    classifier.on_predict = lambda: clock.__setitem__(0, clock[0] + 0.25)
    for camera in ("camera_1", "camera_2"):
        event = pose()
        instance.add_pose_events(camera, [event])
        verify_event(instance, event, camera)
        verify_event(instance, event, camera)
    snapshot = instance.inline_metrics_snapshot()
    assert snapshot["enabled"] is True
    for camera in ("camera_1", "camera_2"):
        assert snapshot["cameras"][camera] == {
            "requests": 2,
            "predict_calls": 1,
            "predict_completed": 1,
            "predict_errors": 0,
            "predict_seconds": 0.25,
            "ok_results": 2,
            "cache_hits": 1,
        }
    registry = CollectorRegistry()
    registry.register(
        ProcessorStatsCollector(
            SimpleNamespace(
                get_stats=lambda: {
                    "falldata_inline_stats": instance.inline_metrics_snapshot()
                }
            )
        )
    )
    labels = {"camera_id": "camera_1"}
    assert (
        registry.get_sample_value("cctv_falldata_inline_predict_calls_total", labels)
        == 1
    )
    assert (
        registry.get_sample_value("cctv_falldata_inline_predict_seconds_total", labels)
        == 0.25
    )
    assert (
        registry.get_sample_value("cctv_falldata_inline_cache_hits_total", labels) == 1
    )
    assert (
        registry.get_sample_value("cctv_falldata_inline_error_results_total", labels)
        == 0
    )
    # Scrapes and mutating a returned snapshot must not alter counters.
    snapshot["cameras"]["camera_1"]["requests"] = -1
    assert instance.inline_metrics_snapshot()["cameras"]["camera_1"]["requests"] == 2


@pytest.mark.parametrize("failure", ["predict", "conversion", "reset"])
def test_inline_metrics_distinguish_model_failure_postprocessing_and_stale_result(
    failure,
):
    instance, classifier = verifier()
    event = pose()
    instance.add_pose_events("camera_1", [event])
    if failure == "predict":
        classifier.predict_proba = Mock(side_effect=RuntimeError("model failed"))
    elif failure == "conversion":
        classifier.predict_proba = Mock(return_value=None)
    else:
        classifier.on_predict = lambda: instance.reset_pose_records("camera_1")
    result = verify_event(instance, event)
    counts = instance.inline_metrics_snapshot()["cameras"]["camera_1"]
    assert counts["predict_calls"] == 1
    assert counts["predict_completed"] == int(failure != "predict")
    assert counts["predict_errors"] == int(failure == "predict")
    assert counts["predict_seconds"] >= 0
    if failure == "reset":
        assert result["status"] == "stale_pose_scope"
        assert counts["unavailable_results"] == 1
    else:
        assert result["status"] == "error"
        assert counts["error_results"] == 1


def test_inline_metrics_insufficient_input_and_reset_do_not_fake_model_execution():
    instance, classifier = verifier(min_pose_frames=2)
    first = pose()
    instance.add_pose_events("camera_1", [first])
    assert verify_event(instance, first)["status"] == "insufficient_pose_records"
    assert verify_event(instance, first)["cache_reused"] is True
    second = pose(frame=2)
    instance.add_pose_events("camera_1", [second])
    assert verify_event(instance, second)["status"] == "ok"
    instance.reset_pose_records("camera_1")
    assert verify_event(instance, second)["status"] == "stale_pose_scope"
    instance.verify(camera_name="unregistered", pose_scope={})
    instance.verify()
    snapshot = instance.inline_metrics_snapshot()
    assert set(snapshot["cameras"]) == {"camera_1"}
    counts = snapshot["cameras"]["camera_1"]
    assert counts["requests"] == 4
    assert counts["unavailable_results"] == 3
    assert counts["cache_hits"] == 1
    assert (
        counts["predict_calls"]
        == counts["predict_completed"]
        == len(classifier.inputs)
        == 1
    )
    assert counts["ok_results"] == 1


def test_inline_metrics_disabled_verifier_has_no_execution_samples():
    instance = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=False, inline_pose_rf=True)
    )
    assert instance.inline_metrics_snapshot() == {"enabled": False, "cameras": {}}


@pytest.mark.parametrize("reset_signals", [1, 2])
def test_native_reset_invalidates_pose_cache_even_with_increasing_frames(monkeypatch, reset_signals):
    import src.core.deepstream_processor as ds
    instance, classifier = verifier(cooldown_seconds=60, min_confirmed_results=2)
    old, other = pose(frame=42, width=320), pose(frame=42, width=320)
    instance.add_pose_events("camera_1", [old])
    instance.add_pose_events("camera_2", [other])
    verify_event(instance, old)
    verify_event(instance, other, "camera_2")
    other_generation = instance._pose_generations["camera_2"]
    proc = object.__new__(ds.DeepStreamProcessor)
    proc._pad_to_camera = {0: "camera_1", 1: "camera_2"}
    proc._falldata_aux = instance
    proc.track_manager = proc.violation_filter = proc._debouncer = Mock()
    proc._synthetic_id_assigner = proc._context_event_store = Mock()
    proc._face_identity_cache = {}
    monkeypatch.setattr(ds, "Gst", SimpleNamespace(PadProbeReturn=SimpleNamespace(OK="ok")))
    event = SimpleNamespace(type=103454, get_structure=lambda: SimpleNamespace(get_uint=lambda name: (True, 0)))
    for _ in range(reset_signals):
        proc._on_stream_event(None, SimpleNamespace(get_event=lambda: event), None)
    assert verify_event(instance, old)["status"] == "stale_pose_scope"
    assert instance._pose_generations["camera_2"] == other_generation
    assert verify_event(instance, other, "camera_2")["cache_reused"] is True
    new = pose(frame=43, width=320)
    instance.add_pose_events("camera_1", [new])
    result = verify_event(instance, new)
    assert result["confirmation_streak"] == 1
    assert result["cache_reused"] is False
    assert [r["frame_index"] for r in instance._pose_records["camera_1"]] == [43]
    assert len(classifier.inputs) == 3
