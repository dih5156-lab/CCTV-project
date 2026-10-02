import json
from types import SimpleNamespace as Namespace

import pytest

from src.core._deepstream_model_metrics import (
    FRAME_STATES,
    ModelInferenceMetrics,
    observe_model_frame,
)


def linked(*items):
    node = None
    for item in reversed(items):
        node = Namespace(data=item, next=node)
    return node


BINDINGS = Namespace(
    **{
        name: Namespace(cast=lambda value: value)
        for name in (
            "NvDsFrameMeta",
            "NvDsObjectMeta",
            "NvDsUserMeta",
            "NvDsInferTensorMeta",
            "NvDsClassifierMeta",
        )
    },
    NvDsMetaType=Namespace(NVDSINFER_TENSOR_OUTPUT_META=10),
)


def tensor(component_id):
    return Namespace(
        base_meta=Namespace(meta_type=10),
        user_meta_data=Namespace(unique_id=component_id),
    )


def obj(tensors=(), classifiers=()):
    return Namespace(
        obj_user_meta_list=linked(*(tensor(value) for value in tensors)),
        classifier_meta_list=linked(
            *(Namespace(unique_component_id=value) for value in classifiers)
        ),
    )


def frame(*objects, source_id=7, tensors=()):
    return Namespace(
        source_id=source_id,
        obj_meta_list=linked(*objects),
        frame_user_meta_list=linked(*(tensor(value) for value in tensors)),
    )


def properties(**overrides):
    return {"unique-id": 3, "process-mode": 2, "output-tensor-meta": True, **overrides}


@pytest.mark.parametrize(
    "sample,expected",
    [
        (frame(), ("no_objects", 0, 0)),
        (frame(obj()), ("unobserved", 0, 0)),
        (frame(obj(tensors=(2,))), ("unobserved", 0, 0)),
        (frame(obj(classifiers=(3, 3))), ("classifier_only", 0, 1)),
        (frame(obj(classifiers=(2,))), ("unobserved", 0, 0)),
        (frame(obj(tensors=(3, 3))), ("tensor_output", 1, 0)),
        (
            frame(obj(tensors=(3,)), obj(tensors=(3,)), obj(classifiers=(3,))),
            ("tensor_output", 2, 1),
        ),
        (frame(obj(tensors=(3,), classifiers=(3,))), ("tensor_output", 1, 0)),
        (frame(obj(), tensors=(3,)), ("unobserved", 0, 0)),
    ],
)
def test_secondary_observations_do_not_conflate_frames_objects_or_cached_labels(
    sample, expected
):
    assert observe_model_frame(sample, BINDINGS, properties()) == expected


def test_primary_uses_frame_scope_and_never_object_scope():
    props = properties(**{"process-mode": 1})
    assert observe_model_frame(frame(tensors=(3, 3)), BINDINGS, props) == (
        "tensor_output",
        0,
        0,
    )
    assert observe_model_frame(frame(obj(tensors=(3,))), BINDINGS, props) == (
        "unobserved",
        0,
        0,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"unique-id": None},
        {"unique-id": True},
        {"unique-id": -1},
        {"process-mode": None},
        {"process-mode": 3},
        {"output-tensor-meta": False},
    ],
)
def test_missing_configuration_is_unknown_not_skipped(overrides):
    assert observe_model_frame(
        frame(obj(tensors=(3,))), BINDINGS, properties(**overrides)
    ) == ("unknown", 0, 0)


def test_malformed_metadata_is_unknown_even_after_an_earlier_valid_result():
    sample = frame(obj(tensors=(3,)), obj(tensors=(None,)))
    assert observe_model_frame(sample, BINDINGS, properties()) == ("unknown", 0, 0)


def element(**overrides):
    return Namespace(get_property=properties(**overrides).__getitem__)


def recorder(*, clock=lambda: 100.0, elements=None):
    metrics = ModelInferenceMetrics(clock=clock)
    metrics.reset(
        ["camera_1", "camera_2"], elements or {"attributes": element(), "helmet": None}
    )
    return metrics


def record(metrics, *frames):
    metrics.record_batch(
        Namespace(frame_meta_list=linked(*frames)),
        pyds_module=BINDINGS,
        pad_to_camera={7: "camera_1", 12: "camera_2"},
    )


def test_classifier_only_does_not_refresh_last_tensor_time():
    now = [100.0]
    metrics = recorder(clock=lambda: now[0])
    record(metrics, frame(obj(tensors=(3,))))
    now[0] = 109.0
    record(metrics, frame(obj(classifiers=(3,))))
    counts = metrics.snapshot()["models"]["attributes"]["cameras"]["camera_1"]
    assert counts["last_frame_age_seconds"] == 0
    assert counts["last_tensor_age_seconds"] == 9


def test_gobject_enum_and_integer_properties_are_json_values():
    from enum import IntEnum

    class Mode(IntEnum):
        SECONDARY = 2

    metrics = recorder(
        elements={
            "helmet": element(
                **{
                    "unique-id": 17,
                    "process-mode": Mode.SECONDARY,
                    "batch-size": 8,
                    "interval": 4,
                }
            )
        }
    )
    record(metrics, frame(obj(tensors=(17,))))
    model = metrics.snapshot()["models"]["helmet"]
    assert type(model["properties"]["process-mode"]) is int
    assert model["properties"]["batch-size"] == 8
    assert model["cameras"]["camera_1"]["frames_tensor_output"] == 1
    json.dumps(model, allow_nan=False)


def test_frame_states_partition_observations_without_inflating_disabled_or_inactive_models():
    metrics = recorder()
    record(
        metrics,
        frame(obj(tensors=(3,))),
        frame(obj(classifiers=(3,))),
        frame(),
        frame(obj()),
        frame(obj(tensors=(None,))),
        frame(source_id=99),
    )
    snapshot = metrics.snapshot()
    counts = snapshot["models"]["attributes"]["cameras"]["camera_1"]
    assert counts["frames_seen"] == 5
    assert sum(counts[f"frames_{state}"] for state in FRAME_STATES) == 5
    assert all(counts[f"frames_{state}"] == 1 for state in FRAME_STATES)
    for role, camera in [("helmet", "camera_1"), ("attributes", "camera_2")]:
        idle = snapshot["models"][role]["cameras"][camera]
        assert idle["frames_seen"] == 0
        assert idle["last_frame_age_seconds"] is None
    json.dumps(snapshot, allow_nan=False)


def test_bad_frame_and_bad_object_do_not_discard_next_camera():
    metrics = recorder()
    record(
        metrics,
        None,
        frame(obj(tensors=(None,))),
        frame(obj(tensors=(3,)), source_id=12),
    )
    snapshot = metrics.snapshot()
    assert snapshot["metadata_errors"] == 1
    assert (
        snapshot["models"]["attributes"]["cameras"]["camera_1"]["frames_unknown"] == 1
    )
    assert (
        snapshot["models"]["attributes"]["cameras"]["camera_2"]["objects_with_tensor"]
        == 1
    )


def test_build_properties_unknown_values_and_duplicate_ids_are_not_completion_evidence():
    metrics = recorder(
        elements={"attributes": element(), "helmet": element(), "pose": Namespace()}
    )
    record(metrics, frame(obj(tensors=(3,))))
    for model in metrics.snapshot()["models"].values():
        assert model["cameras"]["camera_1"]["frames_unknown"] == 1
        assert model["cameras"]["camera_1"]["frames_tensor_output"] == 0
    assert metrics.snapshot()["models"]["pose"]["properties"]["unique-id"] is None


def test_snapshot_is_detached_and_pipeline_rebuild_resets_generation_and_freshness():
    now = [100.0]
    metrics = recorder(clock=lambda: now[0])
    record(metrics, frame(obj(tensors=(3,))))
    now[0] += 7
    snapshot = metrics.snapshot()
    attrs = snapshot["models"]["attributes"]
    assert attrs["cameras"]["camera_1"]["last_frame_age_seconds"] == 7
    attrs["properties"]["unique-id"] = 99
    attrs["cameras"]["camera_1"]["frames_seen"] = 99
    assert metrics.snapshot()["models"]["attributes"]["properties"]["unique-id"] == 3
    assert (
        metrics.snapshot()["models"]["attributes"]["cameras"]["camera_1"]["frames_seen"]
        == 1
    )
    metrics.reset(["camera_2"], {"attributes": element()})
    snapshot = metrics.snapshot()
    assert snapshot["pipeline_generation"] == 2
    assert set(snapshot["models"]["attributes"]["cameras"]) == {"camera_2"}
    assert snapshot["models"]["attributes"]["cameras"]["camera_2"]["frames_seen"] == 0


def test_prometheus_exports_models_even_without_legacy_primary_counters():
    from prometheus_client import CollectorRegistry, generate_latest

    from src.services.processor_metrics import ProcessorStatsCollector

    metrics = recorder()
    record(metrics, frame(obj(tensors=(3,))), frame(obj(classifiers=(3,))))
    registry = CollectorRegistry()
    registry.register(
        ProcessorStatsCollector(
            Namespace(
                get_stats=lambda: {
                    "model_inference_stats": metrics.snapshot(),
                    "model_metrics_errors": 2,
                }
            )
        )
    )
    labels = {"model": "attributes", "camera_id": "camera_1"}
    assert registry.get_sample_value("cctv_model_frames_seen_total", labels) == 2
    assert (
        registry.get_sample_value("cctv_model_frames_tensor_output_total", labels) == 1
    )
    assert (
        registry.get_sample_value("cctv_model_frames_classifier_only_total", labels)
        == 1
    )
    assert registry.get_sample_value("cctv_model_enabled", {"model": "helmet"}) == 0
    assert (
        registry.get_sample_value(
            "cctv_model_last_frame_age_seconds",
            {"model": "helmet", "camera_id": "camera_1"},
        )
        is None
    )
    assert (
        registry.get_sample_value(
            "cctv_model_build_property",
            {"model": "attributes", "property": "unique-id"},
        )
        == 3
    )
    assert (
        registry.get_sample_value(
            "cctv_model_build_property", {"model": "attributes", "property": "interval"}
        )
        is None
    )
    assert registry.get_sample_value("cctv_processor_model_metrics_errors_total") == 2
    text = generate_latest(registry).decode()
    assert "# TYPE cctv_model_frames_seen_total counter" in text
    assert "cctv_camera_primary" not in text


@pytest.mark.parametrize("broken", [False, True])
def test_final_probe_observes_before_event_mutation_and_metrics_failure_does_not_stop_events(
    monkeypatch, broken
):
    import src.core.deepstream_processor as module

    metrics = recorder()
    sample = frame(obj(tensors=(3,)))
    batch = Namespace(frame_meta_list=linked(sample))
    if broken:

        def fail(*args, **kwargs):
            raise RuntimeError("unreadable metadata")

        metrics.record_batch = fail
    bindings = Namespace(
        **vars(BINDINGS), gst_buffer_get_nvds_batch_meta=lambda _: batch
    )
    monkeypatch.setattr(module, "pyds", bindings)
    monkeypatch.setattr(module, "Gst", Namespace(PadProbeReturn=Namespace(OK="ok")))
    calls = []

    def process_events(**kwargs):
        calls.append(kwargs["batch_meta"])
        if not broken:
            assert (
                metrics.snapshot()["models"]["attributes"]["cameras"]["camera_1"][
                    "frames_tensor_output"
                ]
                == 1
            )
        sample.obj_meta_list = None
        return 1, False

    monkeypatch.setattr(module, "process_batch_frames", process_events)
    processor = object.__new__(module.DeepStreamProcessor)
    processor._model_inference_metrics = metrics
    processor._pad_to_camera = {7: "camera_1"}
    processor._frames_processed = 0
    processor._tensor_probe_warned = False
    processor._context_worker = Namespace(
        stamp_frame=lambda frame, camera: None, finish_frame=lambda frame, camera: None)
    assert processor._on_pad_probe(None, Namespace(get_buffer=object), None) == "ok"
    assert calls == [batch]
    assert processor._frames_processed == 1
    assert getattr(processor, "_model_metrics_errors", 0) == int(broken)


def test_pipeline_build_captures_configured_elements_before_installing_runtime_hooks(
    monkeypatch,
):
    from unittest.mock import Mock

    import src.core.deepstream_processor as module

    metrics = ModelInferenceMetrics()
    configured = properties(**{"process-mode": 1, "unique-id": 1, "interval": 0})
    elements = Namespace(
        nvinfer=Namespace(get_property=configured.__getitem__),
        pose_infer=None,
        pphuman_infer=element(),
        helmet_infer=element(**{"unique-id": 2}),
        streammux=object(),
        topology=lambda: (True, True, True),
    )
    processor = Namespace(
        _cameras={"camera_1": {}},
        _crop_pose_enabled=False,
        _source_backoff_until={},
        _pad_to_camera={},
        _primary_frame_metrics=Mock(),
        _falldata_aux=Mock(),
        _model_inference_metrics=metrics,
        _pphuman_gie_id=3,
        _inference_topology_signature=lambda: (True, True, True),
        _frame_capture_enabled=lambda: False,
    )
    for name in (
        "_normalize_uri",
        "_create_output_elements",
        "_make_element",
        "_configure_streammux",
        "_configure_infer_elements",
        "_configure_tracker",
        "_configure_output_queue",
        "_link_or_raise",
        "_on_primary_tensor_probe",
        "_on_primary_metrics_probe",
        "_link_preview_branch",
        "_resolve_pphuman_infer_config",
        "_on_source_pad_added",
        "_on_pad_probe",
        "_on_stream_event",
        "_reset_camera_tracking_state",
        "_on_bus_message",
    ):
        setattr(processor, name, Mock())
    for name in (
        "_ensure_deepstream_loaded",
        "validate_pipeline_prerequisites",
        "add_pipeline_elements",
        "link_deepstream_pipeline_path",
        "attach_camera_sources_batch",
    ):
        monkeypatch.setattr(module, name, Mock())
    monkeypatch.setattr(
        module, "Gst", Namespace(init=Mock(), Pipeline=Namespace(new=Mock()))
    )
    monkeypatch.setattr(module, "GLib", Namespace(MainLoop=Mock()))
    monkeypatch.setattr(
        module, "build_source_entries", lambda **kwargs: [(0, "camera_1", "memory")]
    )
    monkeypatch.setattr(
        module, "create_pipeline_elements_bundle", lambda **kwargs: elements
    )
    monkeypatch.setattr(
        module,
        "configure_pipeline_elements_bundle",
        lambda **kwargs: configured.update({"interval": 9}),
    )
    monkeypatch.setattr(
        module, "rebuild_pad_to_camera", lambda cameras: {0: "camera_1"}
    )

    def install_hooks(**kwargs):
        processor._reset_camera_tracking_state.assert_called_once_with("camera_1")
        snapshot = metrics.snapshot()
        assert snapshot["models"]["primary"]["properties"]["interval"] == 9
        assert snapshot["models"]["pose"]["enabled"] is False
        assert snapshot["models"]["helmet"]["properties"]["unique-id"] == 2
        assert kwargs["on_pad_probe"] is processor._on_pad_probe
        assert kwargs["on_stream_event"] is processor._on_stream_event

    monkeypatch.setattr(module, "register_pipeline_runtime_hooks", install_hooks)
    module.DeepStreamProcessor._build_pipeline(processor)
    assert metrics.snapshot()["pipeline_generation"] == 1


def test_processor_stats_carries_model_snapshot_to_exporter_without_native_bindings():
    from src.core.deepstream_processor import DeepStreamProcessor

    processor = object.__new__(DeepStreamProcessor)
    processor._cameras = {"camera_1": {}}
    for name in (
        "frames_processed",
        "frames_dropped",
        "events_detected",
        "events_filtered",
        "events_failed",
    ):
        setattr(processor, f"_{name}", 0)
    processor._output_mode = "none"
    processor._preview_enabled = False
    processor._yolo_postprocess_stats = lambda: {}
    processor._build_stats_payload = lambda **kwargs: kwargs
    processor._model_inference_metrics = recorder()
    record(processor._model_inference_metrics, frame(obj(tensors=(3,))))
    processor._falldata_aux = Namespace(
        inline_metrics_snapshot=lambda: {"enabled": True, "cameras": {}},
        temporal_metrics_snapshot=lambda: {"process_calls": 2},
    )
    stats = processor.get_stats()
    assert stats["falldata_inline_stats"] == {"enabled": True, "cameras": {}}
    assert stats["falldata_temporal_stats"] == {"process_calls": 2}
    assert (
        stats["model_inference_stats"]["models"]["attributes"]["cameras"]["camera_1"][
            "frames_tensor_output"
        ]
        == 1
    )
    assert stats["primary_camera_stats"] == {}
    assert stats["model_metrics_errors"] == 0
    json.dumps(stats, allow_nan=False)
