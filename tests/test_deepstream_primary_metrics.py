from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry

from src.core._deepstream_pipeline_builder import (
    DeepStreamPipelineElements,
    link_deepstream_pipeline_path,
)
from src.services.processor_metrics import ProcessorStatsCollector


def _metrics(camera_ids=("camera_1", "camera_2"), clock=lambda: 100.0):
    from src.core._deepstream_primary_metrics import PrimaryFrameMetrics

    metrics = PrimaryFrameMetrics(clock=clock)
    metrics.reset(camera_ids)
    return metrics


def _batch(*frames):
    node = None
    for source_id, infer_done in reversed(frames):
        frame = SimpleNamespace(source_id=source_id)
        if infer_done is not None:
            frame.bInferDone = infer_done
        if isinstance(infer_done, (int, bool)) and infer_done in (0, 1):
            frame.frame_user_meta_list = None
            if infer_done:
                frame.frame_user_meta_list = SimpleNamespace(
                    data=SimpleNamespace(
                        base_meta=SimpleNamespace(meta_type=10),
                        user_meta_data=SimpleNamespace(unique_id=1),
                    ),
                    next=None,
                )
        node = SimpleNamespace(data=frame, next=node)
    return SimpleNamespace(frame_meta_list=node)


_PYDS = SimpleNamespace(
    NvDsFrameMeta=SimpleNamespace(cast=lambda data: data),
    NvDsUserMeta=SimpleNamespace(cast=lambda data: data),
    NvDsInferTensorMeta=SimpleNamespace(cast=lambda data: data),
    NvDsMetaType=SimpleNamespace(NVDSINFER_TENSOR_OUTPUT_META=10),
)


def _tensor_batch(*frames):
    """Mirror raw-tensor frames: bInferDone=False even after actual inference."""
    node = None
    for source_id, component_ids in reversed(frames):
        meta_node = None
        for component_id in reversed(component_ids):
            user_meta = SimpleNamespace(
                base_meta=SimpleNamespace(meta_type=10),
                user_meta_data=SimpleNamespace(unique_id=component_id),
            )
            meta_node = SimpleNamespace(data=user_meta, next=meta_node)
        frame = SimpleNamespace(
            source_id=source_id, bInferDone=False, frame_user_meta_list=meta_node
        )
        node = SimpleNamespace(data=frame, next=node)
    return SimpleNamespace(frame_meta_list=node)


_TENSOR_PYDS = _PYDS


def _primary_element(*, component_id=1, tensor_enabled=True):
    properties = {"unique-id": component_id, "output-tensor-meta": tensor_enabled}
    return SimpleNamespace(get_property=properties.__getitem__)


def test_raw_tensor_completion_is_not_lost_when_inference_flag_is_false():
    metrics = _metrics()
    metrics.record_batch(
        _tensor_batch((0, [1])),
        pyds_module=_TENSOR_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=_primary_element(),
    )
    stats = metrics.snapshot()["camera_1"]
    assert stats["frames_inferred"] == 1
    assert stats["frames_skipped"] == 0


def test_tensor_count_is_per_frame_and_uses_actual_primary_component_id():
    metrics = _metrics()
    metrics.record_batch(
        _tensor_batch((0, [7, 7, 8]), (1, [8]), (1, [7])),
        pyds_module=_TENSOR_PYDS,
        pad_to_camera={0: "camera_1", 1: "camera_2"},
        primary_element=_primary_element(component_id=7),
    )
    stats = metrics.snapshot()
    assert stats["camera_1"]["frames_inferred"] == 1
    assert stats["camera_2"]["frames_inferred"] == 1
    assert stats["camera_2"]["frames_skipped"] == 1


@pytest.mark.parametrize(
    "primary",
    [None, _primary_element(tensor_enabled=False), _primary_element(component_id=None)],
)
def test_unobservable_tensor_configuration_is_unknown_even_with_true_flag(primary):
    metrics = _metrics()
    batch = _tensor_batch((0, []))
    batch.frame_meta_list.data.bInferDone = True
    metrics.record_batch(
        batch,
        pyds_module=_TENSOR_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=primary,
    )
    stats = metrics.snapshot()["camera_1"]
    assert stats["frames_seen"] == 1
    assert stats["frames_inferred"] == 0
    assert stats["frames_skipped"] == 0
    assert stats["frames_inference_unknown"] == 1


def test_malformed_tensor_metadata_is_unknown_without_discarding_next_frame():
    metrics = _metrics()
    batch = _tensor_batch((0, [1]), (0, [1]))
    batch.frame_meta_list.data.frame_user_meta_list.data.user_meta_data = None
    metrics.record_batch(
        batch,
        pyds_module=_TENSOR_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=_primary_element(),
    )
    stats = metrics.snapshot()["camera_1"]
    assert stats["frames_seen"] == 2
    assert stats["frames_inferred"] == 1
    assert stats["frames_inference_unknown"] == 1


def test_primary_property_failure_is_unknown_not_a_video_processing_failure():
    metrics = _metrics()

    def unreadable_property(name):
        raise TypeError(name)

    metrics.record_batch(
        _tensor_batch((0, [1])),
        pyds_module=_TENSOR_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=SimpleNamespace(get_property=unreadable_property),
    )
    assert metrics.snapshot()["camera_1"]["frames_inference_unknown"] == 1


def test_inference_counts_are_per_camera_not_batch_or_detection_counts():
    metrics = _metrics()
    metrics.record_batch(
        _batch((7, 1), (12, 0), (7, 1), (12, 1)),
        pyds_module=_PYDS,
        pad_to_camera={7: "camera_1", 12: "camera_2"},
        primary_element=_primary_element(),
    )
    stats = metrics.snapshot()
    assert stats["camera_1"]["frames_seen"] == 2
    assert stats["camera_1"]["frames_inferred"] == 2
    assert stats["camera_2"]["frames_seen"] == 2
    assert stats["camera_2"]["frames_inferred"] == 1
    assert stats["camera_2"]["frames_skipped"] == 1


@pytest.mark.parametrize("infer_done", [None, "1", 2, -1, 0.5])
def test_missing_tensor_list_is_unknown_regardless_of_inference_flag(infer_done):
    metrics = _metrics()
    metrics.record_batch(
        _batch((0, infer_done)),
        pyds_module=_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=_primary_element(),
    )
    stats = metrics.snapshot()["camera_1"]
    assert stats["frames_seen"] == 1
    assert stats["frames_inferred"] == 0
    assert stats["frames_skipped"] == 0
    assert stats["frames_inference_unknown"] == 1


def test_inactive_camera_is_present_with_zero_counts_and_no_false_freshness():
    now = [100.0]
    metrics = _metrics(clock=lambda: now[0])
    metrics.record_batch(
        _batch((0, True)),
        pyds_module=_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=_primary_element(),
    )
    now[0] = 107.5
    stats = metrics.snapshot()
    assert stats["camera_1"]["last_frame_age_seconds"] == 7.5
    assert stats["camera_2"]["frames_seen"] == 0
    assert stats["camera_2"]["last_frame_age_seconds"] is None


def test_unmapped_sources_do_not_create_or_pollute_camera_labels():
    metrics = _metrics()
    metrics.record_batch(
        _batch((99, 1), (0, 1)),
        pyds_module=_PYDS,
        pad_to_camera={0: "not_registered"},
        primary_element=_primary_element(),
    )
    stats = metrics.snapshot()
    assert set(stats) == {"camera_1", "camera_2"}
    assert sum(item["frames_seen"] for item in stats.values()) == 0


def test_snapshot_does_not_expose_mutable_internal_counters():
    metrics = _metrics()
    snapshot = metrics.snapshot()
    snapshot["camera_1"]["frames_inferred"] = 999
    assert metrics.snapshot()["camera_1"]["frames_inferred"] == 0


def test_rebuild_resets_counters_and_removes_old_camera_labels():
    metrics = _metrics()
    metrics.record_batch(
        _batch((0, 1)),
        pyds_module=_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=_primary_element(),
    )
    metrics.reset(["camera_2"])
    assert set(metrics.snapshot()) == {"camera_2"}
    assert metrics.snapshot()["camera_2"]["frames_seen"] == 0


def test_metrics_export_inference_counters_with_camera_labels():
    metrics = _metrics()
    metrics.record_batch(
        _batch((0, 1), (0, 0)),
        pyds_module=_PYDS,
        pad_to_camera={0: "camera_1"},
        primary_element=_primary_element(),
    )
    processor = SimpleNamespace(
        get_stats=lambda: {"primary_camera_stats": metrics.snapshot()}
    )
    registry = CollectorRegistry()
    registry.register(ProcessorStatsCollector(processor))
    labels = {"camera_id": "camera_1"}
    assert (
        registry.get_sample_value("cctv_camera_primary_frames_seen_total", labels) == 2
    )
    assert (
        registry.get_sample_value("cctv_camera_primary_frames_inferred_total", labels)
        == 1
    )
    assert (
        registry.get_sample_value("cctv_camera_primary_frames_skipped_total", labels)
        == 1
    )
    assert (
        registry.get_sample_value(
            "cctv_camera_primary_frames_seen_total", {"camera_id": "camera_2"}
        )
        == 0
    )
    assert (
        registry.get_sample_value(
            "cctv_camera_primary_last_frame_age_seconds", {"camera_id": "camera_2"}
        )
        is None
    )
    families = {metric.name: metric for metric in registry.collect()}
    assert families["cctv_camera_primary_frames_inferred"].type == "counter"


def test_metrics_export_remains_compatible_with_other_processors():
    processor = SimpleNamespace(get_stats=lambda: {"frames_processed": 10})
    families = list(ProcessorStatsCollector(processor).collect())
    assert any(
        item.name == "cctv_processor_frames_processed_total" for item in families
    )
    assert not any(item.name.startswith("cctv_camera_primary") for item in families)


class _Element:
    def __init__(self, name):
        self.name = name
        self.probes = []

    def link(self, other):
        return True

    def get_name(self):
        return self.name

    def get_static_pad(self, name):
        return self

    def add_probe(self, *args):
        self.probes.append(args)


@pytest.mark.parametrize("with_secondary", [False, True])
def test_measurement_probe_is_before_secondary_inference(with_secondary):
    primary = _Element("primary")
    secondary = _Element("secondary") if with_secondary else None
    elements = DeepStreamPipelineElements(
        streammux=_Element("mux"),
        nvinfer=primary,
        primary_tensor_queue=_Element("primary-queue"),
        metadata_probe_queue=_Element("metadata-queue"),
        tracker=None,
        pphuman_infer=secondary,
        helmet_infer=None,
        converter=_Element("converter"),
        osd=_Element("osd"),
        tee=None,
        output_queue=None,
        preview_elements=[],
        output_elements=[],
    )
    observed = []

    def callback(*args):
        observed.append(args)

    gst = SimpleNamespace(PadProbeType=SimpleNamespace(BUFFER="buffer"))
    link_deepstream_pipeline_path(
        elements,
        link_or_raise=lambda first, second, message=None: first.link(second),
        gst_module=gst,
        primary_probe_callback=lambda *args: None,
        primary_metrics_callback=callback,
        link_preview_branch=lambda **kwargs: None,
        pphuman_gie_id=3,
        pphuman_infer_config="unused.txt",
    )
    assert len(primary.probes) == 1
    probe_type, installed, user_data = primary.probes[0]
    installed(primary, "frame-info", user_data)
    assert observed == [(primary, "frame-info", None)]
    assert probe_type == "buffer"
    if secondary is not None:
        assert secondary.probes == []


def test_processor_probe_records_real_metadata_without_creating_events(monkeypatch):
    import src.core.deepstream_processor as module

    metrics = _metrics()
    batch = _tensor_batch((0, [9]), (0, []))
    bindings = SimpleNamespace(
        **vars(_TENSOR_PYDS),
        gst_buffer_get_nvds_batch_meta=lambda buffer_hash: batch,
    )
    monkeypatch.setattr(module, "pyds", bindings)
    monkeypatch.setattr(
        module, "Gst", SimpleNamespace(PadProbeReturn=SimpleNamespace(OK="ok"))
    )
    processor = object.__new__(module.DeepStreamProcessor)
    processor._primary_frame_metrics = metrics
    processor._pad_to_camera = {0: "camera_1"}
    info = SimpleNamespace(get_buffer=lambda: object())
    pad = SimpleNamespace(get_parent_element=lambda: _primary_element(component_id=9))
    assert processor._on_primary_metrics_probe(pad, info, None) == "ok"
    assert metrics.snapshot()["camera_1"]["frames_inferred"] == 1
    assert metrics.snapshot()["camera_1"]["frames_skipped"] == 1


def test_observation_failure_does_not_stop_video_and_is_counted(monkeypatch):
    import src.core.deepstream_processor as module

    def cannot_read_meta(buffer_hash):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(
        module, "pyds", SimpleNamespace(gst_buffer_get_nvds_batch_meta=cannot_read_meta)
    )
    monkeypatch.setattr(
        module, "Gst", SimpleNamespace(PadProbeReturn=SimpleNamespace(OK="ok"))
    )
    processor = object.__new__(module.DeepStreamProcessor)
    processor._primary_frame_metrics = _metrics()
    processor._pad_to_camera = {0: "camera_1"}
    info = SimpleNamespace(get_buffer=lambda: object())
    assert processor._on_primary_metrics_probe(None, info, None) == "ok"
    assert processor._primary_metrics_errors == 1
    assert processor._primary_frame_metrics.snapshot()["camera_1"]["frames_seen"] == 0
