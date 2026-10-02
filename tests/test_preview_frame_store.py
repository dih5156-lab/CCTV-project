from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from src.core._preview_frame_store import (
    FRAME_IDENTITY_MARKER,
    PreviewFrameStore,
    process_preview_sample,
    read_single_preview_identity,
)


def test_native_preview_preserves_generation_and_unique_token():
    pyds, batch = native_identity()
    batch.frame_meta_list.data.misc_frame_info = [FRAME_IDENTITY_MARKER, 3, 91, 0]
    result = read_single_preview_identity(type("Buffer", (), {"pts": 999})(),
        pyds_module=pyds, pad_to_camera={1: "cam2"})
    assert result == dict(camera_id="cam2", source_id=1, frame_num=7,
        source_pts_ns=23, buffer_pts_ns=999, generation=3, frame_token=91)


def native_identity(*, source=1, batch_id=0, count=1, frame_num=7, pts=23):
    frame = NS(source_id=source, batch_id=batch_id, frame_num=frame_num, buf_pts=pts)
    batch = NS(num_frames_in_batch=count, frame_meta_list=NS(data=frame))
    pyds = NS(gst_buffer_get_nvds_batch_meta=lambda ptr: batch,
              NvDsFrameMeta=NS(cast=lambda data: data))
    return pyds, batch


@pytest.mark.parametrize("source,batch_id,count,frame_num", [
    (1, 0, 1, 7), (0, 0, 1, 0), (9, 0, 1, 7),
    (1, 1, 1, 7), (1, 0, 2, 7), (1, 0, 1, -1),
])
def test_native_preview_identity_rejects_unknown_or_ambiguous_surface(source, batch_id, count, frame_num):
    pyds, _ = native_identity(source=source, batch_id=batch_id, count=count, frame_num=frame_num)
    result = read_single_preview_identity(type("Buffer", (), {"pts": 100})(), pyds_module=pyds,
                                         pad_to_camera={0: "cam1", 1: "cam2"})
    if source in (0, 1) and batch_id == 0 and count == 1 and frame_num >= 0:
        assert result == dict(camera_id=f"cam{source+1}", source_id=source,
            frame_num=frame_num, source_pts_ns=23, buffer_pts_ns=100)
    else:
        assert result is None


def test_missing_pts_is_unknown_and_never_replaced_with_receive_time():
    pyds, _ = native_identity(pts=2**64-1)
    result = read_single_preview_identity(type("Buffer", (), {"pts": 2**64-1})(), pyds_module=pyds,
                                         pad_to_camera={1: "cam2"})
    assert result['source_pts_ns'] is None and result['buffer_pts_ns'] is None


@pytest.mark.parametrize("missing", ["batch", "frame"])
def test_missing_native_metadata_never_assumes_first_camera(missing):
    pyds, batch = native_identity()
    if missing == "batch":
        pyds.gst_buffer_get_nvds_batch_meta = lambda ptr: None
    else:
        batch.frame_meta_list = None
    assert read_single_preview_identity(type("Buffer", (), {"pts": 1})(), pyds_module=pyds,
                                        pad_to_camera={1: "cam2"}) is None


@pytest.mark.parametrize("identity", [None, {"camera_id": "missing"}, {
    "camera_id": "cam2", "frame_num": 7, "source_pts_ns": 23, "buffer_pts_ns": 100,
}])
def test_sample_uses_native_camera_and_only_delivers_new_accepted_frame(identity):
    store = PreviewFrameStore(max_fps=0)
    structure = NS(get_value=lambda key: {"width": 2, "height": 2, "format": "BGRx"}.get(key))
    caps = NS(get_size=lambda: 1, get_structure=lambda n: structure)
    buffer = NS(map=Mock(return_value=(True, NS(data=bytes(range(16))))), unmap=Mock())
    sample = NS(get_buffer=lambda: buffer, get_caps=lambda: caps)
    sink = NS(emit=lambda name: sample)
    accepted, metric = Mock(), Mock()
    gst = NS(FlowReturn=NS(OK="ok"), MapFlags=NS(READ=1))
    assert process_preview_sample(sink=sink, preview_store=store,
        preview_camera_id="cam1", cameras={"cam1": {}, "cam2": {}}, gst_module=gst,
        read_frame_identity=lambda buf: identity, on_frame_stored=accepted,
        record_stat=metric) == "ok"
    assert store.get_frame("cam1") is None
    if identity and identity["camera_id"] == "cam2":
        assert store.get_frame("cam2").shape == (2, 2, 3)
        assert store.get_frame_info("cam2")["source_pts_ns"] == 23
        accepted.assert_called_once()
        assert accepted.call_args.args[0] == "cam2"
        assert accepted.call_args.args[2] == identity
        metric.assert_called_once_with("preview_frames_stored")
        buffer.unmap.assert_called_once()
    else:
        accepted.assert_not_called()
        buffer.map.assert_not_called()
        metric.assert_called_once_with("preview_identity_rejected")


def test_throttled_sample_never_replays_previous_frame_to_callback(monkeypatch):
    store = PreviewFrameStore(max_fps=1)
    store.put_frame("cam1", [[1]], now_monotonic=10., wall_time=20.)
    monkeypatch.setattr("src.core._preview_frame_store.time.monotonic", lambda: 10.1)
    sample, accepted = Mock(), Mock()
    process_preview_sample(sink=NS(emit=lambda name: sample), preview_store=store,
        preview_camera_id="cam1", cameras={"cam1": {}},
        gst_module=NS(FlowReturn=NS(OK="ok")), on_frame_stored=accepted)
    sample.get_buffer.assert_not_called()
    accepted.assert_not_called()


def test_clear_camera_removes_identity_and_keeps_other_camera():
    store = PreviewFrameStore(max_fps=1)
    for camera, stamp in [("cam1", 10.), ("cam2", 20.)]:
        store.put_frame(camera, [[stamp]], now_monotonic=stamp, wall_time=stamp,
                        identity={"frame_num": 7})
    store.clear_camera("cam1")
    assert store.get_frame("cam1") is None and store.get_frame_info("cam1") is None
    assert store.get_frame_info("cam2")["frame_num"] == 7
    assert store.last_frame_at == 20.
    store.clear_camera("cam2")
    assert store.last_frame_at is None

