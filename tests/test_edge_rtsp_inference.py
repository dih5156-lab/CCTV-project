"""실제 추론 근거를 손상된 텐서나 운영 설정 변경으로 오인하지 않도록 검증한다."""

import configparser
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.inference import check_edge_rtsp_inference as probe
from scripts.inference.check_edge_rtsp_inference import analyze_tensor, prepare_config


def test_pose_output_reuses_nms_and_counts_seventeen_keypoints():
    tensor = np.zeros((1, 56, 200), dtype=np.float32)
    for index in (0, 1):
        tensor[0, :5, index] = [100, 100, 40, 80, 0.9]
        tensor[0, 7:56:3, index] = 0.8
    assert analyze_tensor(tensor, "pose") == {"detections": 1, "keypoints": 17, "class_counts": {"0": 1}}
    tensor[0, 4, :] = 0
    assert analyze_tensor(tensor, "pose") == {"detections": 0, "keypoints": 0, "class_counts": {}}


def test_detection_classes_survive_separate_nms():
    tensor = np.zeros((1, 6, 200), dtype=np.float32)
    tensor[0, :4, :2] = np.array([[100], [100], [40], [80]])
    tensor[0, 4, 0] = 0.9
    tensor[0, 5, 1] = 0.9
    assert analyze_tensor(tensor, "detect") == {"detections": 2, "keypoints": 0, "class_counts": {"0": 1, "1": 1}}


@pytest.mark.parametrize("tensor", [np.zeros((1, 6, 200)), np.full((1, 56, 200), np.nan), np.zeros(56)])
def test_wrong_or_nonfinite_pose_outputs_fail(tensor):
    with pytest.raises(ValueError):
        analyze_tensor(tensor, "pose")


def test_isolated_config_never_overwrites_or_builds(tmp_path):
    source = tmp_path / "source.txt"
    original = "[property]\nmodel-engine-file=old.engine\nonnx-file=old.onnx\nprocess-mode=2\noperate-on-gie-id=1\nlabelfile-path=labels.txt\n"
    source.write_text(original)
    engine = tmp_path / "private.engine"
    engine.write_bytes(b"engine")
    target = tmp_path / "private.txt"
    prepare_config(source, target, engine)
    config = configparser.ConfigParser()
    config.read(target)
    assert source.read_text() == original
    assert config["property"]["model-engine-file"] == str(engine)
    assert config["property"]["process-mode"] == "1"
    assert "onnx-file" not in config["property"]
    assert "operate-on-gie-id" not in config["property"]
    with pytest.raises(FileExistsError):
        prepare_config(source, target, engine)
    with pytest.raises(ValueError):
        prepare_config(source, source, engine)


def test_final_counts_wait_for_inflight_probe(tmp_path, monkeypatch):
    """종료 순간 frames만 증가한 probe를 끝낸 뒤 판정해야 정상 추론이 오실패하지 않는다."""
    entered, release = threading.Event(), threading.Event()
    clock, calls = [0.0], [0]
    callbacks = {}
    output = tmp_path / "result"
    real_analyze = probe.analyze_tensor

    def delayed_analyze(*args):
        calls[0] += 1
        if calls[0] == 3:
            entered.set()
            assert release.wait(3)
        return real_analyze(*args)

    class Pad:
        def __init__(self, name):
            self.name = name

        def add_probe(self, kind, callback, *unused):
            if self.name == "infer":
                callbacks["infer"] = lambda: callback(self, SimpleNamespace(get_buffer=lambda: None))

        def get_current_caps(self):
            return SimpleNamespace(to_string=lambda: "synthetic")

    class Pipeline:
        def get_by_name(self, name):
            return SimpleNamespace(set_property=lambda *args: None, get_static_pad=lambda _: Pad(name))

        def get_bus(self):
            return None

        def set_state(self, state):
            if state == "PLAYING":
                clock[0] = 1
                callbacks["infer"]()
                clock[0] = 2.5
                callbacks["infer"]()
                self.worker = threading.Thread(target=callbacks["infer"])
                self.worker.start()
                assert entered.wait(3)
            else:
                checkpoint = json.loads((output / "result.json").read_text())
                assert checkpoint["frames"] == 3 and checkpoint["tensor_frames"] == 2
                assert checkpoint["pass"] is False
                clock[0] = 2.6
                release.set()
                self.worker.join(3)
                assert not self.worker.is_alive()
            return 0

    gst = SimpleNamespace(init=lambda _: None, parse_launch=lambda _: Pipeline(),
                          State=SimpleNamespace(PLAYING="PLAYING", NULL="NULL"),
                          StateChangeReturn=SimpleNamespace(FAILURE=-1),
                          PadProbeType=SimpleNamespace(BUFFER=1), PadProbeReturn=SimpleNamespace(OK=1))
    meta = SimpleNamespace(base_meta=SimpleNamespace(meta_type=1), user_meta_data=None)
    frame = SimpleNamespace(frame_user_meta_list=SimpleNamespace(data=meta, next=None))
    cast = SimpleNamespace(cast=lambda value: value)
    pyds = SimpleNamespace(gst_buffer_get_nvds_batch_meta=lambda _: SimpleNamespace(
        frame_meta_list=SimpleNamespace(data=frame, next=None)), NvDsFrameMeta=cast,
        NvDsUserMeta=cast, NvDsInferTensorMeta=cast, NvDsMetaType=SimpleNamespace(NVDSINFER_TENSOR_OUTPUT_META=1))
    monkeypatch.setitem(sys.modules, "gi", SimpleNamespace(require_version=lambda *args: None))
    monkeypatch.setitem(sys.modules, "gi.repository", SimpleNamespace(Gst=gst))
    monkeypatch.setitem(sys.modules, "pyds", pyds)
    monkeypatch.setattr(probe.signal, "signal", lambda *args: None)
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe, "select_yolo_output", lambda *args: object())
    monkeypatch.setattr(probe, "layer_to_numpy", lambda *args: np.zeros((56, 200)))
    monkeypatch.setattr(probe, "analyze_tensor", delayed_analyze)
    camera, config, engine = (tmp_path / name for name in ("camera.json", "infer.txt", "model.engine"))
    camera.write_text(json.dumps({"source": "rtsp://127.0.0.1:18554/test"}))
    config.write_text("[property]\n")
    engine.write_bytes(b"synthetic")
    assert probe.run(SimpleNamespace(camera=camera, infer_config=config, engine=engine,
                                     output=output, seconds=1, task="pose")) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["frames"] == result["tensor_frames"] == 3
    assert result["inference_span_seconds"] == pytest.approx(1.6)
