from pathlib import Path

from src.core._deepstream_element_config import configure_infer_elements


class _Element:
    def __init__(self) -> None:
        self.properties = {}

    def set_property(self, name, value) -> None:
        self.properties[name] = value


def test_sgie_batch_sizes_are_independent_from_camera_batch(monkeypatch) -> None:
    primary = _Element()
    pose = _Element()
    appearance = _Element()
    helmet = _Element()
    monkeypatch.setenv("DS_POSE_SGIE_BATCH_SIZE", "4")
    monkeypatch.setenv("DS_PPHUMAN_SGIE_BATCH_SIZE", "8")
    monkeypatch.setenv("DS_HELMET_SGIE_BATCH_SIZE", "2")
    monkeypatch.setenv("DS_PRIMARY_INTERVAL", "1")
    monkeypatch.setenv("DS_CROP_PERSON_INTERVAL", "0")

    configure_infer_elements(
        nvinfer=primary,
        pose_infer=pose,
        pphuman_infer=appearance,
        helmet_infer=helmet,
        n_cams=3,
        infer_config=Path("primary.txt"),
        pose_infer_config=Path("pose.txt"),
        pphuman_infer_config=Path("appearance.txt"),
        helmet_infer_config=Path("helmet.txt"),
        env_int=lambda name, default: int(__import__("os").environ.get(name, default)),
        set_property_optional=lambda element, name, value: element.set_property(name, value),
        primary_interval_env_name="DS_CROP_PERSON_INTERVAL",
    )

    assert primary.properties["batch-size"] == 3
    assert primary.properties["interval"] == 0
    assert pose.properties["batch-size"] == 4
    assert appearance.properties["batch-size"] == 8
    assert helmet.properties["batch-size"] == 2


def test_sgie_batch_size_defaults_match_current_static_engines() -> None:
    pose = _Element()
    appearance = _Element()
    helmet = _Element()

    configure_infer_elements(
        nvinfer=None,
        pose_infer=pose,
        pphuman_infer=appearance,
        helmet_infer=helmet,
        n_cams=3,
        infer_config=Path("primary.txt"),
        pose_infer_config=Path("pose.txt"),
        pphuman_infer_config=Path("appearance.txt"),
        helmet_infer_config=Path("helmet.txt"),
        env_int=lambda name, default: default,
        set_property_optional=lambda element, name, value: element.set_property(name, value),
    )

    assert pose.properties["batch-size"] == 1
    assert appearance.properties["batch-size"] == 1
    assert helmet.properties["batch-size"] == 1
