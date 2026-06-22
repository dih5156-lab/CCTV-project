"""AppearancePipeline 저장 옵션 테스트."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.ai._appearance_analyzer import AppearanceAnalyzer
from src.core.ai._appearance_pipeline import AppearancePipeline
from src.core.events import DetectionEvent, EventType


def test_save_person_crop_disabled_by_default(tmp_path):
    crop_dir = tmp_path / "crops"
    pipeline = AppearancePipeline(AppearanceAnalyzer(), crop_dir)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    path = pipeline.save_person_crop(frame, 0, 0, 20, 20, "cam1", 1, 1000.0)

    assert path is None
    assert not crop_dir.exists()


def test_save_person_crop_enabled_writes_file(tmp_path):
    crop_dir = tmp_path / "crops"
    pipeline = AppearancePipeline(AppearanceAnalyzer(), crop_dir, save_crops=True)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    path = pipeline.save_person_crop(frame, 0, 0, 20, 20, "cam1", 1, 1000.0)

    assert path is not None
    assert (crop_dir / "cam1_1_1000000.jpg").exists()


def test_save_person_crop_includes_context_by_default(tmp_path):
    pytest.importorskip("cv2")

    crop_dir = tmp_path / "crops"
    pipeline = AppearancePipeline(AppearanceAnalyzer(), crop_dir, save_crops=True)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, 40:60] = (255, 255, 255)

    path = pipeline.save_person_crop(frame, 40, 40, 20, 20, "cam1", 1, 1000.0)

    saved = __import__("cv2").imread(path)
    assert saved is not None
    assert saved.shape[:2] == (44, 44)
    assert saved[12:32, 12:32].mean() > 200


def test_save_person_crop_context_can_be_disabled(tmp_path):
    pytest.importorskip("cv2")

    crop_dir = tmp_path / "crops"
    pipeline = AppearancePipeline(
        AppearanceAnalyzer(),
        crop_dir,
        save_crops=True,
        crop_context_ratio=0.0,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    path = pipeline.save_person_crop(frame, 40, 40, 20, 20, "cam1", 1, 1000.0)

    saved = __import__("cv2").imread(path)
    assert saved is not None
    assert saved.shape[:2] == (20, 20)


def test_build_log_payload_includes_face_meta_and_bbox(tmp_path):
    pipeline = AppearancePipeline(AppearanceAnalyzer(), tmp_path / "crops")
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=10,
        y=20,
        width=30,
        height=40,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )
    attrs = {
        "upper_color": "black",
        "lower_color": "blue",
        "has_helmet": True,
        "helmet_color": "yellow",
        "has_backpack": True,
        "has_handbag": False,
        "has_suitcase": False,
        "attribute_backend": "hsv",
    }
    face_meta = {
        "gender": "male",
        "age_group": "adult",
        "face_name": "홍길동",
    }

    payload = pipeline._build_log_payload(
        camera_id="cam01",
        person=person,
        attrs=attrs,
        face_meta=face_meta,
        crop_path="data/runtime/appearance_crops/cam01_7.jpg",
        timestamp=1234.5,
    )

    assert payload["camera_id"] == "cam01"
    assert payload["track_id"] == 7
    assert payload["upper_color"] == "black"
    assert payload["lower_color"] == "blue"
    assert payload["has_helmet"] is True
    assert payload["helmet_color"] == "yellow"
    assert payload["has_backpack"] is True
    assert payload["gender"] == "male"
    assert payload["age_group"] == "adult"
    assert payload["face_name"] == "홍길동"
    assert payload["attribute_backend"] == "hsv"
    assert payload["bbox_x"] == 10
    assert payload["bbox_y"] == 20
    assert payload["bbox_w"] == 30
    assert payload["bbox_h"] == 40


def test_build_log_parts_includes_registered_face_name(tmp_path):
    pipeline = AppearancePipeline(AppearanceAnalyzer(), tmp_path / "crops")
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=10,
        y=20,
        width=30,
        height=40,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )

    parts = pipeline._build_log_parts(
        person,
        {
            "upper_color": "black",
            "lower_color": "blue",
            "has_helmet": True,
            "has_backpack": False,
            "has_handbag": False,
            "has_suitcase": False,
        },
        {"face_name": "홍길동"},
    )

    assert "이름=홍길동" in parts


def test_log_person_appearance_builds_and_inserts_payload(tmp_path):
    pipeline = AppearancePipeline(AppearanceAnalyzer(), tmp_path / "crops")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=1,
        y=2,
        width=20,
        height=25,
        confidence=0.95,
        timestamp=1000.0,
        object_id=3,
        class_name="person",
    )
    inserted_payloads = []

    class DummyAppearance:
        conditions = []

        def extract_attributes(self, *args, **kwargs):
            return {
                "upper_color": "white",
                "lower_color": "black",
                "has_helmet": False,
                "helmet_color": None,
                "has_backpack": False,
                "has_handbag": True,
                "has_suitcase": False,
                "attribute_backend": "hsv",
            }

    pipeline._appearance = DummyAppearance()
    pipeline._appearance_log = SimpleNamespace(
        insert=lambda **payload: inserted_payloads.append(payload)
    )

    pipeline.log_person_appearance(
        frame,
        person,
        1111.0,
        "cam01",
        [],
        {"gender": "female", "age_group": "adult", "face_name": "tester"},
    )

    assert len(inserted_payloads) == 1
    payload = inserted_payloads[0]
    assert payload["camera_id"] == "cam01"
    assert payload["track_id"] == 3
    assert payload["upper_color"] == "white"
    assert payload["lower_color"] == "black"
    assert payload["has_handbag"] is True
    assert payload["gender"] == "female"
    assert payload["face_name"] == "tester"


def test_log_person_appearance_saves_scaled_deepstream_crop(tmp_path):
    crop_dir = tmp_path / "crops"
    pipeline = AppearancePipeline(AppearanceAnalyzer(), crop_dir, save_crops=True)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[15:25, 10:30] = (255, 255, 255)
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=20,
        y=30,
        width=40,
        height=20,
        confidence=0.95,
        timestamp=1000.0,
        object_id=3,
        class_name="person",
        metadata={"frame_width": 100, "frame_height": 100},
    )
    inserted_payloads = []

    class DummyAppearance:
        conditions = []

        def extract_attributes(self, *args, **kwargs):
            return {"upper_color": "white", "lower_color": "black", "attribute_backend": "hsv"}

    pipeline._appearance = DummyAppearance()
    pipeline._appearance_log = SimpleNamespace(
        insert=lambda **payload: inserted_payloads.append(payload)
    )

    pipeline.log_person_appearance(frame, person, 1111.0, "cam01", [], {})

    assert len(inserted_payloads) == 1
    crop_path = inserted_payloads[0]["crop_path"]
    assert crop_path is not None
    saved = __import__("cv2").imread(crop_path)
    assert saved is not None
    assert saved.shape[:2] == (22, 42)
    assert saved[6:16, 10:30].mean() > 200


def test_extract_person_attributes_scales_deepstream_bbox_to_preview_frame(tmp_path):
    pipeline = AppearancePipeline(AppearanceAnalyzer(), tmp_path / "crops")
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=20,
        y=30,
        width=40,
        height=20,
        confidence=0.95,
        timestamp=1000.0,
        object_id=3,
        class_name="person",
        keypoints=[[30, 40, 0.9]],
        metadata={"frame_width": 100, "frame_height": 100},
    )
    nearby_objects = [{"class_name": "backpack", "x": 60, "y": 20, "width": 20, "height": 30}]
    captured = {}

    class DummyAppearance:
        def extract_attributes(self, frame_arg, x, y, width, height, nearby_objects=None, keypoints=None):
            captured.update(
                x=x,
                y=y,
                width=width,
                height=height,
                nearby_objects=nearby_objects,
                keypoints=keypoints,
            )
            return {"upper_color": "red", "lower_color": "blue", "attribute_backend": "hsv"}

    pipeline._appearance = DummyAppearance()

    attrs = pipeline._extract_person_attributes(frame, person, nearby_objects)

    assert attrs["upper_color"] == "red"
    assert (captured["x"], captured["y"], captured["width"], captured["height"]) == (10, 15, 20, 10)
    assert captured["keypoints"] == [[15.0, 20.0, 0.9]]
    assert captured["nearby_objects"][0]["x"] == 30
    assert captured["nearby_objects"][0]["height"] == 15


def test_smooth_track_attributes_uses_majority_color_per_track(tmp_path):
    pipeline = AppearancePipeline(
        AppearanceAnalyzer(),
        tmp_path / "crops",
        color_smoothing_window=5,
        color_min_samples=2,
    )
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=0,
        y=0,
        width=20,
        height=40,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )

    first = pipeline._smooth_track_attributes(
        "cam01",
        person,
        {"upper_color": "blue", "lower_color": "black", "attribute_backend": "hsv"},
    )
    second = pipeline._smooth_track_attributes(
        "cam01",
        person,
        {"upper_color": "black", "lower_color": "black", "attribute_backend": "hsv"},
    )
    third = pipeline._smooth_track_attributes(
        "cam01",
        person,
        {"upper_color": "blue", "lower_color": "unknown", "attribute_backend": "hsv"},
    )

    assert first["upper_color"] == "blue"
    assert second["upper_color"] == "black"
    assert third["upper_color"] == "blue"
    assert third["lower_color"] == "black"
    assert third["attribute_metadata"]["color_observations"]["upper_color"] == 3
    assert third["attribute_metadata"]["color_observations"]["lower_color"] == 2


def test_smooth_track_attributes_requires_stable_gender_samples(tmp_path):
    pipeline = AppearancePipeline(
        AppearanceAnalyzer(),
        tmp_path / "crops",
        color_smoothing_window=5,
        color_min_samples=2,
    )
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=0,
        y=0,
        width=20,
        height=40,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )

    first = pipeline._smooth_track_attributes("cam01", person, {"gender": "male"})
    second = pipeline._smooth_track_attributes("cam01", person, {"gender": "male"})
    third = pipeline._smooth_track_attributes("cam01", person, {"gender": "female"})

    assert first["gender"] == "unknown"
    assert second["gender"] == "unknown"
    assert third["gender"] == "male"
    assert third["attribute_metadata"]["gender_observations"] == 3
    assert third["attribute_metadata"]["gender_min_samples"] == 3


def test_extract_person_attributes_merges_deepstream_sgie_metadata(tmp_path):
    pipeline = AppearancePipeline(AppearanceAnalyzer(), tmp_path / "crops")
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=0,
        y=0,
        width=20,
        height=40,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
        metadata={
            "appearance": {
                "gender": "male",
                "age_group": "adult",
                "has_backpack": True,
            },
            "appearance_backend": "pphuman_sgie",
        },
    )
    frame = np.zeros((80, 40, 3), dtype=np.uint8)

    attrs = pipeline._extract_person_attributes(frame, person, [])

    assert attrs["gender"] == "male"
    assert attrs["age_group"] == "adult"
    assert attrs["has_backpack"] is True
    assert attrs["attribute_backend"] == "pphuman_sgie"


def test_build_log_parts_uses_attribute_gender_when_face_meta_missing():
    person = DetectionEvent(
        event_type=EventType.PERSON,
        x=0,
        y=0,
        width=20,
        height=40,
        confidence=0.9,
        timestamp=1000.0,
        object_id=7,
        class_name="person",
    )

    parts = AppearancePipeline._build_log_parts(
        person,
        {"gender": "male", "age_group": "adult", "has_helmet": False},
        {},
    )

    assert "성별=male" in parts
    assert "나이=adult" in parts
