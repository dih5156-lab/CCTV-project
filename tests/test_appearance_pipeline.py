"""AppearancePipeline 저장 옵션 테스트."""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace

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
        crop_path="data/appearance_crops/cam01_7.jpg",
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
