import numpy as np

from src.core.ai._sqlite_face_gallery import SQLiteFaceGallery


def _unit(index):
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_sqlite_gallery_persists_people_samples_and_search(tmp_path):
    database = tmp_path / "faces.db"
    gallery = SQLiteFaceGallery(database, model_id="opencv-sface-tensorrt-v1")

    person = gallery.enroll_person(
        {
            "person_id": "employee-1",
            "name": "tester",
            "phone": "010-0000-0000",
            "category": "employee",
        },
        [_unit(0)],
        ["known_faces/one.jpg"],
    )

    assert person["sample_count"] == 1
    reloaded = SQLiteFaceGallery(database, model_id="opencv-sface-tensorrt-v1")
    assert reloaded.search(_unit(0), threshold=0.5).best.person_id == "employee-1"
    assert reloaded.list_people()[0]["enrollment_status"] == "single_sample"


def test_sqlite_gallery_adds_samples_to_existing_person(tmp_path):
    gallery = SQLiteFaceGallery(
        tmp_path / "faces.db", model_id="opencv-sface-tensorrt-v1"
    )
    gallery.enroll_person(
        {"person_id": "a", "name": "alpha", "phone": "010"},
        [_unit(0)],
        ["known_faces/one.jpg"],
    )

    updated = gallery.add_samples("a", [_unit(1)], ["known_faces/two.jpg"])

    assert updated["sample_count"] == 2
    assert updated["enrollment_status"] == "multi_sample"
    assert len(updated["images"]) == 2
    assert gallery.search(_unit(1), threshold=0.5).best.person_id == "a"


def test_sqlite_gallery_deactivate_and_delete_persist(tmp_path):
    database = tmp_path / "faces.db"
    gallery = SQLiteFaceGallery(database, model_id="opencv-sface-tensorrt-v1")
    gallery.enroll_person(
        {"person_id": "a", "name": "alpha", "phone": "010"},
        [_unit(0)],
        ["known_faces/one.jpg"],
    )

    gallery.deactivate("a")
    assert gallery.search(_unit(0), threshold=0.5).matched is False
    assert SQLiteFaceGallery(database, model_id="opencv-sface-tensorrt-v1").search(
        _unit(0), threshold=0.5
    ).matched is False

    assert gallery.delete("a") is True
    assert gallery.delete("a") is False
    assert gallery.list_people() == []


def test_sqlite_gallery_does_not_load_other_model_embeddings(tmp_path):
    database = tmp_path / "faces.db"
    first = SQLiteFaceGallery(database, model_id="opencv-sface-tensorrt-v1")
    first.enroll_person(
        {"person_id": "a", "name": "alpha", "phone": "010"},
        [_unit(0)],
        ["known_faces/one.jpg"],
    )

    other = SQLiteFaceGallery(database, model_id="future-model-v2")

    assert other.list_people() == []
    assert other.search(_unit(0)).best is None
