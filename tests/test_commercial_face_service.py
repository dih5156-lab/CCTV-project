import base64

import cv2
import numpy as np

from src.core.ai._commercial_face_service import CommercialTensorRTFaceService
from src.core.ai._sqlite_face_gallery import SQLiteFaceGallery


class FakeRecognizer:
    def extract_enrollment_embedding(self, image):
        vector = np.zeros(128, dtype=np.float32)
        vector[int(image.mean()) % 2] = 1.0
        return vector

    def detect_and_recognize(self, frame, bbox):
        return []


def _image_base64(value: int) -> str:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode()


def _service(tmp_path):
    gallery = SQLiteFaceGallery(
        tmp_path / "faces.db", model_id="opencv-sface-tensorrt-v1"
    )
    return CommercialTensorRTFaceService(
        gallery=gallery,
        recognizer=FakeRecognizer(),
        faces_dir=tmp_path / "known_faces",
    )


def test_service_registers_one_photo_and_persists_image(tmp_path):
    service = _service(tmp_path)

    person = service.register_face("tester", "010", _image_base64(0))

    assert person["sample_count"] == 1
    assert person["enrollment_status"] == "single_sample"
    assert len(list((tmp_path / "known_faces").iterdir())) == 1


def test_service_same_phone_adds_sample_to_existing_person(tmp_path):
    service = _service(tmp_path)
    first = service.register_face("tester", "010", _image_base64(0))

    second = service.register_face("tester", "010", _image_base64(1))

    assert second["person_id"] == first["person_id"]
    assert second["sample_count"] == 2
    assert second["enrollment_status"] == "multi_sample"


def test_service_delete_removes_database_record_and_images(tmp_path):
    service = _service(tmp_path)
    person = service.register_face("tester", "010", _image_base64(0))
    service.register_face("tester", "010", _image_base64(1))

    assert service.delete_face(str(person["person_id"])) is True
    assert service.delete_face(str(person["person_id"])) is False
    assert service.list_faces() == []
    assert list((tmp_path / "known_faces").iterdir()) == []
