"""I24 수정 후 인물별 입력 대조. CPU 모형 분류기만 사용한다."""

import json
from pathlib import Path

import numpy as np

from scripts.datasets.train_yolo_pose_fall_rf import FEATURE_NAMES
from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier
from src.core.events import DetectionEvent, EventType


class RecordingClassifier:
    classes_ = np.asarray([0, 1])
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self):
        self.aspects = []

    def predict_proba(self, features):
        self.aspects.append(float(features[0, FEATURE_NAMES.index("mean_bbox_aspect")]))
        return np.asarray([[0.8, 0.2]])


def verify():
    classifier = RecordingClassifier()
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, inline_pose_rf=True, cooldown_seconds=60),
        inline_pose_rf_bundle={"model": classifier, "feature_names": FEATURE_NAMES},
    )
    for frame_num, winner in enumerate([101, 202, 101], start=1):
        events = [
            DetectionEvent(
                event_type=EventType.PERSON,
                x=10,
                y=20,
                width=width,
                height=160,
                confidence=0.92,
                timestamp=100.0 + frame_num / 30.0,
                object_id=track_id,
                keypoints=[[float(10 + i), float(20 + i), 0.9] for i in range(17)],
                metadata={
                    "frame_num": frame_num,
                    "frame_width": 640,
                    "frame_height": 480,
                    "fall_score": 4.0 if track_id == winner else 1.0,
                },
            )
            for track_id, width in [(101, 80), (202, 320)]
        ]
        verifier.add_pose_events("synthetic_camera", events)
    results = [
        verifier.verify(
            camera_name="synthetic_camera", pose_scope=e.metadata["falldata_pose_scope"]
        )
        for e in events
    ]
    assert classifier.aspects == [0.5, 2.0]
    assert all(
        result["frames_with_pose"] == 3 and not result["cache_reused"]
        for result in results
    )
    return {
        "scope": "CPU synthetic input/model-feature isolation; no real classifier, GPU or physical output",
        "before": "pose_track_gap_20260922.json (historical pre-fix evidence)",
        "alternating_high_score_tracks": [101, 202, 101],
        "buffered_records": len(verifier._pose_records["synthetic_camera"]),
        "model_input_mean_bbox_aspects": classifier.aspects,
        "verified_tracks": [result["pose_scope"]["object_id"] for result in results],
        "frames_per_track": [result["frames_with_pose"] for result in results],
        "same_camera_other_track_cache_reused": results[1]["cache_reused"],
        "distinct_track_input_isolated": True,
        "physical_identity_accuracy_verified": False,
    }


if __name__ == "__main__":
    result = verify()
    Path(__file__).with_name("pose_track_isolation_example_20260922.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))
