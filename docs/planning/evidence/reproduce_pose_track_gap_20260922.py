"""P5.1 입력 계약 차이 재현. 모델 추론·MQTT·장치 출력은 실행하지 않는다."""

import json
from pathlib import Path

from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier
from src.core.events import DetectionEvent, EventType


def reproduce():
    verifier = FallDataAuxVerifier(FallDataAuxConfig(enabled=True, inline_pose_rf=True))
    expected_winners = [101, 202, 101]
    for frame_num, winner in enumerate(expected_winners, start=1):
        events = []
        for track_id, width in ((101, 80), (202, 320)):
            events.append(
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
                        "fall_reasons": [],
                    },
                )
            )
        verifier.add_pose_events("synthetic_camera", events)
    records = list(verifier._pose_records["synthetic_camera"])
    inferred_tracks = [101 if r["bbox_aspect"] == 0.5 else 202 for r in records]
    result = {
        "scope": "synthetic input buffering only; no classifier or physical output",
        "input_tracks_each_frame": [101, 202],
        "expected_high_score_track_sequence": expected_winners,
        "observed_track_sequence_from_synthetic_bbox_aspect": inferred_tracks,
        "buffer_count": len(verifier._pose_records),
        "record_count": len(records),
        "record_has_track_identity": any(
            "object_id" in r or "track_id" in r for r in records
        ),
        "single_track_window_contract_met": len(set(inferred_tracks)) == 1,
        "actual_false_alarm_observed": False,
    }
    assert inferred_tracks == expected_winners
    assert not result["single_track_window_contract_met"]
    assert not result["record_has_track_identity"]
    return result


if __name__ == "__main__":
    result = reproduce()
    Path(__file__).with_name("pose_track_gap_20260922.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))
