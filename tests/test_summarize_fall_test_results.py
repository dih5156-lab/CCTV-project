from pathlib import Path

from scripts.ops.summarize_fall_test_results import summarize_manifest


def test_summarize_manifest_counts_confusion_and_pending(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_path,pose,expected_fall,model_prediction\n"
        "a.mp4,forward_fall,1,1\n"
        "b.mp4,walking,0,1\n"
        "c.mp4,backward_fall,1,0\n"
        "d.mp4,left_fall,,\n",
        encoding="utf-8",
    )

    result = summarize_manifest(manifest)

    assert result["evaluated_rows"] == 3
    assert result["pending_rows"] == 1
    assert result["confusion"] == {"tp": 1, "tn": 0, "fp": 1, "fn": 1}
