from pathlib import Path


def test_testing_document_index_points_to_existing_files():
    docs_dir = Path(__file__).resolve().parents[1] / "docs" / "testing"
    index = (docs_dir / "README.md").read_text(encoding="utf-8")
    expected = {
        "FALL_TEST_DATA_COLLECTION.md",
        "FALL_MODEL_COMPARISON_RUNBOOK.md",
        "POSE_CROP_COMPARISON_CHECKLIST.md",
        "pose_crop_compare_report.template.json",
    }
    for filename in expected:
        assert f"]({filename})" in index
        assert (docs_dir / filename).is_file()

