from pathlib import Path

from scripts.promote_fall_model import promotion_plan


def test_promotion_is_dry_run_without_approval(tmp_path: Path):
    result = promotion_plan(
        {"promote_candidate": True},
        tmp_path / "candidate.pkl",
        tmp_path / "current.pkl",
        approve=False,
    )
    assert result["action"] == "keep_current"
    assert result["approved"] is False


def test_failed_comparison_never_promotes_even_with_approval(tmp_path: Path):
    result = promotion_plan(
        {"promote_candidate": False},
        tmp_path / "candidate.pkl",
        tmp_path / "current.pkl",
        approve=True,
    )
    assert result["action"] == "keep_current"
