import pytest

from scene_evaluator.schemas import CriticResult


def test_critic_result_accepts_required_schema():
    result = CriticResult.from_dict(
        {
            "overall_match_score": 88,
            "missing_elements": ["compass"],
            "hallucinated_elements": [],
            "actionable_feedback": "Make the compass prominent.",
        }
    )
    assert result.overall_match_score == 88


def test_critic_result_rejects_bad_score():
    with pytest.raises(ValueError):
        CriticResult.from_dict(
            {
                "overall_match_score": 101,
                "missing_elements": [],
                "hallucinated_elements": [],
                "actionable_feedback": "x",
            }
        )
