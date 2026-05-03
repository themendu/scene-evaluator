from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CriticResult:
    overall_match_score: int
    missing_elements: list[str]
    hallucinated_elements: list[str]
    actionable_feedback: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriticResult":
        required = {
            "overall_match_score",
            "missing_elements",
            "hallucinated_elements",
            "actionable_feedback",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError("Critic JSON missing keys: " + ", ".join(sorted(missing)))

        score = data["overall_match_score"]
        if not isinstance(score, int) or not 0 <= score <= 100:
            raise ValueError("overall_match_score must be an integer from 0 to 100")

        missing_elements = _string_list(data["missing_elements"], "missing_elements")
        hallucinated_elements = _string_list(
            data["hallucinated_elements"], "hallucinated_elements"
        )
        feedback = data["actionable_feedback"]
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("actionable_feedback must be a non-empty string")

        return cls(
            overall_match_score=score,
            missing_elements=missing_elements,
            hallucinated_elements=hallucinated_elements,
            actionable_feedback=feedback,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_match_score": self.overall_match_score,
            "missing_elements": self.missing_elements,
            "hallucinated_elements": self.hallucinated_elements,
            "actionable_feedback": self.actionable_feedback,
        }


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
        }


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value

