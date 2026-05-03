from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.json"


@dataclass(frozen=True)
class ModelRoleConfig:
    provider: str
    model: str
    max_output_tokens: int
    reasoning_format: str | None = None
    image_detail: str | None = None


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    description: str
    generator: ModelRoleConfig
    critic: ModelRoleConfig


@dataclass(frozen=True)
class BudgetConfig:
    max_usd: float
    warn_at_usd: float
    stop_when_exceeded: bool


@dataclass(frozen=True)
class GoldTruthImageConfig:
    provider: str
    model: str
    size: str
    quality: str
    output_format: str
    cost_per_image_usd: float


@dataclass(frozen=True)
class AppConfig:
    active_profile: str
    budget: BudgetConfig
    gold_truth_image_generation: GoldTruthImageConfig
    candidate_image_generation: GoldTruthImageConfig
    profiles: dict[str, ProfileConfig]
    pricing_per_1m_tokens: dict[str, dict[str, float]]

    def get_profile(self, name: str | None = None) -> ProfileConfig:
        profile_name = name or self.active_profile
        try:
            return self.profiles[profile_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles))
            raise ValueError(
                f"Unknown profile '{profile_name}'. Available profiles: {available}"
            ) from exc


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_profile_scoped_controls(data)

    budget_data = data["budget"]
    gold_truth_data = data["gold_truth_image_generation"]
    candidate_image_data = data["candidate_image_generation"]
    profiles = {
        name: ProfileConfig(
            name=name,
            description=profile_data.get("description", ""),
            generator=_role_config(profile_data["generator"]),
            critic=_role_config(profile_data["critic"]),
        )
        for name, profile_data in data["profiles"].items()
    }

    return AppConfig(
        active_profile=data["active_profile"],
        budget=BudgetConfig(
            max_usd=float(budget_data["max_usd"]),
            warn_at_usd=float(budget_data["warn_at_usd"]),
            stop_when_exceeded=bool(budget_data["stop_when_exceeded"]),
        ),
        gold_truth_image_generation=GoldTruthImageConfig(
            provider=gold_truth_data["provider"],
            model=gold_truth_data["model"],
            size=gold_truth_data["size"],
            quality=gold_truth_data["quality"],
            output_format=gold_truth_data["output_format"],
            cost_per_image_usd=float(gold_truth_data["cost_per_image_usd"]),
        ),
        candidate_image_generation=GoldTruthImageConfig(
            provider=candidate_image_data["provider"],
            model=candidate_image_data["model"],
            size=candidate_image_data["size"],
            quality=candidate_image_data["quality"],
            output_format=candidate_image_data["output_format"],
            cost_per_image_usd=float(candidate_image_data["cost_per_image_usd"]),
        ),
        profiles=profiles,
        pricing_per_1m_tokens=data["pricing_per_1m_tokens"],
    )


def _role_config(data: dict[str, Any]) -> ModelRoleConfig:
    return ModelRoleConfig(
        provider=data["provider"],
        model=data["model"],
        max_output_tokens=int(data["max_output_tokens"]),
        reasoning_format=data.get("reasoning_format"),
        image_detail=data.get("image_detail"),
    )


def _validate_profile_scoped_controls(data: dict[str, Any]) -> None:
    """Keep model-control knobs inside profile tags, as requested."""
    forbidden_top_level = {"reasoning_format", "image_detail"}
    leaked = forbidden_top_level.intersection(data)
    if leaked:
        raise ValueError(
            "Model controls must be scoped inside profiles, not top level: "
            + ", ".join(sorted(leaked))
        )

    for key in (
        "profiles",
        "pricing_per_1m_tokens",
        "budget",
        "active_profile",
        "gold_truth_image_generation",
        "candidate_image_generation",
    ):
        if key not in data:
            raise ValueError(f"Missing required config key: {key}")

    for profile_name, profile_data in data["profiles"].items():
        for role in ("generator", "critic"):
            if role not in profile_data:
                raise ValueError(f"Profile '{profile_name}' is missing '{role}' config")
