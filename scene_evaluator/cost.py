from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scene_evaluator.config import AppConfig, PROJECT_ROOT
from scene_evaluator.schemas import Usage


DEFAULT_LEDGER_PATH = PROJECT_ROOT / "logs" / "cost_ledger.json"


def estimate_cost_usd(
    *,
    model: str,
    usage: Usage,
    pricing_per_1m_tokens: dict[str, dict[str, float]],
) -> float:
    pricing = pricing_per_1m_tokens.get(model)
    if not pricing:
        return 0.0

    non_cached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        (non_cached_input / 1_000_000) * float(pricing["input"])
        + (usage.cached_input_tokens / 1_000_000) * float(pricing["cached_input"])
        + (usage.output_tokens / 1_000_000) * float(pricing["output"])
    )


class CostLedger:
    def __init__(
        self,
        app_config: AppConfig,
        path: Path | str = DEFAULT_LEDGER_PATH,
    ) -> None:
        self.app_config = app_config
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def can_spend(self) -> bool:
        ledger = self.read()
        if not self.app_config.budget.stop_when_exceeded:
            return True
        return float(ledger["total_estimated_cost_usd"]) < self.app_config.budget.max_usd

    def record_call(
        self,
        *,
        run_id: str,
        profile: str,
        prompt_version: str,
        role: str,
        model: str,
        usage: Usage,
    ) -> dict[str, Any]:
        ledger = self.read()
        estimated_cost = estimate_cost_usd(
            model=model,
            usage=usage,
            pricing_per_1m_tokens=self.app_config.pricing_per_1m_tokens,
        )

        ledger["total_estimated_cost_usd"] += estimated_cost
        ledger["total_input_tokens"] += usage.input_tokens
        ledger["total_cached_input_tokens"] += usage.cached_input_tokens
        ledger["total_output_tokens"] += usage.output_tokens
        ledger["call_count"] += 1
        ledger["last_updated_at"] = _utc_now()
        ledger["calls"].append(
            {
                "run_id": run_id,
                "profile": profile,
                "prompt_version": prompt_version,
                "role": role,
                "model": model,
                "usage": usage.to_dict(),
                "estimated_cost_usd": estimated_cost,
                "created_at": ledger["last_updated_at"],
            }
        )
        self._write(ledger)
        return ledger

    def record_fixed_cost(
        self,
        *,
        run_id: str,
        profile: str,
        prompt_version: str,
        role: str,
        model: str,
        estimated_cost_usd: float,
        usage: Usage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ledger = self.read()
        usage = usage or Usage()
        ledger["total_estimated_cost_usd"] += estimated_cost_usd
        ledger["total_input_tokens"] += usage.input_tokens
        ledger["total_cached_input_tokens"] += usage.cached_input_tokens
        ledger["total_output_tokens"] += usage.output_tokens
        ledger["call_count"] += 1
        ledger["last_updated_at"] = _utc_now()
        ledger["calls"].append(
            {
                "run_id": run_id,
                "profile": profile,
                "prompt_version": prompt_version,
                "role": role,
                "model": model,
                "usage": usage.to_dict(),
                "estimated_cost_usd": estimated_cost_usd,
                "metadata": metadata or {},
                "created_at": ledger["last_updated_at"],
            }
        )
        self._write(ledger)
        return ledger

    def record_run_complete(self, run_id: str) -> dict[str, Any]:
        ledger = self.read()
        ledger["runs"] += 1
        ledger["completed_run_ids"].append(run_id)
        ledger["last_updated_at"] = _utc_now()
        self._write(ledger)
        return ledger

    def _empty(self) -> dict[str, Any]:
        return {
            "budget_usd": self.app_config.budget.max_usd,
            "warn_at_usd": self.app_config.budget.warn_at_usd,
            "total_estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "call_count": 0,
            "runs": 0,
            "completed_run_ids": [],
            "calls": [],
            "last_updated_at": None,
        }

    def _write(self, ledger: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
