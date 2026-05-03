from pathlib import Path

from scene_evaluator.config import load_config
from scene_evaluator.cost import CostLedger, estimate_cost_usd
from scene_evaluator.schemas import Usage


def test_estimate_cost_uses_cached_input_discount():
    config = load_config()
    cost = estimate_cost_usd(
        model="gpt-5.4-mini",
        usage=Usage(input_tokens=1000, cached_input_tokens=400, output_tokens=200),
        pricing_per_1m_tokens=config.pricing_per_1m_tokens,
    )
    expected = (600 / 1_000_000) * 0.75 + (400 / 1_000_000) * 0.075 + (
        200 / 1_000_000
    ) * 4.50
    assert cost == expected


def test_cost_ledger_records_call(tmp_path: Path):
    config = load_config()
    ledger = CostLedger(config, tmp_path / "ledger.json")
    updated = ledger.record_call(
        run_id="run-1",
        profile="budget_testing",
        prompt_version="v1",
        role="critic",
        model="gpt-5.4-mini",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    assert updated["call_count"] == 1
    assert updated["total_input_tokens"] == 10
    assert updated["total_output_tokens"] == 5
