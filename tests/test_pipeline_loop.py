from scene_evaluator import pipeline
from scene_evaluator.openai_client import DryRunClient


def test_pipeline_dry_run_completes_without_api_calls(monkeypatch):
    monkeypatch.setattr(pipeline, "log_experiment", lambda record: None)
    monkeypatch.setattr(pipeline, "log_generation", lambda record: None)

    result = pipeline.run_pipeline(
        case_id="alleyway_kael",
        profile_name="budget_testing",
        prompt_version="v1",
        max_iterations=2,
        dry_run=True,
        client=DryRunClient(),
    )

    assert result["dry_run"] is True
    assert result["final_score"] == 42
    assert len(result["iterations"]) == 3
    assert "brass compass" in result["final_prompt"]


def test_gold_truth_dry_run_writes_reference_image(monkeypatch, tmp_path):
    test_case = pipeline.load_test_case("alleyway_kael")
    patched_case = pipeline.TestCase(
        case_id=test_case.case_id,
        scene_text=test_case.scene_text,
        image_url=test_case.image_url,
        expected_metadata=test_case.expected_metadata,
        case_dir=tmp_path,
        gold_truth_image_path=tmp_path / "gold_truth.png",
        gold_truth_metadata_path=tmp_path / "gold_truth_metadata.json",
    )

    monkeypatch.setattr(pipeline, "load_test_case", lambda case_id: patched_case)
    monkeypatch.setattr(pipeline, "log_generation", lambda record: None)

    result = pipeline.generate_gold_truth_image_for_case(
        case_id="alleyway_kael",
        prompt_version="v1",
        dry_run=True,
        client=DryRunClient(),
    )

    assert result["dry_run"] is True
    assert patched_case.gold_truth_image_path.exists()
    assert patched_case.gold_truth_metadata_path.exists()
