from scene_evaluator.config import load_config


def test_model_controls_are_scoped_inside_profiles():
    config = load_config()
    assert set(config.profiles) == {
        "budget_testing",
        "quality_run",
        "final_prompt_tuning",
    }
    assert config.get_profile("budget_testing").generator.reasoning_format == "low"
    assert config.get_profile("budget_testing").critic.image_detail == "low"


def test_budget_testing_models_match_agreed_profile():
    profile = load_config().get_profile("budget_testing")
    assert profile.generator.model == "gpt-5.4-nano"
    assert profile.critic.model == "gpt-5.4-mini"


def test_gold_truth_image_generation_config_uses_best_image_model():
    config = load_config()
    assert config.gold_truth_image_generation.model == "gpt-image-1.5"
    assert config.gold_truth_image_generation.quality == "high"


def test_candidate_image_generation_config_exists():
    config = load_config()
    assert config.candidate_image_generation.model == "gpt-image-1.5"
    assert config.candidate_image_generation.quality == "medium"
