from scene_evaluator.prompts import extract_json_object, load_prompt_set, render_template


def test_load_prompt_set_v1():
    prompt_set = load_prompt_set("v1")
    assert prompt_set["version"] == "v1"
    assert "generator_initial" in prompt_set
    assert "critic_continuity_supervisor" in prompt_set
    assert "gold_truth_image_generation" in prompt_set


def test_render_template_replaces_values():
    assert render_template("Scene: {{ scene_text }}", {"scene_text": "rain"}) == "Scene: rain"


def test_extract_json_object_handles_fenced_json():
    data = extract_json_object('```json\n{"a": 1}\n```')
    assert data == {"a": 1}
