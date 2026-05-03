# Agentic Scene Evaluator

This repo builds a small Python CLI pipeline for evaluating whether an image
matches a scripted scene.

It takes a raw scene, generates a text-to-image prompt, generates a candidate
image from that prompt, critiques the candidate image against a gold-truth
reference, and automatically revises the prompt when the image score is below the
passing threshold.

## What It Does

- Reads a scene from `test_cases/<case_id>/scene.txt`.
- Uses mock character metadata, currently `Kael`, to keep prompt details
  consistent.
- Generates an initial text-to-image prompt.
- Generates a gold-truth reference image for each test case.
- Treats the gold-truth image as the 100% scoring reference.
- Generates a candidate image from every prompt produced by the generator.
- Sends the generated prompt, generated candidate image, and gold-truth image to
  a VLM critic.
- Gets back a strict JSON critique with:
  - overall match score
  - missing elements
  - hallucinated elements
  - actionable feedback
- If the score is below `90`, revises the prompt using critic feedback.
- Generates a new candidate image from each revised prompt before scoring again.
- Stops after a maximum of 2 revision iterations.
- Logs generations, experiment runs, token usage, and estimated cumulative cost.

## Repository Layout

```text
scene_evaluator/       Python package and CLI pipeline
configs/               Model profiles, image model config, and budget settings
prompts/               Versioned prompt templates
test_cases/            Scene fixtures, candidate image URLs, and gold references
logs/                  Runtime logs and cost ledger
tests/                 Pytest tests
```

## Model Profiles

The model profiles are defined in `configs/model_config.json`.

```text
budget_testing        Cheap profile for prompt and pipeline iteration
quality_run           Higher quality scoring run
final_prompt_tuning   Expensive final run after prompts are selected
```

Gold-truth and candidate image generation use `gpt-image-1.5`.

## Prompt Versions

The active prompt version is configured in `prompts/prompt_versions.json`.

Current prompt set:

```text
prompts/v1/generator_initial.md
prompts/v1/critic_continuity_supervisor.md
prompts/v1/generator_revision.md
prompts/v1/gold_truth_image_generation.md
```

## Test Cases

Each test case should contain:

```text
scene.txt
image_url.txt
expected_metadata.json
```

After generating the gold-truth reference, the case will also contain:

```text
gold_truth.png
gold_truth_prompt.txt
gold_truth_metadata.json
```

Pipeline runs write generated candidate images to:

```text
logs/generated_images/<run_id>/
```

Each iteration writes:

```text
iteration_00_candidate.png
iteration_00_candidate_prompt.txt
iteration_00_candidate_metadata.json
```

## Assumptions

- The gold-truth image is generated from the scene and metadata using the best
  configured image model, so it represents the best available reference this
  system can create.
- The gold-truth image can be biased by the gold-generation prompt, model style,
  and model interpretation of the scene.
- The critic treats the gold-truth image as the 100% reference point, so scores
  measure similarity to that generated reference, not an objective ground truth.
- A revised prompt is only useful when the pipeline also generates a fresh
  candidate image from that revised prompt.
- Run `cf4195c0-ec14-4977-8fe0-9b76f810977b` produced a decent result and can be
  used as a reference run when comparing later experiments.

## Cost Tracking

The repo keeps a cumulative cost ledger in:

```text
logs/cost_ledger.json
```

Generation and experiment details are logged to:

```text
logs/generations.jsonl
logs/experiments.jsonl
```

`logs/generations.jsonl` stores each prompt call with both `response_text` and
parsed `response_json` when the output is valid JSON.

Candidate image generation records include the generated image path in
`response_json.image_path`.

Dry runs do not update the cost ledger.

## Commands

Set up:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
```

List model profiles:

```bash
python -m scene_evaluator.cli profiles
```

Generate gold-truth image for a test case:

```bash
python -m scene_evaluator.cli generate-gold --case alleyway_kael
```

Run the pipeline:

```bash
python -m scene_evaluator.cli generate-gold --case alleyway_kael
python -m scene_evaluator.cli run --case alleyway_kael --profile budget_testing
```

Run a higher quality pass:

```bash
python -m scene_evaluator.cli run --case alleyway_kael --profile quality_run
```

Run final prompt tuning:

```bash
python -m scene_evaluator.cli run --case alleyway_kael --profile final_prompt_tuning
```

Dry-run gold generation:

```bash
python -m scene_evaluator.cli generate-gold --case alleyway_kael --dry-run --overwrite
```

Dry-run pipeline:

```bash
python -m scene_evaluator.cli run --case alleyway_kael --profile budget_testing --dry-run
```

Inspect generated candidate images:

```bash
find logs/generated_images -type f
```

Inspect the latest experiment log:

```bash
tail -n 1 logs/experiments.jsonl
```

Inspect individual model outputs:

```bash
tail -n 5 logs/generations.jsonl
```

Run tests:

```bash
python -m pytest
```
