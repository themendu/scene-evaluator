# Prompt Set v1

Version: `v1`

Status: draft

Purpose: explicit prompt templates for the first scene evaluator experiment.

Prompt files:

- `generator_initial.md`: creates the first text-to-image prompt from scene text
  and required character metadata.
- `critic_continuity_supervisor.md`: grades a candidate image against the
  generated prompt and the gold-truth reference image.
- `generator_revision.md`: revises the prompt using the critic's actionable
  feedback.
- `gold_truth_image_generation.md`: generates the reference image that is treated
  as the 100% target for critic scoring.

Gold-truth image generation model:

- `gpt-image-1.5`
