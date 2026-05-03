from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scene_evaluator.config import AppConfig, PROJECT_ROOT, load_config
from scene_evaluator.cost import CostLedger
from scene_evaluator.logging_utils import log_experiment, log_generation, utc_now
from scene_evaluator.openai_client import (
    DryRunClient,
    ImageGenerationResult,
    ModelCallResult,
    OpenAIResponsesClient,
    image_file_to_data_uri,
)
from scene_evaluator.prompts import extract_json_object, load_prompt_set, render_template
from scene_evaluator.schemas import CriticResult


class SceneModelClient(Protocol):
    def generate_initial_prompt(
        self,
        *,
        rendered_prompt: str,
        role_config: Any,
    ) -> ModelCallResult: ...

    def revise_prompt(
        self,
        *,
        rendered_prompt: str,
        role_config: Any,
    ) -> ModelCallResult: ...

    def critique_image(
        self,
        *,
        rendered_prompt: str,
        image_url: str,
        gold_truth_image_reference: str,
        role_config: Any,
    ) -> ModelCallResult: ...

    def generate_gold_truth_image(
        self,
        *,
        rendered_prompt: str,
        image_config: Any,
    ) -> ImageGenerationResult: ...

    def generate_candidate_image(
        self,
        *,
        rendered_prompt: str,
        image_config: Any,
    ) -> ImageGenerationResult: ...


@dataclass(frozen=True)
class TestCase:
    case_id: str
    scene_text: str
    image_url: str
    expected_metadata: dict[str, Any]
    case_dir: Path
    gold_truth_image_path: Path
    gold_truth_metadata_path: Path


def run_pipeline(
    *,
    case_id: str,
    profile_name: str | None = None,
    prompt_version: str | None = None,
    max_iterations: int = 2,
    dry_run: bool = False,
    app_config: AppConfig | None = None,
    client: SceneModelClient | None = None,
) -> dict[str, Any]:
    app_config = app_config or load_config()
    profile = app_config.get_profile(profile_name)
    prompt_set = load_prompt_set(prompt_version)
    prompt_version_name = prompt_set["version"]
    test_case = load_test_case(case_id)
    gold_truth_reference = _gold_truth_reference(test_case, allow_missing=dry_run)
    run_id = str(uuid.uuid4())
    ledger = CostLedger(app_config)

    if not dry_run and not ledger.can_spend():
        raise RuntimeError("Budget cap reached; refusing to start another API run.")

    client = client or (DryRunClient() if dry_run else OpenAIResponsesClient())
    started_at = utc_now()

    print(f"Run ID: {run_id}")
    print(f"Profile: {profile.name}")
    print(f"Prompt version: {prompt_version_name}")
    print(f"Case: {test_case.case_id}")

    initial_rendered = render_template(
        prompt_set["generator_initial"],
        {"scene_text": test_case.scene_text},
    )
    _assert_can_call(ledger=ledger, dry_run=dry_run)
    initial_result = client.generate_initial_prompt(
        rendered_prompt=initial_rendered,
        role_config=profile.generator,
    )
    _record_call(
        app_config=app_config,
        ledger=ledger,
        dry_run=dry_run,
        run_id=run_id,
        profile_name=profile.name,
        prompt_version=prompt_version_name,
        role="generator_initial",
        model=profile.generator.model,
        rendered_prompt=initial_rendered,
        result=initial_result,
    )

    initial_json = extract_json_object(initial_result.text)
    current_prompt = initial_json["text_to_image_prompt"]
    character_metadata = initial_json.get("character_metadata_used", {})
    _print_json("Initial Generator Output", initial_json)
    print(f"Initial Prompt: {current_prompt}")

    iterations: list[dict[str, Any]] = []
    final_score: int | None = None

    for iteration_number in range(0, max_iterations + 1):
        candidate = _generate_candidate_image_for_prompt(
            client=client,
            app_config=app_config,
            ledger=ledger,
            dry_run=dry_run,
            run_id=run_id,
            profile_name=profile.name,
            prompt_version=prompt_version_name,
            iteration_number=iteration_number,
            prompt=current_prompt,
        )
        candidate_image_reference = image_file_to_data_uri(Path(candidate["image_path"]))
        print(f"Generated Candidate Image: {candidate['image_path']}")

        critic = _run_critic_with_one_retry(
            client=client,
            app_config=app_config,
            ledger=ledger,
            dry_run=dry_run,
            run_id=run_id,
            profile_name=profile.name,
            prompt_version=prompt_version_name,
            prompt_set=prompt_set,
            prompt=current_prompt,
            image_url="attached generated candidate image",
            candidate_image_reference=candidate_image_reference,
            gold_truth_image_reference=gold_truth_reference,
            critic_config=profile.critic,
        )
        final_score = critic.overall_match_score
        _print_json("Critic Output", critic.to_dict())
        print(f"VLM Score: {critic.overall_match_score}")

        iteration_record: dict[str, Any] = {
            "iteration": iteration_number,
            "prompt": current_prompt,
            "candidate_image": candidate,
            "critic": critic.to_dict(),
        }
        iterations.append(iteration_record)

        if critic.overall_match_score >= 90:
            break
        if iteration_number >= max_iterations:
            break

        revision_rendered = render_template(
            prompt_set["generator_revision"],
            {
                "scene_text": test_case.scene_text,
                "character_metadata": character_metadata,
                "previous_prompt": current_prompt,
                "critic_json": critic.to_dict(),
                "iteration_number": str(iteration_number + 1),
                "max_iterations": str(max_iterations),
            },
        )
        _assert_can_call(ledger=ledger, dry_run=dry_run)
        revision_result = client.revise_prompt(
            rendered_prompt=revision_rendered,
            role_config=profile.generator,
        )
        _record_call(
            app_config=app_config,
            ledger=ledger,
            dry_run=dry_run,
            run_id=run_id,
            profile_name=profile.name,
            prompt_version=prompt_version_name,
            role="generator_revision",
            model=profile.generator.model,
            rendered_prompt=revision_rendered,
            result=revision_result,
        )
        revision_json = extract_json_object(revision_result.text)
        _print_json("Revision Generator Output", revision_json)
        current_prompt = revision_json["revised_text_to_image_prompt"]
        print(f"Revised Prompt: {current_prompt}")

    final_ledger = None if dry_run else ledger.record_run_complete(run_id)
    run_record = {
        "run_id": run_id,
        "dry_run": dry_run,
        "case_id": test_case.case_id,
        "scene_hash": _sha256(test_case.scene_text),
        "image_url": test_case.image_url,
        "gold_truth_image_path": str(test_case.gold_truth_image_path),
        "profile": profile.name,
        "prompt_version": prompt_version_name,
        "generator_model": profile.generator.model,
        "critic_model": profile.critic.model,
        "max_iterations": max_iterations,
        "initial_prompt": initial_json,
        "iterations": iterations,
        "final_prompt": current_prompt,
        "final_score": final_score,
        "ledger_after_run": _ledger_summary(final_ledger),
        "started_at": started_at,
        "completed_at": utc_now(),
    }
    log_experiment(run_record)
    return run_record


def generate_gold_truth_image_for_case(
    *,
    case_id: str,
    prompt_version: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    app_config: AppConfig | None = None,
    client: SceneModelClient | None = None,
) -> dict[str, Any]:
    app_config = app_config or load_config()
    prompt_set = load_prompt_set(prompt_version)
    prompt_version_name = prompt_set["version"]
    test_case = load_test_case(case_id)
    run_id = str(uuid.uuid4())
    ledger = CostLedger(app_config)

    if test_case.gold_truth_image_path.exists() and not overwrite:
        raise RuntimeError(
            "Gold-truth image already exists. Pass --overwrite to replace it: "
            f"{test_case.gold_truth_image_path}"
        )
    if not dry_run and not ledger.can_spend():
        raise RuntimeError("Budget cap reached; refusing to generate a gold-truth image.")

    client = client or (DryRunClient() if dry_run else OpenAIResponsesClient())
    rendered_prompt = render_template(
        prompt_set["gold_truth_image_generation"],
        {
            "scene_text": test_case.scene_text,
            "character_metadata": test_case.expected_metadata,
        },
    )
    print(f"Run ID: {run_id}")
    print(f"Case: {test_case.case_id}")
    print(f"Gold model: {app_config.gold_truth_image_generation.model}")

    _assert_can_call(ledger=ledger, dry_run=dry_run)
    result = client.generate_gold_truth_image(
        rendered_prompt=rendered_prompt,
        image_config=app_config.gold_truth_image_generation,
    )

    test_case.gold_truth_image_path.write_bytes(result.image_bytes)
    prompt_path = test_case.case_dir / "gold_truth_prompt.txt"
    prompt_path.write_text(rendered_prompt, encoding="utf-8")

    estimated_cost = (
        0.0 if dry_run else app_config.gold_truth_image_generation.cost_per_image_usd
    )
    ledger_after = None
    if not dry_run:
        ledger_after = ledger.record_fixed_cost(
            run_id=run_id,
            profile="gold_truth_image_generation",
            prompt_version=prompt_version_name,
            role="gold_truth_image_generation",
            model=app_config.gold_truth_image_generation.model,
            estimated_cost_usd=estimated_cost,
            usage=result.usage,
            metadata={
                "case_id": case_id,
                "image_path": str(test_case.gold_truth_image_path),
                "size": app_config.gold_truth_image_generation.size,
                "quality": app_config.gold_truth_image_generation.quality,
                "output_format": app_config.gold_truth_image_generation.output_format,
            },
        )

    metadata = {
        "run_id": run_id,
        "dry_run": dry_run,
        "case_id": case_id,
        "prompt_version": prompt_version_name,
        "model": app_config.gold_truth_image_generation.model,
        "size": app_config.gold_truth_image_generation.size,
        "quality": app_config.gold_truth_image_generation.quality,
        "output_format": app_config.gold_truth_image_generation.output_format,
        "image_path": str(test_case.gold_truth_image_path),
        "prompt_path": str(prompt_path),
        "usage": result.usage.to_dict(),
        "estimated_cost_usd": estimated_cost,
        "revised_prompt": result.revised_prompt,
        "ledger_after_call": _ledger_summary(ledger_after),
        "created_at": utc_now(),
    }
    test_case.gold_truth_metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    log_generation(
        {
            "run_id": run_id,
            "dry_run": dry_run,
            "profile": "gold_truth_image_generation",
            "prompt_version": prompt_version_name,
            "role": "gold_truth_image_generation",
            "model": app_config.gold_truth_image_generation.model,
            "prompt": rendered_prompt,
            "response_text": json.dumps(metadata, sort_keys=True),
            "response_json": metadata,
            "response_id": result.response_id,
            "usage": result.usage.to_dict(),
            "ledger_after_call": _ledger_summary(ledger_after),
            "created_at": utc_now(),
        }
    )
    print(f"Gold-truth image: {test_case.gold_truth_image_path}")
    return metadata


def load_test_case(case_id: str) -> TestCase:
    case_dir = PROJECT_ROOT / "test_cases" / case_id
    if not case_dir.exists():
        raise ValueError(f"Test case '{case_id}' does not exist at {case_dir}")

    scene_path = case_dir / "scene.txt"
    image_path = case_dir / "image_url.txt"
    metadata_path = case_dir / "expected_metadata.json"
    for path in (scene_path, image_path, metadata_path):
        if not path.exists():
            raise ValueError(f"Missing required test-case file: {path}")

    return TestCase(
        case_id=case_id,
        scene_text=scene_path.read_text(encoding="utf-8").strip(),
        image_url=image_path.read_text(encoding="utf-8").strip(),
        expected_metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        case_dir=case_dir,
        gold_truth_image_path=case_dir / "gold_truth.png",
        gold_truth_metadata_path=case_dir / "gold_truth_metadata.json",
    )


def _run_critic_with_one_retry(
    *,
    client: SceneModelClient,
    app_config: AppConfig,
    ledger: CostLedger,
    dry_run: bool,
    run_id: str,
    profile_name: str,
    prompt_version: str,
    prompt_set: dict[str, str],
    prompt: str,
    image_url: str,
    candidate_image_reference: str,
    gold_truth_image_reference: str,
    critic_config: Any,
) -> CriticResult:
    rendered = render_template(
        prompt_set["critic_continuity_supervisor"],
        {
            "text_to_image_prompt": prompt,
            "image_url": image_url,
            "gold_truth_image_reference": "attached gold-truth reference image",
        },
    )

    for attempt in range(2):
        prompt_to_send = rendered
        if attempt:
            prompt_to_send += (
                "\n\nRetry instruction: your prior response failed JSON validation. "
                "Return only the exact required JSON object. Keep it concise. "
                "Do not include reasoning, markdown, or explanatory prose."
            )
        _assert_can_call(ledger=ledger, dry_run=dry_run)
        result = client.critique_image(
            rendered_prompt=prompt_to_send,
            image_url=candidate_image_reference,
            gold_truth_image_reference=gold_truth_image_reference,
            role_config=critic_config,
        )
        _record_call(
            app_config=app_config,
            ledger=ledger,
            dry_run=dry_run,
            run_id=run_id,
            profile_name=profile_name,
            prompt_version=prompt_version,
            role="critic",
            model=critic_config.model,
            rendered_prompt=prompt_to_send,
            result=result,
        )
        try:
            if not result.text.strip():
                raise ValueError(
                    "Critic returned an empty response. This usually means the "
                    "model exhausted max_output_tokens before producing JSON."
                )
            return CriticResult.from_dict(extract_json_object(result.text))
        except Exception:
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


def _generate_candidate_image_for_prompt(
    *,
    client: SceneModelClient,
    app_config: AppConfig,
    ledger: CostLedger,
    dry_run: bool,
    run_id: str,
    profile_name: str,
    prompt_version: str,
    iteration_number: int,
    prompt: str,
) -> dict[str, Any]:
    image_dir = PROJECT_ROOT / "logs" / "generated_images" / run_id
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"iteration_{iteration_number:02d}_candidate.png"
    prompt_path = image_dir / f"iteration_{iteration_number:02d}_candidate_prompt.txt"
    metadata_path = image_dir / f"iteration_{iteration_number:02d}_candidate_metadata.json"

    _assert_can_call(ledger=ledger, dry_run=dry_run)
    result = client.generate_candidate_image(
        rendered_prompt=prompt,
        image_config=app_config.candidate_image_generation,
    )

    image_path.write_bytes(result.image_bytes)
    prompt_path.write_text(prompt, encoding="utf-8")

    estimated_cost = (
        0.0 if dry_run else app_config.candidate_image_generation.cost_per_image_usd
    )
    ledger_after = None
    if not dry_run:
        ledger_after = ledger.record_fixed_cost(
            run_id=run_id,
            profile=profile_name,
            prompt_version=prompt_version,
            role="candidate_image_generation",
            model=app_config.candidate_image_generation.model,
            estimated_cost_usd=estimated_cost,
            usage=result.usage,
            metadata={
                "iteration": iteration_number,
                "image_path": str(image_path),
                "size": app_config.candidate_image_generation.size,
                "quality": app_config.candidate_image_generation.quality,
                "output_format": app_config.candidate_image_generation.output_format,
            },
        )

    metadata = {
        "run_id": run_id,
        "dry_run": dry_run,
        "iteration": iteration_number,
        "prompt_version": prompt_version,
        "model": app_config.candidate_image_generation.model,
        "size": app_config.candidate_image_generation.size,
        "quality": app_config.candidate_image_generation.quality,
        "output_format": app_config.candidate_image_generation.output_format,
        "image_path": str(image_path),
        "prompt_path": str(prompt_path),
        "usage": result.usage.to_dict(),
        "estimated_cost_usd": estimated_cost,
        "revised_prompt": result.revised_prompt,
        "ledger_after_call": _ledger_summary(ledger_after),
        "created_at": utc_now(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log_generation(
        {
            "run_id": run_id,
            "dry_run": dry_run,
            "profile": profile_name,
            "prompt_version": prompt_version,
            "role": "candidate_image_generation",
            "model": app_config.candidate_image_generation.model,
            "prompt": prompt,
            "response_text": json.dumps(metadata, sort_keys=True),
            "response_json": metadata,
            "response_id": result.response_id,
            "usage": result.usage.to_dict(),
            "ledger_after_call": _ledger_summary(ledger_after),
            "created_at": utc_now(),
        }
    )
    return metadata


def _record_call(
    *,
    app_config: AppConfig,
    ledger: CostLedger,
    dry_run: bool,
    run_id: str,
    profile_name: str,
    prompt_version: str,
    role: str,
    model: str,
    rendered_prompt: str,
    result: ModelCallResult,
) -> None:
    if not dry_run and not ledger.can_spend():
        raise RuntimeError("Budget cap reached before model call could be recorded.")

    ledger_after = None
    if not dry_run:
        ledger_after = ledger.record_call(
            run_id=run_id,
            profile=profile_name,
            prompt_version=prompt_version,
            role=role,
            model=model,
            usage=result.usage,
        )

    log_generation(
        {
            "run_id": run_id,
            "dry_run": dry_run,
            "profile": profile_name,
            "prompt_version": prompt_version,
            "role": role,
            "model": model,
            "prompt": rendered_prompt,
            "response_text": result.text,
            "response_json": _try_extract_json_object(result.text),
            "response_id": result.response_id,
            "usage": result.usage.to_dict(),
            "ledger_after_call": _ledger_summary(ledger_after),
            "created_at": utc_now(),
        }
    )


def _assert_can_call(*, ledger: CostLedger, dry_run: bool) -> None:
    if not dry_run and not ledger.can_spend():
        raise RuntimeError("Budget cap reached before making another model call.")


def _print_json(label: str, value: dict[str, Any]) -> None:
    print(f"{label}:")
    print(json.dumps(value, indent=2, sort_keys=True))


def _try_extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        return extract_json_object(text)
    except Exception:
        return None


def _gold_truth_reference(test_case: TestCase, *, allow_missing: bool) -> str:
    if test_case.gold_truth_image_path.exists():
        return image_file_to_data_uri(test_case.gold_truth_image_path)
    if allow_missing:
        return (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAF"
            "gwJ/lw0XAAAAAABJRU5ErkJggg=="
        )
    raise RuntimeError(
        "Gold-truth image is missing for this test case. Generate it first:\n"
        f"python -m scene_evaluator.cli generate-gold --case {test_case.case_id}"
    )


def _ledger_summary(ledger: dict[str, Any] | None) -> dict[str, Any] | None:
    if ledger is None:
        return None
    return {
        "budget_usd": ledger["budget_usd"],
        "total_estimated_cost_usd": ledger["total_estimated_cost_usd"],
        "total_input_tokens": ledger["total_input_tokens"],
        "total_cached_input_tokens": ledger["total_cached_input_tokens"],
        "total_output_tokens": ledger["total_output_tokens"],
        "call_count": ledger["call_count"],
        "runs": ledger["runs"],
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
