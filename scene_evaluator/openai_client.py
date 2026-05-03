from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scene_evaluator.config import GoldTruthImageConfig
from scene_evaluator.config import ModelRoleConfig
from scene_evaluator.schemas import Usage
from scene_evaluator.tools import OPENAI_TOOL_SCHEMA, get_character_metadata


CRITIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_match_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "missing_elements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hallucinated_elements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "actionable_feedback": {"type": "string"},
    },
    "required": [
        "overall_match_score",
        "missing_elements",
        "hallucinated_elements",
        "actionable_feedback",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ModelCallResult:
    text: str
    usage: Usage
    response_id: str | None = None


@dataclass(frozen=True)
class ImageGenerationResult:
    image_bytes: bytes
    usage: Usage
    response_id: str | None = None
    revised_prompt: str | None = None


class OpenAIResponsesClient:
    def __init__(self, timeout_seconds: float = 60.0, max_retries: int = 2) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.client = OpenAI(timeout=timeout_seconds)
        self.max_retries = max_retries

    def generate_initial_prompt(
        self,
        *,
        rendered_prompt: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        return self._complete_with_tool_loop(
            rendered_prompt=rendered_prompt,
            role_config=role_config,
        )

    def revise_prompt(
        self,
        *,
        rendered_prompt: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        return self._complete_text(
            rendered_prompt=rendered_prompt,
            role_config=role_config,
        )

    def critique_image(
        self,
        *,
        rendered_prompt: str,
        image_url: str,
        gold_truth_image_reference: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        return self._complete_text(
            rendered_prompt=rendered_prompt,
            role_config=role_config,
            image_urls=[image_url, gold_truth_image_reference],
            json_schema=CRITIC_JSON_SCHEMA,
        )

    def generate_gold_truth_image(
        self,
        *,
        rendered_prompt: str,
        image_config: GoldTruthImageConfig,
    ) -> ImageGenerationResult:
        return self._generate_image(rendered_prompt=rendered_prompt, image_config=image_config)

    def generate_candidate_image(
        self,
        *,
        rendered_prompt: str,
        image_config: GoldTruthImageConfig,
    ) -> ImageGenerationResult:
        return self._generate_image(rendered_prompt=rendered_prompt, image_config=image_config)

    def _generate_image(
        self,
        *,
        rendered_prompt: str,
        image_config: GoldTruthImageConfig,
    ) -> ImageGenerationResult:
        response = self._image_request(
            model=image_config.model,
            prompt=rendered_prompt,
            n=1,
            size=image_config.size,
            quality=image_config.quality,
            output_format=image_config.output_format,
        )
        data = (_attr(response, "data", None) or [])[0]
        image_base64 = _attr(data, "b64_json")
        if not image_base64:
            raise RuntimeError("Image generation response did not contain b64_json data.")

        return ImageGenerationResult(
            image_bytes=base64.b64decode(image_base64),
            usage=_usage_from_response(response),
            response_id=_attr(response, "id"),
            revised_prompt=_attr(data, "revised_prompt"),
        )

    def _complete_with_tool_loop(
        self,
        *,
        rendered_prompt: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        response = self._request(
            model=role_config.model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": rendered_prompt}],
                }
            ],
            tools=[OPENAI_TOOL_SCHEMA],
            max_output_tokens=role_config.max_output_tokens,
            reasoning=_reasoning(role_config),
        )

        total_usage = _usage_from_response(response)
        response_id = _attr(response, "id")
        function_calls = _function_calls(response)

        while function_calls:
            outputs = []
            for call in function_calls:
                if call["name"] != "get_character_metadata":
                    raise RuntimeError(f"Unsupported tool call requested: {call['name']}")
                arguments = json.loads(call["arguments"] or "{}")
                result = get_character_metadata(arguments["character_name"])
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(result),
                    }
                )

            response = self._request(
                model=role_config.model,
                input=outputs,
                previous_response_id=response_id,
                tools=[OPENAI_TOOL_SCHEMA],
                max_output_tokens=role_config.max_output_tokens,
                reasoning=_reasoning(role_config),
            )
            total_usage = _add_usage(total_usage, _usage_from_response(response))
            response_id = _attr(response, "id")
            function_calls = _function_calls(response)

        return ModelCallResult(
            text=_response_text(response),
            usage=total_usage,
            response_id=response_id,
        )

    def _complete_text(
        self,
        *,
        rendered_prompt: str,
        role_config: ModelRoleConfig,
        image_urls: list[str] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ModelCallResult:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": rendered_prompt}]
        for image_url in image_urls or []:
            image_content: dict[str, Any] = {
                "type": "input_image",
                "image_url": image_url,
            }
            if role_config.image_detail:
                image_content["detail"] = role_config.image_detail
            content.append(image_content)

        kwargs: dict[str, Any] = {}
        if json_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "critic_result",
                    "schema": json_schema,
                    "strict": True,
                }
            }

        response = self._request(
            model=role_config.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=role_config.max_output_tokens,
            reasoning=_reasoning(role_config),
            **kwargs,
        )
        return ModelCallResult(
            text=_response_text(response),
            usage=_usage_from_response(response),
            response_id=_attr(response, "id"),
        )

    def _request(self, **kwargs: Any) -> Any:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.responses.create(**_without_none(kwargs))
            except Exception as exc:  # OpenAI SDK exception classes vary by version.
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"OpenAI request failed after retries: {last_error}") from last_error

    def _image_request(self, **kwargs: Any) -> Any:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.images.generate(**_without_none(kwargs))
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(
            f"OpenAI image generation failed after retries: {last_error}"
        ) from last_error


class DryRunClient:
    """Deterministic no-cost client for local validation and tests."""

    def generate_initial_prompt(
        self,
        *,
        rendered_prompt: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        text = json.dumps(
            {
                "character_metadata_used": {
                    "Kael": {
                        "age": 30,
                        "wardrobe": "formal suit",
                        "prop": "brass compass",
                    }
                },
                "text_to_image_prompt": (
                    "Kael, a 30-year-old man in a formal suit, stands alone in a "
                    "nighttime alleyway during a torrential downpour, completely "
                    "soaked beneath a flickering neon sign, holding a vintage brass "
                    "compass clearly visible in one hand."
                ),
            }
        )
        return ModelCallResult(text=text, usage=Usage(input_tokens=350, output_tokens=90))

    def revise_prompt(
        self,
        *,
        rendered_prompt: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        text = json.dumps(
            {
                "revision_summary": "Emphasized the required prop, rain, wardrobe, and neon alley setting.",
                "revised_text_to_image_prompt": (
                    "A single named character, Kael, age 30, in a drenched formal suit "
                    "stands in a dark alley at night under a flickering neon sign. "
                    "Heavy rain is visibly pouring over him, and a vintage brass "
                    "compass is held prominently in his hand."
                ),
            }
        )
        return ModelCallResult(text=text, usage=Usage(input_tokens=420, output_tokens=85))

    def critique_image(
        self,
        *,
        rendered_prompt: str,
        image_url: str,
        gold_truth_image_reference: str,
        role_config: ModelRoleConfig,
    ) -> ModelCallResult:
        text = json.dumps(
            {
                "overall_match_score": 42,
                "missing_elements": [
                    "Kael is not clearly identifiable",
                    "formal suit is not clearly present",
                    "vintage brass compass is missing",
                    "flickering neon sign is missing or unclear",
                ],
                "hallucinated_elements": ["dummy image content does not match the scene"],
                "actionable_feedback": (
                    "Make the next prompt explicitly require one visible man named Kael, "
                    "a soaked formal suit, a prominent brass compass, nighttime alley, "
                    "heavy rain, and flickering neon sign."
                ),
            }
        )
        return ModelCallResult(text=text, usage=Usage(input_tokens=500, output_tokens=120))

    def generate_gold_truth_image(
        self,
        *,
        rendered_prompt: str,
        image_config: GoldTruthImageConfig,
    ) -> ImageGenerationResult:
        return self._dry_image_result("dry-run-gold-truth")

    def generate_candidate_image(
        self,
        *,
        rendered_prompt: str,
        image_config: GoldTruthImageConfig,
    ) -> ImageGenerationResult:
        return self._dry_image_result("dry-run-candidate")

    def _dry_image_result(self, response_id: str) -> ImageGenerationResult:
        # 1x1 transparent PNG, enough for filesystem plumbing tests.
        png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAF"
            "gwJ/lw0XAAAAAABJRU5ErkJggg=="
        )
        return ImageGenerationResult(
            image_bytes=base64.b64decode(png_base64),
            usage=Usage(input_tokens=120, output_tokens=0),
            response_id=response_id,
            revised_prompt=None,
        )


def image_file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _reasoning(role_config: ModelRoleConfig) -> dict[str, str] | None:
    if not role_config.reasoning_format:
        return None
    return {"effort": role_config.reasoning_format}


def _without_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _response_text(response: Any) -> str:
    output_text = _attr(response, "output_text")
    if output_text:
        return output_text

    texts: list[str] = []
    for item in _attr(response, "output", []) or []:
        for content in _attr(item, "content", []) or []:
            if _attr(content, "type") in {"output_text", "text"}:
                text = _attr(content, "text")
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def _function_calls(response: Any) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for item in _attr(response, "output", []) or []:
        if _attr(item, "type") == "function_call":
            calls.append(
                {
                    "name": _attr(item, "name"),
                    "call_id": _attr(item, "call_id"),
                    "arguments": _attr(item, "arguments") or "{}",
                }
            )
    return calls


def _usage_from_response(response: Any) -> Usage:
    usage = _attr(response, "usage", {}) or {}
    input_tokens = int(_get(usage, "input_tokens", _get(usage, "prompt_tokens", 0)) or 0)
    output_tokens = int(
        _get(usage, "output_tokens", _get(usage, "completion_tokens", 0)) or 0
    )

    details = _get(usage, "input_tokens_details", None) or _get(
        usage, "prompt_tokens_details", {}
    )
    cached_input_tokens = int(_get(details, "cached_tokens", 0) or 0)
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
    )


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _get(value: Any, name: str, default: Any = None) -> Any:
    return _attr(value, name, default)
