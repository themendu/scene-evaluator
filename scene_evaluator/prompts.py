from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scene_evaluator.config import PROJECT_ROOT


PROMPT_VERSIONS_PATH = PROJECT_ROOT / "prompts" / "prompt_versions.json"


def load_prompt_set(version: str | None = None) -> dict[str, str]:
    data = json.loads(PROMPT_VERSIONS_PATH.read_text(encoding="utf-8"))
    selected = version or data["active"]
    try:
        version_data = data["versions"][selected]
    except KeyError as exc:
        available = ", ".join(sorted(data["versions"]))
        raise ValueError(
            f"Unknown prompt version '{selected}'. Available versions: {available}"
        ) from exc

    prompts: dict[str, str] = {"version": selected}
    for key, relative_path in version_data.items():
        if key == "status":
            continue
        prompts[key] = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return prompts


def render_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, indent=2, sort_keys=True)
        rendered = rendered.replace("{{ " + key + " }}", value)
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def extract_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(clean[start : end + 1])

