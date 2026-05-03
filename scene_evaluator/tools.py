from __future__ import annotations


def get_character_metadata(character_name: str) -> dict[str, object]:
    metadata = {
        "Kael": {
            "age": 30,
            "wardrobe": "formal suit",
            "prop": "brass compass",
        }
    }
    try:
        return metadata[character_name]
    except KeyError as exc:
        raise ValueError(f"No mock metadata exists for character '{character_name}'") from exc


OPENAI_TOOL_SCHEMA = {
    "type": "function",
    "name": "get_character_metadata",
    "description": "Return hardcoded continuity metadata for a named character.",
    "parameters": {
        "type": "object",
        "properties": {
            "character_name": {
                "type": "string",
                "description": "The character name to look up.",
            }
        },
        "required": ["character_name"],
        "additionalProperties": False,
    },
}

