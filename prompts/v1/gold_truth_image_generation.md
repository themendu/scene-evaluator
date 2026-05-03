# Gold Truth Image Generation Prompt v1

You are generating the gold-truth reference image for a scene evaluator test case.

This image will be treated as the 100% reference target for critic scoring. Make
the result as faithful as possible to the scene and continuity metadata.

## Raw Scene Text

```text
{{ scene_text }}
```

## Character Metadata

```json
{{ character_metadata }}
```

## Image Requirements

Create a single high-quality reference image that clearly shows:

1. The exact named character or characters required by the scene.
2. The required wardrobe from the scene and metadata.
3. The required prop or props from the scene and metadata.
4. The location, time of day, weather, lighting, and physical state.
5. A composition where all required continuity elements are easy to inspect.

Do not add extra named characters, unsupported props, unsupported wardrobe, or a
different setting.

For this scene, prioritize verifiable continuity over abstract atmosphere.
