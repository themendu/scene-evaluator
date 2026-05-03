from __future__ import annotations

import argparse
import json
import sys

from scene_evaluator.config import load_config
from scene_evaluator.pipeline import generate_gold_truth_image_for_case, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scene-evaluator",
        description="Run the agentic scene evaluator self-correction loop.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one scene test case.")
    run_parser.add_argument("--case", default="alleyway_kael")
    run_parser.add_argument("--profile", default=None)
    run_parser.add_argument("--prompt-version", default=None)
    run_parser.add_argument("--max-iterations", type=int, default=2)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic local responses and make no API calls.",
    )

    gold_parser = subparsers.add_parser(
        "generate-gold",
        help="Generate the gold-truth reference image for a test case.",
    )
    gold_parser.add_argument("--case", default="alleyway_kael")
    gold_parser.add_argument("--prompt-version", default=None)
    gold_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing gold-truth image.",
    )
    gold_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a deterministic placeholder image and make no API calls.",
    )

    profiles_parser = subparsers.add_parser("profiles", help="List model profiles.")
    profiles_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "profiles":
        app_config = load_config()
        data = {
            "active_profile": app_config.active_profile,
            "profiles": {
                name: {
                    "description": profile.description,
                    "generator_model": profile.generator.model,
                    "critic_model": profile.critic.model,
                    "critic_image_detail": profile.critic.image_detail,
                }
                for name, profile in app_config.profiles.items()
            },
        }
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"Active profile: {data['active_profile']}")
            for name, profile_data in data["profiles"].items():
                print(
                    f"- {name}: generator={profile_data['generator_model']}, "
                    f"critic={profile_data['critic_model']}"
                )
        return 0

    if args.command == "run":
        try:
            result = run_pipeline(
                case_id=args.case,
                profile_name=args.profile,
                prompt_version=args.prompt_version,
                max_iterations=args.max_iterations,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print("\nFinal Result")
        print(json.dumps({"run_id": result["run_id"], "final_score": result["final_score"]}, indent=2))
        return 0

    if args.command == "generate-gold":
        try:
            result = generate_gold_truth_image_for_case(
                case_id=args.case,
                prompt_version=args.prompt_version,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print("\nGold Truth Result")
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "image_path": result["image_path"],
                    "model": result["model"],
                    "dry_run": result["dry_run"],
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
