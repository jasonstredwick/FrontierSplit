import argparse
import evals.records
import evals.runner
import json
import pathlib


def Main() -> int:
    parser = argparse.ArgumentParser(description="Run LM Evaluation Harness against a configured FrontierSplit API target.")
    parser.add_argument("--target", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--evals", nargs="+")
    selection.add_argument("--suite")
    parser.add_argument("--config-root", type=pathlib.Path, default=pathlib.Path("configs"))
    parser.add_argument("--limit", type=int, help="Use the first N questions of each task (including each MMLU subject).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and display configurations without network access.")
    mode.add_argument("--prepare-only", action="store_true", help="Save datasets and harness commands without contacting the server.")
    args = parser.parse_args()
    root = pathlib.Path.cwd()
    if not args.config_root.resolve().is_relative_to(root):
        parser.error("--config-root must be inside the repository")
    try:
        plan = evals.runner.LoadPlan(
            config_root=args.config_root,
            target_id=args.target,
            evaluations=args.evals or [],
            suite=args.suite,
            limit=args.limit,
        )
        if args.dry_run:
            print(json.dumps(plan.ToDict(), indent=2))
            return 0
        output = evals.runner.Run(root=root, plan=plan, prepare_only=args.prepare_only)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    status = evals.records.ReadJSON(path=output / "manifest.json")["status"]
    return 0 if status in {"completed", "prepared"} else 130 if status == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(Main())
