import argparse
import datasets
import evals.config
import evals.records
import filelock
import importlib.metadata
import lm_eval
import lm_eval.tasks
import pathlib
import uuid
import yaml


def _Snapshot(*, source: evals.config.Dataset, name: str | None, root: pathlib.Path, identity: dict,
              destination: pathlib.Path) -> dict:
    snapshot_id = destination.name
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = evals.records.ReadJSON(path=manifest_path)
        if manifest["identity"] != identity or manifest["id"] != snapshot_id:
            raise ValueError(f"Dataset snapshot identity mismatch: {destination}")
        for filename, digest in manifest["checksums"].items():
            if evals.records.FileDigest(path=destination / filename) != digest:
                raise ValueError(f"Dataset snapshot checksum mismatch: {destination / filename}")
        return {"path": str(destination), **manifest}

    temporary = destination.with_name(f"{snapshot_id}.{uuid.uuid4().hex}.partial")
    temporary.mkdir(parents=True)
    cache_dir = str(root / ".__pycache__" / "datasets")
    if source.local_files is not None:
        data = datasets.load_dataset(
            "json", data_files={split: str(root / filename) for split, filename in source.local_files.items()},
            cache_dir=cache_dir,
        )
    else:
        data = datasets.load_dataset(
            source.repository, name=name, revision=source.revision, cache_dir=cache_dir,
        )
    splits = {}
    for index, (split, rows) in enumerate(sorted(data.items())):
        filename = f"{index}.parquet"
        rows.to_parquet(temporary / filename)
        splits[split] = {"file": filename, "rows": len(rows)}
    manifest = {
        "checksums": {item["file"]: evals.records.FileDigest(path=temporary / item["file"]) for item in splits.values()},
        "id": snapshot_id,
        "identity": identity,
        "splits": splits,
    }
    evals.records.WriteJSON(path=temporary / "manifest.json", value=manifest)
    temporary.rename(destination)
    return {"path": str(destination), **manifest}


def Main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workspace", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not workspace.is_relative_to(pathlib.Path.cwd()):
        parser.error("Evaluation workspace must be inside the repository")
    evaluation = evals.config.Evaluation.model_validate_json(args.config.read_text())
    Prepare(evaluation=evaluation, root=workspace, output=args.output)


def Prepare(*, evaluation: evals.config.Evaluation, root: pathlib.Path, output: pathlib.Path) -> dict:
    if importlib.metadata.version("lm-eval") != evals.config.HARNESS_VERSION:
        raise ValueError(f"Install evals/requirements.lock; lm-eval {evals.config.HARNESS_VERSION} is required")
    manager = lm_eval.tasks.TaskManager()
    task_root = pathlib.Path(lm_eval.__file__).parent / "tasks"
    paths = set()
    for pattern in evaluation.task_files:
        matches = list(task_root.glob(pattern))
        if not matches or any(not path.resolve().is_relative_to(task_root.resolve()) for path in matches):
            raise ValueError(f"Invalid or unmatched harness task pattern: {pattern}")
        paths.update(path.resolve() for path in matches)
    entries = [entry for entry in manager.task_index.values() if entry.yaml_path in paths]
    if len(entries) != len(paths):
        raise ValueError("task_files must select individual harness task definitions")
    include_path = output / "tasks"
    include_path.mkdir(parents=True)
    snapshots = {}
    for entry in entries:
        task_config = entry.cfg
        if task_config["metadata"]["version"] != evaluation.task_version:
            raise ValueError(f"Task version mismatch for {entry.name}")
        if evaluation.dataset.repository is not None and task_config["dataset_path"] != evaluation.dataset.repository:
            raise ValueError(f"Dataset repository mismatch for {entry.name}")
        if evaluation.adapter == "local-chat-completions" and task_config["output_type"] != "generate_until":
            raise ValueError("Chat completions cannot score candidate token probabilities")
        snapshot = Snapshot(source=evaluation.dataset, name=task_config.get("dataset_name"), root=root)
        snapshots[entry.name] = snapshot
        override = {
            "dataset_kwargs": {"data_files": {
                split: str(pathlib.Path(snapshot["path"]) / item["file"])
                for split, item in snapshot["splits"].items()
            }},
            "dataset_name": None,
            "dataset_path": "parquet",
            "include": str(entry.yaml_path),
            "task": entry.name,
        }
        (include_path / f"{entry.name}.yaml").write_text(yaml.safe_dump(override))
    # Loading through the harness preserves its preprocessing, few-shot pools, and group structure.
    resources = root / "eval_data" / "nltk"
    resource_manifest = resources.with_suffix(".json")
    with filelock.FileLock(str(resource_manifest) + ".lock"):
        if resource_manifest.exists() and evals.records.ReadJSON(path=resource_manifest) != evals.records.Files(directory=resources):
            raise ValueError("NLTK resource checksum mismatch")
        loaded = lm_eval.tasks.TaskManager(include_path=include_path).load(evaluation.task)
        resource_hashes = evals.records.Files(directory=resources)
        if resource_hashes:
            evals.records.WriteJSON(path=resource_manifest, value=resource_hashes)
    if set(loaded["tasks"]) != set(snapshots):
        raise ValueError("task_files must cover exactly the leaf tasks selected by task")
    selected = {}
    for name, task in loaded["tasks"].items():
        count = len(task.eval_docs)
        selected[name] = list(range(count if evaluation.limit is None else min(count, evaluation.limit)))
    prepared = {"include_path": str(include_path), "selected_examples": selected, "snapshots": snapshots,
                "scoring_resources": resource_hashes}
    evals.records.WriteJSON(path=output / "prepared.json", value=prepared)
    return prepared


def Snapshot(*, source: evals.config.Dataset, name: str | None, root: pathlib.Path) -> dict:
    identity = {"source": source.model_dump(), "name": name, "datasets_version": importlib.metadata.version("datasets")}
    if source.local_files is not None:
        if any(not (root / filename).resolve().is_relative_to(root) for filename in source.local_files.values()):
            raise ValueError("Local dataset files must be inside the repository")
        identity["files"] = {
            split: evals.records.FileDigest(path=root / filename) for split, filename in source.local_files.items()
        }
    snapshot_id = evals.records.Digest(value=identity)
    destination = root / "eval_data" / snapshot_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(destination) + ".lock"):
        return _Snapshot(source=source, name=name, root=root, identity=identity, destination=destination)


if __name__ == "__main__":
    Main()
