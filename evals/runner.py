import dataclasses
import evals.config
import evals.harness
import evals.records
import os
import pathlib
import requests
import uuid


@dataclasses.dataclass(kw_only=True)
class Plan:
    backend: evals.config.Backend
    deployment: evals.config.Deployment
    evaluations: list[evals.config.Evaluation]
    suite: evals.config.Suite | None
    target: evals.config.Target

    def ToDict(self) -> dict:
        return {
            "backend": self.backend.model_dump(),
            "deployment": self.deployment.model_dump(),
            "evaluations": [evaluation.model_dump() for evaluation in self.evaluations],
            "suite": self.suite.model_dump() if self.suite else None,
            "target": self.target.model_dump(),
        }


def LoadPlan(*, config_root: pathlib.Path, target_id: str, evaluations: list[str], suite: str | None,
             limit: int | None = None) -> Plan:
    if bool(evaluations) == bool(suite):
        raise ValueError("Select evaluations or a suite")
    target = evals.config.Load(root=config_root, kind="targets", name=target_id, schema=evals.config.Target)
    backend = evals.config.Load(root=config_root, kind="backends", name=target.backend, schema=evals.config.Backend)
    deployment = evals.config.Load(root=config_root, kind="deployments", name=target.deployment, schema=evals.config.Deployment)
    selected_suite = evals.config.Load(root=config_root, kind="suites", name=suite, schema=evals.config.Suite) if suite else None
    names = selected_suite.evaluations if selected_suite else evaluations
    if len(set(names)) != len(names):
        raise ValueError("Evaluations must be distinct")
    selected = []
    for name in names:
        evaluation = evals.config.Load(root=config_root, kind="evals", name=name, schema=evals.config.Evaluation)
        if limit is not None:
            evaluation = evals.config.Evaluation.model_validate({**evaluation.model_dump(), "limit": limit})
        if evaluation.max_gen_toks >= backend.max_model_len:
            raise ValueError(f"{name}: generation budget must leave room for the prompt")
        selected.append(evaluation)
    return Plan(backend=backend, deployment=deployment, evaluations=selected, suite=selected_suite, target=target)


def Probe(*, target: evals.config.Target, backend: evals.config.Backend) -> dict:
    headers = {"Authorization": f"Bearer {os.environ[target.api_key_env]}"} if target.api_key_env else {}
    health = requests.get(target.api_base_url.removesuffix("/v1") + "/health", timeout=30)
    health.raise_for_status()
    models = requests.get(target.api_base_url + "/models", headers=headers, timeout=30)
    models.raise_for_status()
    matches = [model for model in models.json()["data"] if model["id"] == target.served_model]
    if len(matches) != 1:
        raise ValueError(f"Server does not advertise the selected model: {target.served_model}")
    if matches[0]["max_model_len"] != backend.max_model_len:
        raise ValueError("Server context limit differs from the backend configuration")
    return {"health_status": health.status_code, "model": matches[0]}


def Run(*, root: pathlib.Path, plan: Plan, prepare_only: bool = False,
        workspace: pathlib.Path | None = None) -> pathlib.Path:
    root = root.resolve()
    workspace = root if workspace is None else workspace.resolve()
    if not workspace.is_relative_to(root):
        raise ValueError("Evaluation workspace must be inside the repository")
    evals.harness.CheckInstallation(root=root)
    environment = evals.harness.Environment(
        root=root, workspace=workspace, target=plan.target, require_credentials=not prepare_only,
    )
    run_id = evals.records.Timestamp().replace(":", "").replace("+0000", "Z") + "-" + uuid.uuid4().hex[:8]
    destination = workspace / "eval_results" / plan.target.id / run_id
    destination.mkdir(parents=True)
    configuration = plan.ToDict()
    manifest = {
        "client": evals.records.Runtime(),
        "configuration": configuration,
        "configuration_sha256": evals.records.Digest(value=configuration),
        "evaluations": {evaluation.id: "pending" for evaluation in plan.evaluations},
        "id": run_id,
        "mode": "prepare" if prepare_only else "evaluate",
        "source": evals.records.SourceState(root=root),
        "started": evals.records.Timestamp(),
        "status": "running",
    }
    path = destination / "manifest.json"
    evals.records.WriteJSON(path=path, value=manifest)
    try:
        if not prepare_only:
            manifest["observed_server"] = Probe(target=plan.target, backend=plan.backend)
        for evaluation in plan.evaluations:
            print(f"{evaluation.id}: preparing", flush=True)
            output = destination / "evaluations" / evaluation.id
            output.mkdir(parents=True)
            state = {"configuration": evaluation.model_dump(), "started": evals.records.Timestamp(), "status": "preparing"}
            evals.records.WriteJSON(path=output / "manifest.json", value=state)
            evals.records.WriteJSON(path=output / "evaluation.json", value=evaluation.model_dump())
            try:
                command = evals.harness.Command(module="evals.prepare", arguments=[
                    "--config", str(output / "evaluation.json"), "--output", str(output),
                    "--workspace", str(workspace),
                ])
                code = evals.harness.Execute(command=command, root=root, environment=environment, log=output / "logs/prepare.log")
                state["exit_status"] = code
                if code != 0:
                    raise RuntimeError(f"Dataset preparation failed; see {output / 'logs/prepare.log'}")
                prepared = evals.records.ReadJSON(path=output / "prepared.json")
                state.update(prepared)
                state["expected_samples"] = {name: len(ids) for name, ids in prepared["selected_examples"].items()}
                harness_config = evals.harness.HarnessConfig(
                    evaluation=evaluation, backend=plan.backend, target=plan.target, prepared=prepared, output=output,
                )
                evals.records.WriteJSON(path=output / "harness.json", value=harness_config)
                command = evals.harness.Command(module="lm_eval", arguments=["run", "--config", str(output / "harness.json")])
                state["command"] = command
                evals.harness.Replay(
                    command=command, root=root, workspace=workspace, target=plan.target, path=output / "replay.sh",
                )
                state["status"] = "prepared" if prepare_only else "running"
                evals.records.WriteJSON(path=output / "manifest.json", value=state)
                if not prepare_only:
                    print(f"{evaluation.id}: running", flush=True)
                    code = evals.harness.Execute(
                        command=command, root=root, environment=environment, log=output / "logs/harness.log",
                    )
                    state["exit_status"] = code
                    if code != 0:
                        raise RuntimeError(f"Harness failed; see {output / 'logs/harness.log'}")
                    state.update(evals.harness.Results(output=output, expected=prepared["selected_examples"]))
                    state["status"] = "completed"
            except KeyboardInterrupt:
                state["status"] = "interrupted"
                raise
            except Exception as error:
                state.update({"status": "failed", "error": str(error)})
                print(f"{evaluation.id}: {error}", flush=True)
            finally:
                state["finished"] = evals.records.Timestamp()
                state["file_checksums"] = {name: digest for name, digest in evals.records.Files(directory=output).items()
                                           if name != "manifest.json"}
                evals.records.WriteJSON(path=output / "manifest.json", value=state)
                manifest["evaluations"][evaluation.id] = state["status"]
                evals.records.WriteJSON(path=path, value=manifest)
        manifest["status"] = "failed" if "failed" in manifest["evaluations"].values() else (
            "prepared" if prepare_only else "completed"
        )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error)})
        print(str(error), flush=True)
    finally:
        manifest["finished"] = evals.records.Timestamp()
        evals.records.WriteJSON(path=path, value=manifest)
    print(f"{manifest['status']}: {destination}", flush=True)
    return destination
