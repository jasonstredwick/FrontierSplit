import contextlib
import evals.config
import evals.records
import evals.runner
import json
import os
import pathlib
import pytest
import socket
import subprocess
import sys
import tempfile
import tests.backend
import tests.helpers
import threading
import time
import uvicorn

from frontiersplit.openai_api import generation as fs_generation
from frontiersplit.openai_api import server as fs_server


def ArtifactState(*, root: pathlib.Path) -> dict:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for directory in [root / "eval_data", root / "eval_results"]
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(name="workspace")
def Workspace():
    root = pathlib.Path.cwd()
    original = ArtifactState(root=root)
    (root / "scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "scratch", prefix="eval-test-") as directory:
        try:
            yield pathlib.Path(directory)
        finally:
            assert ArtifactState(root=root) == original


def WriteConfig(*, root: pathlib.Path, kind: str, value: dict) -> None:
    evals.records.WriteJSON(path=root / "configs" / kind / f"{value['id']}.json", value={
        "display_name": value["id"], "schema_version": 1, **value,
    })


@contextlib.contextmanager
def Server(*, workspace: pathlib.Path, api_key: str | None = None):
    bundle = tests.helpers.BuildTestTokenizer()
    bundle.tokenizer.save_pretrained(workspace / "tokenizer")
    backend = tests.backend.ScriptedBackend(
        default_tokens=[bundle.world_id, bundle.eos_id], eos_token_ids=frozenset({bundle.eos_id}),
        max_model_len=2048, model_id="fixture", vocab_size=bundle.vocab_size,
    )
    app = fs_server.BuildApp(api_key=api_key, engine=fs_generation.GenerationAdapter(backend=backend, tokenizer=bundle.tokenizer))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


def Configure(*, workspace: pathlib.Path, url: str, api_key_env: str | None = None) -> None:
    WriteConfig(root=workspace, kind="backends", value={
        "id": "fixture", "model": "fixture", "model_revision": "scripted", "precision": "fp16",
        "tokenizer": str(workspace / "tokenizer"), "tokenizer_revision": "0" * 40,
        "max_model_len": 2048, "software": {"implementation": "tests.backend.ScriptedBackend"},
    })
    WriteConfig(root=workspace, kind="deployments", value={
        "id": "local", "provider": "local", "location": "test", "hardware": {"device": "cpu"},
        "software": {}, "launch_instructions": "pytest",
    })
    WriteConfig(root=workspace, kind="targets", value={
        "id": "fixture", "backend": "fixture", "deployment": "local", "api_base_url": url, "served_model": "fixture",
        "api_key_env": api_key_env,
    })
    (workspace / "answers.jsonl").write_text("\n".join(json.dumps({
        "question": f"Hello {index}", "answer": "#### 2",
    }) for index in range(3)))
    (workspace / "choices.jsonl").write_text("\n".join(json.dumps({
        "id": str(index), "question": "Hello", "answerKey": "A",
        "choices": {"text": ["world", "stop"], "label": ["A", "B"]},
    }) for index in range(3)))
    for name in ["completion", "chat", "choices"]:
        choices = name == "choices"
        WriteConfig(root=workspace, kind="evals", value={
            "id": name, "task": "arc_challenge" if choices else "gsm8k",
            "task_files": ["arc/arc_challenge.yaml" if choices else "gsm8k/gsm8k.yaml"],
            "task_version": 1.0 if choices else 3.0, "num_fewshot": 0, "max_gen_toks": 16,
            "adapter": "local-chat-completions" if name == "chat" else "local-completions",
            "apply_chat_template": name == "chat",
            "dataset": {"local_files": {
                split: str(workspace / ("choices.jsonl" if choices else "answers.jsonl"))
                for split in ["test", "train", "validation"]
            }},
        })
    WriteConfig(root=workspace, kind="suites", value={"id": "all", "evaluations": ["choices", "completion", "chat"]})


@pytest.fixture(name="credentials")
def Credentials():
    name = "FRONTIERSPLIT_EVAL_TEST_KEY"
    original = os.environ.get(name)
    secret = "fixture-credential-not-for-publication"
    os.environ[name] = secret
    try:
        yield name, secret
    finally:
        if original is None:
            del os.environ[name]
        else:
            os.environ[name] = original


def test_harness_through_frontiersplit_api(*, workspace, credentials):
    name, secret = credentials
    with Server(workspace=workspace, api_key=secret) as url:
        Configure(workspace=workspace, url=url, api_key_env=name)
        plan = evals.runner.LoadPlan(config_root=workspace / "configs", target_id="fixture", evaluations=[], suite="all", limit=2)
        output = evals.runner.Run(root=pathlib.Path.cwd(), workspace=workspace, plan=plan)
        assert output.is_relative_to(workspace / "eval_results")
        manifest = evals.records.ReadJSON(path=output / "manifest.json")
        assert manifest["status"] == "completed", output
        assert set(manifest["evaluations"].values()) == {"completed"}
        for evaluation in plan.evaluations:
            path = output / "evaluations" / evaluation.id
            result = evals.records.ReadJSON(path=path / "manifest.json")
            assert result["expected_samples"] == result["completed_samples"] == {evaluation.task: 2}
            assert result["selected_examples"] == {evaluation.task: [0, 1]}
            for snapshot in result["snapshots"].values():
                assert pathlib.Path(snapshot["path"]).is_relative_to(workspace / "eval_data")
            assert (path / "replay.sh").is_file()
            if evaluation.task == "gsm8k":
                assert result["metrics"]["gsm8k"]["exact_match,strict-match"] == 0
        for path in output.rglob("*"):
            if path.is_file():
                assert secret not in path.read_text()


@pytest.mark.parametrize("target", ["mistral-base", "mixtral-base"])
def test_module_cli_dry_run(*, target):
    result = subprocess.run(
        [sys.executable, "-X", "pycache_prefix=.__pycache__", "-m", "evals",
         "--target", target, "--suite", "quality", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["target"]["id"] == target
    assert [evaluation["id"] for evaluation in plan["evaluations"]] == plan["suite"]["evaluations"]


def test_preparation_failure_preserves_other_evaluations(*, workspace):
    Configure(workspace=workspace, url="http://127.0.0.1:1/v1")
    path = workspace / "configs/evals/choices.json"
    config = evals.records.ReadJSON(path=path)
    config["task_version"] = 999
    evals.records.WriteJSON(path=path, value=config)
    plan = evals.runner.LoadPlan(config_root=workspace / "configs", target_id="fixture", evaluations=["choices", "completion"],
                              suite=None, limit=1)
    output = evals.runner.Run(root=pathlib.Path.cwd(), workspace=workspace, plan=plan, prepare_only=True)
    manifest = evals.records.ReadJSON(path=output / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["evaluations"] == {"choices": "failed", "completion": "prepared"}
    second = evals.runner.Run(root=pathlib.Path.cwd(), workspace=workspace, plan=plan, prepare_only=True)
    assert second != output
    assert evals.records.ReadJSON(path=output / "manifest.json") == manifest


def test_snapshot_corruption_is_rejected(*, workspace):
    Configure(workspace=workspace, url="http://127.0.0.1:1/v1")
    plan = evals.runner.LoadPlan(config_root=workspace / "configs", target_id="fixture", evaluations=["completion"], suite=None,
                              limit=1)
    first = evals.runner.Run(root=pathlib.Path.cwd(), workspace=workspace, plan=plan, prepare_only=True)
    prepared = evals.records.ReadJSON(path=first / "evaluations/completion/prepared.json")
    snapshot = prepared["snapshots"]["gsm8k"]
    data_file = pathlib.Path(snapshot["path"]) / snapshot["splits"]["test"]["file"]
    data_file.write_bytes(b"corrupted")
    second = evals.runner.Run(root=pathlib.Path.cwd(), workspace=workspace, plan=plan, prepare_only=True)
    assert evals.records.ReadJSON(path=second / "manifest.json")["status"] == "failed"
    assert "checksum mismatch" in (second / "evaluations/completion/logs/prepare.log").read_text()


@pytest.mark.parametrize("changes", [
    {"id": "../bad"}, {"schema_version": 2}, {"api_base_url": "http://user:secret@localhost/v1"},
    {"api_base_url": "http://localhost/v1?api_key=secret"}, {"unexpected": True},
])
def test_target_validation(*, changes):
    with pytest.raises(ValueError):
        evals.config.Target.model_validate({"id": "target", "display_name": "Target", "backend": "backend",
                                        "deployment": "local", "api_base_url": "http://localhost/v1",
                                        "served_model": "model", **changes})


def test_wrong_model_is_rejected_before_evaluation(*, workspace):
    with Server(workspace=workspace) as url:
        Configure(workspace=workspace, url=url)
        path = workspace / "configs/targets/fixture.json"
        target = evals.records.ReadJSON(path=path)
        target["served_model"] = "wrong"
        evals.records.WriteJSON(path=path, value=target)
        plan = evals.runner.LoadPlan(config_root=workspace / "configs", target_id="fixture", evaluations=["completion"], suite=None)
        output = evals.runner.Run(root=pathlib.Path.cwd(), workspace=workspace, plan=plan)
        manifest = evals.records.ReadJSON(path=output / "manifest.json")
        assert manifest["status"] == "failed"
        assert "does not advertise" in manifest["error"]
        assert manifest["evaluations"] == {"completion": "pending"}
