import evals.config
import evals.records
import importlib.metadata
import json
import os
import packaging.requirements
import pathlib
import shlex
import subprocess
import sys


def CheckInstallation(*, root: pathlib.Path) -> None:
    for line in (root / "evals/requirements.lock").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = packaging.requirements.Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Install evals/requirements.lock: missing {requirement.name}") from error
        if version not in requirement.specifier:
            raise ValueError(f"Install evals/requirements.lock: {requirement.name} {version} does not match {requirement.specifier}")


def Command(*, module: str, arguments: list[str]) -> list[str]:
    return [sys.executable, "-X", "pycache_prefix=.__pycache__", "-m", module, *arguments]


def Environment(*, root: pathlib.Path, workspace: pathlib.Path, target: evals.config.Target,
                require_credentials: bool = True) -> dict[str, str]:
    cache = workspace / ".__pycache__"
    environment = os.environ.copy()
    environment.update({
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_MODULES_CACHE": str(cache / "huggingface" / "modules"),
        "NLTK_DATA": str(workspace / "eval_data" / "nltk"),
        "PYTHONPATH": str(root),
        "PYTHONPYCACHEPREFIX": str(cache),
        "TOKENIZERS_PARALLELISM": "false",
        "XDG_CACHE_HOME": str(cache),
    })
    pathlib.Path(environment["NLTK_DATA"]).mkdir(parents=True, exist_ok=True)
    if target.api_key_env and require_credentials and not os.environ.get(target.api_key_env):
        raise ValueError(f"Set the API credential environment variable: {target.api_key_env}")
    environment["OPENAI_API_KEY"] = os.environ.get(target.api_key_env, "") if target.api_key_env else ""
    return environment


def Execute(*, command: list[str], root: pathlib.Path, environment: dict[str, str], log: pathlib.Path) -> int:
    secret = environment["OPENAI_API_KEY"]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            for line in process.stdout:
                output.write(line.replace(secret, "[REDACTED]") if secret else line)
                output.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def HarnessConfig(*, evaluation: evals.config.Evaluation, backend: evals.config.Backend, target: evals.config.Target,
                  prepared: dict, output: pathlib.Path) -> dict:
    chat = evaluation.adapter == "local-chat-completions"
    model_args = {
        "base_url": target.api_base_url + ("/chat/completions" if chat else "/completions"),
        "max_gen_toks": evaluation.max_gen_toks,
        "max_length": backend.max_model_len,
        "max_retries": evaluation.max_retries,
        "model": target.served_model,
        "num_concurrent": evaluation.num_concurrent,
        "seed": evaluation.seeds[2],
        "timeout": evaluation.timeout,
        "tokenized_requests": not chat,
        "tokenizer_backend": "None" if chat else "huggingface",
        "truncate": False,
    }
    if not chat:
        model_args.update({"add_bos_token": backend.add_bos_token, "revision": backend.tokenizer_revision,
                           "tokenizer": backend.tokenizer})
    return {
        "apply_chat_template": (backend.chat_template or True) if evaluation.apply_chat_template else False,
        "batch_size": evaluation.batch_size,
        "cache_requests": {},
        "device": "cpu",
        "fewshot_as_multiturn": evaluation.fewshot_as_multiturn,
        "gen_kwargs": evaluation.generation,
        "include_path": prepared["include_path"],
        "log_samples": True,
        "model": evaluation.adapter,
        "model_args": model_args,
        "num_fewshot": evaluation.num_fewshot,
        "output_path": str(output / "harness"),
        "samples": prepared["selected_examples"],
        "seed": evaluation.seeds,
        "tasks": [evaluation.task],
        "use_cache": None,
    }


def Replay(*, command: list[str], root: pathlib.Path, workspace: pathlib.Path,
           target: evals.config.Target, path: pathlib.Path) -> None:
    environment = Environment(root=root, workspace=workspace, target=target, require_credentials=False)
    names = ["HF_DATASETS_CACHE", "HF_HOME", "HF_MODULES_CACHE", "NLTK_DATA", "PYTHONPATH",
             "PYTHONPYCACHEPREFIX", "TOKENIZERS_PARALLELISM", "XDG_CACHE_HOME"]
    lines = ["#!/bin/sh", "set -eu", f"cd {shlex.quote(str(root))}"]
    lines.extend(f"export {name}={shlex.quote(environment[name])}" for name in names)
    if target.api_key_env:
        lines.append(f'export OPENAI_API_KEY="${{{target.api_key_env}:?Set {target.api_key_env}}}"')
    else:
        lines.append("export OPENAI_API_KEY=''")
    # A direct replay must also preserve the original run's outputs.
    lines.extend([': "${1:?Supply a new output directory}"', 'test ! -e "$1"',
                  shlex.join(command) + ' --output_path "$1"'])
    path.write_text("\n".join(lines) + "\n")


def Results(*, output: pathlib.Path, expected: dict[str, list[int]]) -> dict:
    result_files = list((output / "harness").rglob("results_*.json"))
    if len(result_files) != 1:
        raise ValueError("Harness did not write exactly one aggregate result")
    result = evals.records.ReadJSON(path=result_files[0])
    samples = {}
    for path in (output / "harness").rglob("samples_*.jsonl"):
        task = next((name for name in sorted(expected, key=len, reverse=True)
                     if path.name.startswith(f"samples_{name}_")), None)
        if task is None:
            raise ValueError(f"Unexpected sample file: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        filters = {item["name"]: [] for item in result["configs"][task].get("filter_list", [{"name": "none"}])}
        for row in rows:
            filters[row["filter"]].append(row["doc_id"])
        # GSM8K emits one row per question for each answer-extraction filter.
        if any(sorted(ids) != expected[task] for ids in filters.values()):
            raise ValueError(f"Harness sample output does not match the selected questions for {task}")
        samples[task] = len(expected[task])
    if samples.keys() != expected.keys():
        raise ValueError("Harness sample output does not match the selected questions")
    return {"completed_samples": samples,
            "metrics": result["results"], "task_versions": result["versions"]}
