# Evaluation Runner

The runner invokes [LM Evaluation Harness 0.4.13](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13)
against a running API server. It supports MMLU, HellaSwag, ARC Challenge, GSM8K, and IFEval. It does not deploy servers
or load model weights. See [Evaluations](../docs/evals.md) for the evaluation and comparison approach.

The runner code, command-line entry point, and dependency files live in this directory. Run it from the repository root
with `python -m evals`, using the project environment as shown below.

## Setup

Run commands from the repository root using the project's Python 3.14 environment:

```sh
.venv/bin/python -X pycache_prefix=.__pycache__ -m pip install --cache-dir .__pycache__/pip -r evals/requirements.lock
```

The lock file pins the harness and its dependencies. The harness is installed as a package; its source and scoring code
are not copied into FrontierSplit. `requirements.in` lists the direct evaluation dependencies.

## Configure a Target

Configuration files are JSON. Their filenames match their `id` fields. Start with the provided Mistral or Mixtral
configurations:

| File | Values to set before a model run |
| :--- | :--- |
| `configs/targets/mistral-base.json` | API base URL ending in `/v1`, served model name, and credential environment variable if needed. |
| `configs/backends/mistral-base.json` | Model and tokenizer revisions, precision, context limit, software versions, and engine settings. |
| `configs/deployments/gcp-ref.json` | Location, hardware, software, and launch instructions for the running deployment. |

The equivalent Mixtral target and backend files are also provided. The localhost URLs and empty deployment details are
placeholders. Set them to describe the actual server before collecting baseline results. The runner checks readiness,
the served model name, and the advertised context limit. Other deployment details are supplied by the configuration;
the API does not expose them.

If authentication is enabled, set `api_key_env` to the name of an environment variable containing the key. The runner
passes its value to the harness through the environment. Credentials are not written into configurations or commands.

## Select Evaluations

Each file in `configs/evals/` selects a harness task, its dataset revision, and its evaluation settings. The supplied
settings are a starting configuration, not a reproduction of a particular published score:

| ID | Harness task | Few-shot examples | Prompt format |
| :--- | :--- | ---: | :--- |
| `mmlu` | `mmlu` | 5 | Task's completion prompt |
| `hellaswag` | `hellaswag` | 0 | Task's completion prompt |
| `arc-challenge` | `arc_challenge` | 0 | Task's completion prompt |
| `gsm8k` | `gsm8k` | 5 | Task's completion prompt |
| `ifeval` | `ifeval` | 0 | Task's completion prompt |

The task definitions come from the pinned harness. Their prompts, preprocessing, scoring, and MMLU group aggregation
remain upstream definitions. The runner redirects dataset loading to verified local snapshots.

Evaluation settings include `num_fewshot`, `batch_size`, `num_concurrent`, `timeout`, `max_retries`, `max_gen_toks`,
`generation`, and `seeds` (Python, numerical-library, Torch, and few-shot seeds, in that order). Unspecified values are
resolved and saved with each run. Task generation settings still apply unless explicitly overridden in `generation`.

For chat generation, select `local-chat-completions`, set `apply_chat_template` to `true`, and use `batch_size: 1`.
Candidate scoring requires `local-completions`. Completion evaluations can also apply the pinned tokenizer's chat
template on the client. These are different prompt configurations and should have distinct evaluation IDs.

## Commands

Validate configurations without downloading data or contacting a server:

```sh
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ -m evals \
  --target mistral-base --suite quality --dry-run
```

Download and verify evaluation data and save harness commands without contacting a server:

```sh
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ -m evals \
  --target mistral-base --evals gsm8k --limit 10 --prepare-only
```

Run a short evaluation against the configured server:

```sh
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ -m evals \
  --target mistral-base --evals gsm8k --limit 10
```

Run the full suite:

```sh
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ -m evals \
  --target mistral-base --suite quality
```

`--evals` accepts one or more IDs. `--suite` selects the ordered list in `configs/suites/`. `--config-root` selects an
alternative configuration directory inside the repository. `--limit N` selects the first N questions from each task;
for MMLU it applies separately to each subject. The selected indices are saved. It limits inference, not dataset
downloads: the full dataset and few-shot pools are retained.

## Saved Files

Every invocation creates `eval_results/<target-id>/<run-id>/`. Its manifest records configurations, client package
versions, source revision and file hashes, server observations, and completion status. Source hashes identify local
changes but do not preserve their contents; publish a corresponding source revision with results.

Each evaluation directory contains:

- `manifest.json`: selected questions, dataset checksums, expected and completed counts, status, and scores.
- `harness/`: native harness results and per-question inputs, outputs, and scores.
- `logs/`: preparation and harness output.
- `harness.json` and `tasks/`: the resolved harness invocation and dataset overrides.
- `replay.sh`: the harness command and environment needed to repeat the evaluation directly.

Invoke a saved replay with `sh <evaluation-directory>/replay.sh <new-output-directory>`. The original run is preserved.
The command uses paths in the current checkout; on another machine, prepare the same configurations to regenerate them.
Normal runner invocations verify snapshots before each use. Direct replay invokes the harness alone.

Datasets are stored in `eval_data/` with source revisions, split names, and file checksums. IFEval's downloaded language
resources are also retained and checksummed. Native sample logs retain the actual prompts, including few-shot examples.
No previous model responses are reused.

A failed evaluation retains its output and does not prevent later evaluations from running. Interrupting a run stops the
current subprocess and records the interruption. Exit status is `0` for a completed or prepared run, `1` for a failed
run, `2` for invalid CLI configuration, and `130` for interruption. Preparation-only runs do not contain model scores.

## Current Scope

Comparison-report generation, serving-performance workloads, and RULER integration are deferred. The runner retains
the harness's native metrics and uncertainty estimates; it does not make a quality-equivalence decision. Actual
Mistral and Mixtral baseline scores still require the configured model backend.
