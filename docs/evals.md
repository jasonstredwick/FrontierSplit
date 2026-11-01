# FrontierSplit Evaluations

FrontierSplit changes how model inference is run. Evaluations measure whether those changes affect model quality. We first evaluate the selected model using the baseline backend. When published scores include enough information to reproduce the evaluation, we use them to check that baseline. We then evaluate the same model with changes to the backend. Both runs use the same API, evaluation questions, prompting, and scoring settings. Comparing their scores shows whether the backend changes affect performance on those evaluations.

**Status:** The evaluation tools and baseline results described in this document are not yet available.

## Evaluations

Quality evaluations score the model's answers. Serving performance evaluations measure how quickly the system handles
requests. The table lists both, including evaluations whose integration is deferred.

| Evaluation | What it measures | Tool | Integration |
| :--- | :--- | :--- | :--- |
| MMLU | Knowledge and reasoning across subjects | LM Evaluation Harness | Initial implementation |
| HellaSwag | Choosing a plausible continuation of a situation | LM Evaluation Harness | Initial implementation |
| ARC Challenge | Answering science questions | LM Evaluation Harness | Initial implementation |
| GSM8K | Solving mathematical word problems | LM Evaluation Harness | Initial implementation |
| IFEval | Following explicit instructions | LM Evaluation Harness | Initial implementation |
| RULER | Retrieval and reasoning over long contexts | RULER | Deferred — requires a compatible API client |
| Serving performance | Throughput and latency under specified request loads | GuideLLM | Deferred — workloads not yet defined |

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) runs the first five benchmarks. It loads
their datasets, prepares requests, and scores responses. MMLU, HellaSwag, and ARC Challenge compare the probabilities
assigned to candidate answers or continuations. GSM8K checks generated answers against expected answers; IFEval checks
responses against instruction constraints. These checks are automatic.

Each evaluation configuration selects the exact task variant, dataset version, prompt format, and scoring method.
GuideLLM and RULER use separate tools, described under [Deferred Evaluations](#deferred-evaluations).

## Baseline

The baseline provides measurements to guage the effect of backend changes. The baseline runs the selected model through FrontierSplit's OpenAI compatible API. The serving layer adapts vLLM's request and response processing for the features required by the evaluation harnesses. The baseline backend executes the model before experimental changes are introduced. Later backend versions are evaluated through the same API and compared with the baseline results.

The baseline configuration records the model implementation, model and tokenizer revisions, precision, software versions, runtime settings, hardware, and evaluation settings. When a suitable published result is available, the baseline report identifies that result and describes any differences between its setup and the baseline setup. Otherwise, the recorded baseline backend scores provide the comparison point. Published evaluation configurations and expected scores, including those used by vLLM's accuracy checks, can help validate the baseline where they match the selected model and evaluation settings.

## Running Evaluations

FrontierSplit’s evaluation runner uses a fixed version of LM Evaluation Harness. The runner reads the selected configurations, checks the server endpoint and model name, and starts the harness. The harness connects to an already running server at the target’s API URL, loads the dataset and any required tokenizer, prepares requests, and scores the responses. The runner saves the results and the harness commands needed to repeat the evaluation directly.

A run selects a target and one or more evaluations. A suite is a named list of evaluations that can be selected together. Each evaluation keeps its own settings, whether run alone or as part of a suite. A short test run can use a subset of questions; the results record which questions were used.

The evaluations listed above use the model in two ways:
- **Score supplied answers:** The harness sends a question with each possible answer and asks for the probabilities of the answer’s tokens. It uses those probabilities to rank the answers.
- **Generate an answer:** The harness sends a question or instruction and scores the text the model produces.

Both use the endpoints described in [API.md](API.md).

## Configuration

The configurations separate what is evaluated from the software and machines used to run the model.

| Configuration | Describes |
| :--- | :--- |
| Evaluation | The questions, prompting, generation settings, and scoring method. |
| Backend | The model and the software that serves and executes it. |
| Deployment | The machines and runtime environment where that software runs. |
| Target | A backend running in a deployment, with an API address and served model name. |
| Suite | A named list of evaluations. |
| Comparison | The saved results to compare and the method used to compare them. |

An **evaluation configuration** records the harness and task versions, dataset version and splits, selected examples,
number of few shot examples, prompt settings, generation controls, metrics, and random seeds. Few shot examples are
the demonstration questions and answers included in a prompt. The configuration also records the API adapter and path,
batch size, number of concurrent requests, timeouts, retry settings, and any published scores used to check the baseline.

A **backend configuration** records the model and tokenizer versions, precision, serving and inference software
versions, chat template, context limit, and engine settings. It identifies the software package, image, or dependency
versions needed to reproduce the execution. The baseline and modified backends use separate configurations.

A **deployment configuration** records the cloud and location, machine types, accelerators, memory, operating system,
drivers, and runtime versions. It describes where processes or model stages run and provides the launch instructions.

A **target configuration** connects a backend ID and deployment ID to an API base URL and served model name. Credentials
are read from a named environment variable. The same evaluation can run against the baseline or a modified backend by
selecting a different target.

A **suite configuration** lists evaluation IDs. It does not change their settings. If a suite reports a combined score,
its configuration defines that calculation. A **comparison configuration** identifies the saved results to use, the
settings deliberately changed between runs, and the calculations used in the report.

### Names and Versions

Every configuration has three common fields:

| Field | Use |
| :--- | :--- |
| `id` | A short name for filenames, directories, and references, such as `gsm8k`, `mistral-base`, or `gcp-ref`. |
| `display_name` | A descriptive name for reports. |
| `schema_version` | The version of the configuration file format. |

IDs use lowercase ASCII letters and digits, with single hyphens between words or numbers. IDs are unique within each
configuration type. For example, the target ID `gcp-ref` uses a configuration filename whose stem is `gcp-ref`.
The full model identifier, such as `mistralai/Mistral-7B-Instruct-v0.3`, is a separate value.

Each run saves the configurations it used, including defaults and overrides, along with their checksums. Later edits to
a configuration do not alter the settings recorded for an earlier run. Passwords and API keys are excluded from saved
configurations, commands, and results.

### Directory Structure

Paths are relative to the repository root.

```text
configs/
  evals/                 # Evaluation settings
  backends/              # Model and execution settings
  deployments/           # Machine descriptions and launch instructions
  targets/               # Backend, deployment, and API address
  suites/                # Lists of evaluations
  comparisons/           # Saved results to compare and analysis settings
evals/
  requirements.lock      # Exact harness dependency versions
  reports/               # Comparison reports and links to saved results
scripts/
  run_eval.py            # Evaluation runner
eval_data/               # Saved dataset versions; outside Git
eval_results/            # Run results; outside Git
```

## Reproducing a Run

Repeating an evaluation requires the same model, evaluation settings, and data. Each run saves its configurations and
commands, along with the software versions and hardware used by both the evaluation client and the server. The record
includes the code revisions and any local changes that affected the run.

Evaluation datasets are pinned to specific versions and retained in `eval_data/`. Each saved dataset includes its source
and checksums so its contents can be verified before reuse. Runs record the exact questions used, any examples included
in the prompts, and random seeds. This allows a later run to use the same inputs even if the source dataset changes.
Each run generates new model responses rather than reusing answers from an earlier run.

Published results include the saved settings and instructions for recreating the environment and running the evaluation.
They also provide the dataset versions and checksums, with copies of the data where redistribution is permitted.
Otherwise, the instructions describe how to obtain and verify the required data.

## Saved Results

A run is one invocation against one target and can contain several evaluations. The target's configuration stays fixed
during the run. Every invocation, including a rerun, receives a new run ID.

Each evaluation result is identified by `(target_id, run_id, evaluation_id)` and stored as follows:

```text
eval_results/<target-id>/<run-id>/
  manifest.json
  evaluations/<evaluation-id>/
    manifest.json
    harness/
    logs/
```

The run's `manifest.json` describes the run: its configurations and checksums, client and server environments, commands,
start and end times, and completion status. It distinguishes information supplied in deployment records from information
observed through the API.

Each evaluation's `manifest.json` records its settings, dataset snapshot, selected examples, expected and completed
sample counts, exit status, and errors. The `harness/` directory contains the harness's original scores and per example
inputs and outputs. It retains scores for individual subtasks and their totals or averages, such as MMLU subjects and
the overall MMLU score. The `logs/` directory contains execution logs.

Completed results are kept without overwriting them. If one evaluation fails or is interrupted, the run records that
status and retains results from the evaluations that completed.

## Comparing Results

A comparison report brings together saved results from the baseline and modified backends. Evaluations can be run separately and combined later; they do not need to come from a single run.

For each evaluation, the report checks that both results used the same questions, prompts, scoring method, model weights, precision, and generation settings. Task versions must also match. The report identifies any differences and distinguishes deliberate experimental changes from differences that limit the comparison. Missing evaluations are listed.

Reports show the baseline score, modified backend score, score difference, and number of questions evaluated. Repeated runs remain individually identifiable. If their scores are combined, the report explains how.

When a comparison assesses whether model quality is preserved, its configuration defines the acceptable score difference and statistical method before the modified backend’s results are examined. Any uncertainty estimates distinguish between variation across evaluation questions and variation across repeated runs.

Comparison configurations are stored in `configs/comparisons/` and selected reports in `evals/reports/`. Published reports link to the saved results and include checksums for downloaded files.

## Deferred Evaluations

- **[GuideLLM](https://vllm-project.github.io/guidellm/stable/getting-started/benchmark/)** measures serving throughput and
  latency. Integration is deferred until workloads are defined. A workload specifies request lengths, arrival rate or
  concurrency, streaming, conversation history, client location, and measurement duration.
- **[RULER](https://github.com/NVIDIA/RULER/blob/main/scripts/pred/client_wrappers.py)** measures performance over long
  contexts. Integration is deferred until a client supports the FrontierSplit API.

## Initial Models

| Model | Role | Precision |
| :--- | :--- | :--- |
| `mistralai/Mistral-7B-Instruct-v0.3` | Dense baseline | Unquantized FP16 or BF16 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE baseline | Unquantized FP16 or BF16 |

Each backend configuration selects its precision explicitly. FP16 and BF16 runs use separate configurations.
Baseline scores identify the exact model and tokenizer revisions used.
