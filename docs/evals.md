# FrontierSplit Evaluations

**Status: evaluation plan.** The suite below is the intended baseline. This document does not yet identify pinned run
configurations, selected published reference results, or completed baseline reports. It should not be read as a claim
that these evaluations have been run successfully against FrontierSplit.

## Why We Run Evals

FrontierSplit changes how a model is partitioned and executed. External evaluations will measure two properties through
the public interface:

1. **Quality preservation:** Determine whether execution changes degrade performance on the selected benchmarks,
   relative to a reference run of the same checkpoint.
2. **Serving behavior under load:** Measure throughput and latency under specified concurrent and multi-turn workloads,
   alongside a single-request baseline.

The plan is to reuse established evaluation and load harnesses (**LM Evaluation Harness**, **RULER**, and **GuideLLM**).
Together, this list covers general knowledge and commonsense (`MMLU`, `HellaSwag`, `ARC-Challenge`), multi-step
reasoning (`GSM8K`), objectively verifiable instruction following without an LLM judge (`IFEval`), long-context
retrieval (`RULER`), and concurrent serving performance (`GuideLLM`).

The harnesses provide existing tasks and scoring or load-generation machinery. Project-specific workload profiles and
session-lifecycle checks may be added where needed. Benchmark agreement provides evidence on the tested tasks; it does
not establish identical behavior for every possible input.

---

## Evaluation Suite

| Evaluation | What It Measures | Existing Harness | Priority |
| :--- | :--- | :--- | :--- |
| **MMLU** | Knowledge and reasoning across subjects | [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) | Initial baseline |
| **HellaSwag** | Commonsense continuation selection | LM Evaluation Harness | Initial baseline |
| **ARC-Challenge** | Science question answering and reasoning | LM Evaluation Harness | Initial baseline |
| **GSM8K** | Mathematical problem solving | LM Evaluation Harness | Initial baseline |
| **IFEval** | Following objectively verifiable instructions | LM Evaluation Harness | Initial baseline |
| **RULER** | Retrieval and reasoning across long contexts | [RULER](https://github.com/NVIDIA/RULER) | Before & after context/KV changes |
| **Concurrent & multi-turn workloads** | Throughput, response latency, streaming latency, and behavior under load | [GuideLLM](https://vllm-project.github.io/guidellm/stable/getting-started/benchmark/) | Serving baseline & distribution work |

---

## Harness Integration

The intended integration runs each harness out-of-process against the HTTP contract in [`API.md`](API.md), using a
supported client backend. Compatible endpoint names alone are insufficient: the server must provide the response fields
and scoring capabilities required by that backend.

Quality harnesses supply datasets, task prompts, answer processing, and scoring. GuideLLM supplies traffic generation
and serving measurements. FrontierSplit supplies model execution. A configuration should be reusable against a
compatible reference server, changing the endpoint and served-model identifier while preserving evaluation settings.

For chat templates, exactly one side owns rendering: either the harness renders a prompt for `/v1/completions`, or the
server renders structured messages sent to `/v1/chat/completions`. The chosen benchmark protocol determines which is
appropriate; an Instruct checkpoint alone does not determine the evaluation's formatting. When comparing FrontierSplit
against a reference server on `/v1/chat/completions`, both deployments must use a matching chat rendering path and
verify identical prompt token IDs on the rendered inputs before attributing differences to model execution. Because the
HTTP contract caps generation budgets to remaining context capacity rather than rejecting oversized budgets,
fixed-length evaluation configurations must validate beforehand that prompts leave the full required output capacity
within `max_model_len`.

## References and Reproducibility

First select a reproducible published result for the exact checkpoint, where one is available. Paper and leaderboard
scores may use different prompting and scoring protocols even when they share a benchmark name. Record the source and
its configuration before treating a score as a reproduction target.

Where a published protocol is incomplete or no matching result is available, use a documented run of the unmodified
reference implementation. Label this as a local reference rather than a reproduction of a published score.

Each evaluation configuration and report should record:

| Area | Required Record |
| :--- | :--- |
| Model | Checkpoint and tokenizer revisions, tokenizer/renderer mode, chat template, precision, effective context limit |
| Evaluation | Harness version/commit, exact task identifiers, dataset revision and split, sample count, few-shot settings, prompt/chat-template configuration, generation settings, scoring metric |
| Execution | Reference or FrontierSplit implementation revision, relevant runtime and tokenizer library versions, hardware and device layout |
| Serving workload | Input/output length distribution, arrival profile, concurrency, turn/history policy, streaming mode, warmup and measurement interval |
| Evidence | Invocation/configuration, rendered prompt-token-ID parity check, per-example results or request traces, aggregate metrics, failures, and reference-result source |

For quality comparisons, define the acceptance margin, confidence level, and uncertainty method for each reported metric
before comparing results. Distinguish finite-dataset uncertainty from variability across repeated generation runs.
Overlapping confidence intervals alone do not establish equivalence. These criteria remain unspecified in this document.

Serving reports should identify the reported latency percentiles, throughput definition, error counts, and variability
across repeated runs. Published speed figures are comparison targets only when hardware, runtime, and workload details
are sufficiently matched.

## Configuration and Results Status

Pinned harness configurations, executable invocations, reference-result links, and report locations remain to be added
for LM Evaluation Harness, RULER, and GuideLLM. The suite table is a selection of evaluations, not a runnable preset.
Those artifacts will define the exact task variants and metrics; this document does not substitute for them.

The intended progression is an unmodified reference run, a straightforward single-machine FrontierSplit run, and then
distributed execution. Add RULER before context/KV changes and retain the reference configurations for later comparisons.

---

## Eval-Specific Notes

Different evaluations rely on distinct server capabilities covered in [`API.md`](API.md):

* **Likelihood-scored evaluations (`MMLU`, `HellaSwag`, `ARC-Challenge`):** The selected likelihood-based task variants
  score candidate continuations rather than free-form answers. Their HTTP integration requires pre-tokenized prompts,
  echoed prompt-token log-probabilities, and aligned batch responses on [`POST /v1/completions`](API.md#completions).
  The selected `lm-evaluation-harness` adapter requests one generated token and excludes it from continuation scoring.
* **Generative and instruction evaluations (`GSM8K`, `IFEval`):** Score generated responses using the selected task's
  prompting, decoding, stop conditions, and answer processing. Greedy decoding is a configuration choice, not a universal
  property of these benchmarks. Endpoint selection must follow the template-rendering policy above.
* **Long-context evaluations (`RULER`):** Vary sequence length and task complexity to test capabilities that short-prompt
  benchmarks may miss. Record tested lengths and reserve space for generated output within the effective context limit.
  Do not silently truncate inputs or describe untested lengths as validated
  ([context limits](API.md#tokenization-and-context-limits)).
* **Concurrent serving workloads (`GuideLLM`):** Measure responsiveness and capacity across concurrent streams and
  multi-turn conversations. Record client-observed time-to-first-token, streaming latency, throughput, and token usage;
  streaming event timing is not necessarily individual model-token timing ([streaming contract](API.md#streaming)).
  Specify how conversation history is carried between turns ([chat completions](API.md#chat-completions)). These
  workloads do not themselves score answer quality or establish session isolation.

---

## Initial Checkpoints

| Model | Role | Execution Precision |
| :--- | :--- | :--- |
| `mistralai/Mistral-7B-Instruct-v0.3` | Dense baseline | Unquantized FP16 or BF16, explicitly selected per configuration |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE baseline | Unquantized FP16 or BF16, explicitly selected per configuration |

Each run must pin the checkpoint revision, tokenizer revision, and explicit tokenizer/renderer configuration rather than
relying on automatic defaults, which can change across repository revisions; record any observed defaults as part of
that pinned configuration rather than permanent guarantees. FP16 and BF16 runs are separate configurations. Published
results for base models or other checkpoint versions are not interchangeable with results for these Instruct models.
