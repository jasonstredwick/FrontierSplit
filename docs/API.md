# FrontierSplit HTTP API

**Status: interface specification (pre-implementation).** This document defines the HTTP contract that the FrontierSplit
API server implements; examples illustrate wire formats rather than recorded output from a running deployment.

FrontierSplit specifies an OpenAI-compatible text and chat API, with selected vLLM extensions needed by existing
evaluation harnesses, over HTTP and Server-Sent Events (SSE). This living document defines the supported subset and
its compatibility limits and evolves with the project's serving and evaluation needs.

The compatibility reference is the text and chat interface of
[vLLM's OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
Raw text and token-ID prompts, echoed prompt scores, chat messages, and streaming use OpenAI-compatible formats.
Selected vLLM extensions provide readiness probing (`GET /health`), context-window discovery (`max_model_len` on
`GET /v1/models`), EOS control (`ignore_eos`), per-chunk streaming usage (`continuous_usage_stats`), and special-token
controls (`add_special_tokens`, `add_generation_prompt`). The fields, defaults, validation rules, and restrictions
below define FrontierSplit's contract; other vLLM behavior and future upstream changes are not implicitly included.

Compatibility is scoped strictly to the four endpoints below. Unrelated `vLLM` and OpenAI surfaces (`/v1/embeddings`,
`/v1/audio/*`, `/v1/responses`, LoRA management, tool-calling parsers, structured-output grammars, and beam search) are
out of scope. The external HTTP contract is identical across local and cloud deployments.

---

## Endpoints

| Endpoint | Purpose |
| :--- | :--- |
| `GET /health` | Reports whether the server and its backing model service are ready (`200`) or unavailable (`503`). |
| `GET /v1/models` | Lists served model card(s) and effective context window capacity (`max_model_len`). |
| `POST /v1/completions` | Raw text or token-ID completions and token log-probabilities. |
| `POST /v1/chat/completions` | Structured multi-turn chat generation rendered with the checkpoint's configured chat renderer. |

### Evaluation Integration

Evaluation clients select an endpoint according to the operation they need:

| Evaluation Path | Endpoint | Required API Behavior |
| :--- | :--- | :--- |
| lm-evaluation-harness `local-completions`: likelihood scoring | `/v1/completions` | Token-ID prompts, prompt echo, aligned token scores, and ordered batch results. |
| lm-evaluation-harness `local-completions`: text generation | `/v1/completions` | Raw prompts, generation budgets, sampling controls, and stop strings. |
| lm-evaluation-harness `local-chat-completions`: chat generation | `/v1/chat/completions` | Structured messages, server-side chat rendering, and assistant message responses. |
| Concurrent and streaming serving workloads | Either generation endpoint | Independent requests, incremental SSE output, finish reasons, and token usage. |

The harness's [local-completions adapter](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py)
requests likelihood scores using `echo: true`, `logprobs: 1`, and `max_tokens: 1`. It excludes the generated token when
scoring the prompt continuation. Supporting prefill-only requests (`max_tokens: 0`) does not replace this request path.
Harness versions, task settings, invocation details, and results belong with the evaluation records; this document
defines the API those clients use.

### Transport, Binding, and Authentication

* **Default bind address:** `127.0.0.1:8000` unless overridden via server configuration (`--host`, `--port`).
* **Concurrency:** The HTTP server handles multiple in-flight requests and open SSE streams asynchronously without
  HTTP-layer head-of-line blocking. Backend batching and execution scheduling remain isolated behind the internal
  backend boundary.
* **Authentication:** Optional Bearer token (`--api-key` or `FRONTIERSPLIT_API_KEY`).
  When unset, authentication is disabled. When set, every `/v1/*` request must supply `Authorization: Bearer <key>`
  (returning HTTP `401` otherwise), while `GET /health` remains unauthenticated for readiness probes.

### Health, Model Discovery, and Error Envelopes

`GET /health` returns HTTP `200 OK` when the server and an attached backend are healthy and ready to accept inference
work, and HTTP `503 Service Unavailable` when no backend is attached, initialization is still in progress, or the
backend is unhealthy.

`GET /v1/models` returns an OpenAI-compatible model list with the vLLM fields `root`, `parent`, and `max_model_len`.
The latter reports the deployment's effective context limit:

```json
{
  "object": "list",
  "data": [
    {
      "id": "<served-model-id>",
      "object": "model",
      "created": 0,
      "owned_by": "frontiersplit",
      "root": "<served-model-id>",
      "parent": null,
      "max_model_len": 32768
    }
  ]
}
```

All non-`200` HTTP responses on `/v1/*` routes use the standard OpenAI error JSON envelope:

```json
{
  "error": {
    "message": "Human-readable description of the failure.",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

| HTTP Status | `error.type` | Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `invalid_request_error` | Malformed JSON, missing required field, out-of-range value, requested unsupported capability, conflicting budget fields, invalid chat role sequence, or context overflow. |
| `401 Unauthorized` | `authentication_error` | Missing or invalid `Authorization: Bearer <key>` header when API-key auth is enabled. |
| `404 Not Found` | `not_found_error` | Unknown HTTP route or unrecognized `model` identifier. |
| `500 Internal Server Error` | `server_error` | Unhandled backend failure prior to committing HTTP `200` headers. |
| `503 Service Unavailable` | `service_unavailable` | Backend unattached, unhealthy, or still initializing. |

---

## Completions

`POST /v1/completions` is the raw-prompt interface and never applies a chat template.

### Request Fields

| Field | Type | Default | Validation & Behavior |
| :--- | :--- | :--- | :--- |
| `model` | `string` | *(required)* | Must match a served `id` in `GET /v1/models` (`404` otherwise). |
| `prompt` | `string \| list[string] \| list[int] \| list[list[int]]` | *(required)* | Non-empty single prompt or batch of non-empty prompts as UTF-8 text or integer token-ID lists (`0 <= id < vocab_size`, `400` otherwise). |
| `max_tokens` | `int \| null` | `16` | `null` uses the default (`16`). Otherwise `0 <= max_tokens <= max_model_len`; explicit `0` runs prompt prefill only (`stream: false`), and positive upper bounds cap to remaining context; see [Tokenization and Context Limits](#tokenization-and-context-limits). |
| `temperature` | `float \| null` | `1.0` | `null` uses the default. Otherwise finite `0.0 <= temperature <= 2.0`; `0.0` selects greedy (`argmax`) decoding. |
| `top_p` | `float \| null` | `1.0` | `null` uses the default. Otherwise finite `0.0 < top_p <= 1.0`. Applied when `temperature > 0.0`. |
| `seed` | `int \| null` | `null` | Signed 64-bit integer (`-2^63 <= seed <= 2^63 - 1`) per-request PRNG seed, or `null`. Reproducibility holds only under identical backend hardware, precision, and execution batching. |
| `stop` | `string \| list[string] \| null` | `null` | Non-empty string or list of non-empty strings. `null` or `[]` disables string stopping; empty strings and null list entries return `400`. Matched stop text is excluded from output. |
| `ignore_eos` | `bool` | `false` | `vLLM` extension: when `true`, ignores the checkpoint's EOS token and continues generating until the effective token budget or a `stop` string is reached. |
| `add_special_tokens` | `bool` | `true` | Controls tokenizer special-token insertion (such as BOS) for string prompts; ignored for pre-tokenized integer prompts. |
| `echo` | `bool` | `false` | Prepends prompt text (and prompt token scores when `logprobs` is set) to output. |
| `logprobs` | `int \| null` | `null` | `0 <= N <= 20`. Returns per-token natural log-probabilities and top-`N` candidates plus the selected token. |
| `stream` | `bool` | `false` | Streams SSE chunks. Requires a single prompt, a positive effective generation budget, `echo: false`, and `logprobs: null`. Incompatible combinations return HTTP `400` before streaming begins. |
| `stream_options` | `object \| null` | `null` | Controls streaming usage reporting; see [Streaming](#streaming) for fields and validation. |

### Additional Fields and Unsupported Capabilities

Omitted fields use the listed defaults. Fields accept `null` only where explicitly documented, boolean values are not
accepted for integer fields, and non-standard JSON numeric literals (`NaN`, `Infinity`, `-Infinity`) return HTTP `400`
and are never emitted in responses. The following compatibility fields are also accepted; values outside the listed set
return HTTP `400`:

| Field | Accepted Values | Behavior |
| :--- | :--- | :--- |
| `n`, `best_of` | `1` or `null` | One completion per prompt; no candidate selection. |
| `frequency_penalty`, `presence_penalty` | `0.0` or `null` | No token-frequency or presence penalty. |
| `repetition_penalty` | `1.0` or `null` | No repetition penalty. |
| `suffix` | `null` | No suffix-conditioned generation. |
| `logit_bias` | `{}` or `null` | No token-score adjustments. |
| `user` | `string` or `null` | Accepted as client metadata; does not affect generation. |
| `metadata` | `object` or `null` | Accepted as client metadata; does not affect generation. |

Unlisted fields return HTTP `400`, including unsupported generation controls. Clients can place additional metadata
inside `metadata`; arbitrary top-level fields are not silently ignored. This explicit field policy is narrower than
vLLM's general acceptance of extra fields.

### Response Semantics and Log-Probability Alignment

```json
{
  "id": "cmpl-01",
  "object": "text_completion",
  "created": 1730000000,
  "model": "<served-model-id>",
  "choices": [
    {
      "index": 0,
      "text": " world",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 1,
    "total_tokens": 3
  }
}
```

* **Batch ordering:** `choices` preserve input batch order (`index: 0 .. batch_size - 1`), and `usage` sums counts
  across the batch.
* **BOS / special-token policy:** Pre-tokenized integer ID prompts (`list[int]` / `list[list[int]]`) are always
  executed verbatim without prepending BOS or other special tokens. String prompts (`string` / `list[string]`) apply
  `add_special_tokens` (default `true`).
* **Independent `echo` and `logprobs` flags:**
  * `echo: true` with `logprobs: null` prepends the prompt to `choices[].text` and keeps `choices[].logprobs: null`.
  * `logprobs: N` with `echo: false` populates `choices[].logprobs` for generated completion tokens only.
  * `echo: true` with `logprobs: N` populates `choices[].logprobs` across all prompt tokens followed by any generated
    completion tokens (including `max_tokens: 0`, which returns `completion_tokens: 0` and `finish_reason: "length"`).
* **Probability distribution:** Prompt and generated-token scores use `log_softmax` over the model's unmodified
  logits across the full vocabulary, computed in `FP32` before temperature scaling or sampling filters such as
  `top_p`. Top candidates are ranked using this same distribution. Sampling controls affect token selection, not
  the reported scores; `temperature: 0` selects greedy generation without converting prompt scores into a one-hot
  distribution. Each score conditions on the preceding token IDs in that sequence.
* **`logprobs` structure:** Each `choices[].logprobs` object contains four equal-length arrays—`tokens` (detokenized
  token piece strings), `token_logprobs` (`FP32` natural log-probabilities $\ln P$), `top_logprobs` (maps from candidate
  token strings to $\ln P$), and `text_offset` (cumulative character offsets from the start of `choices[].text`).
  `token_logprobs[i]` always records the score of the actual selected token ID (for echoed prompt positions, the actual
  prompt token ID being scored). Each conditioned position in `top_logprobs` includes the top `N` candidate token IDs
  and the selected token ID, yielding up to `N + 1` map entries (`1` entry when `N = 0`). If multiple included token IDs
  detokenize to the same string key, `top_logprobs[i]` retains the maximum score among those colliding candidates.
  Scores are clamped below at `-9999.0` (so `-inf` serializes as valid JSON `-9999.0`, matching vLLM and OpenAI) and use
  identical `FP32` representation without divergent rounding across both structures: `token_logprobs[i]` equals
  `top_logprobs[i][tokens[i]]` unless a higher-scoring candidate shares that decoded string, and for `N >= 1`, when the
  selected token achieves the greedy maximum log-probability, `token_logprobs[i]` equals `max(top_logprobs[i].values())`.
  When `echo: true`, `token_logprobs[0]` and `top_logprobs[0]` are `null` for the unconditioned first prompt token.
* **Stop, EOS, and `ignore_eos` accounting:** When `ignore_eos: false` and generation hits the checkpoint's EOS token,
  `finish_reason` is `"stop"`, EOS is counted in `completion_tokens`, and EOS text is omitted from `text`. When
  `ignore_eos: true`, EOS tokens do not terminate generation. Stopping on a caller `stop` string sets
  `finish_reason: "stop"`, counts generated tokens up through the match in `completion_tokens`, and strips the matched
  `stop` substring from `text`. Reaching the effective generation budget sets `finish_reason: "length"`.

---

## Chat Completions

`POST /v1/chat/completions` accepts structured `messages` and renders them server-side using the served checkpoint's
configured chat renderer. Callers must pass raw message content rather than pre-rendered template strings.

### Request and Response Contract

* **Messages:** Non-empty list of objects with `role: "system" | "user" | "assistant"` and `content`. Content accepts
  either a string or a non-empty array of text parts, each shaped as `{"type": "text", "text": string}`. Text-part
  arrays follow the active chat renderer's content format (matching vLLM): string-content HuggingFace Jinja templates
  skip empty-string parts and join remaining text parts in array order with `"\n"` (`"\n".join`, yielding `""` if all
  parts are empty), whereas structured-content Jinja templates and native Mistral rendering (`mistral_common`) pass the
  ordered text-part objects directly to the renderer without inserting newline separators. Non-text parts, malformed
  parts, and null content return HTTP `400`. Role and turn-sequence constraints follow the active chat renderer; invalid
  turn sequences also return HTTP `400`.
* **Template & special-token controls:** `add_generation_prompt` (`bool`, default `true`) requests an assistant turn
  header where the active chat renderer supports one. `add_special_tokens` (`bool`, default `false`) avoids duplicating
  BOS or control tokens already emitted by chat rendering.
* **Token budgets:** Accepts `max_tokens` (`1 <= N <= max_model_len` or `null`) and `max_completion_tokens`
  (`1 <= N <= max_model_len` or `null`) as generation upper bounds. Passing both as non-`null` with unequal values
  returns HTTP `400`. Otherwise, a supplied non-null value is capped to remaining context capacity, or remaining
  capacity is used directly when neither field supplies one. The resulting effective budget must be positive (`>= 1`);
  see [Tokenization and Context Limits](#tokenization-and-context-limits).
* **Sampling, stopping, and streaming controls:** `model`, `temperature`, `top_p`, `seed`, `stop`, `ignore_eos`,
  `stream`, and `stream_options` follow the same rules as `POST /v1/completions`. The additional-fields table and
  rejection policy above also apply. Chat additionally accepts `tools: null`, `tool_choice: null` or `"none"`, and
  `response_format: null` or `{"type": "text"}` as no-ops; other values return HTTP `400`. Completion-only fields such
  as `prompt`, `echo`, and `logprobs` are not supported on this endpoint.
* **Non-streaming response:** Includes `id`, `object`, `created` (Unix seconds), `model`, `choices`, and `usage`.
  Each choice contains an assistant message and `finish_reason: "stop" | "length"`:

```json
{
  "id": "chatcmpl-01",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "<served-model-id>",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 2,
    "total_tokens": 14
  }
}
```

---

## Streaming

Streaming completions on `POST /v1/completions` require a single prompt, a positive effective generation budget,
`echo: false`, and `logprobs: null` (including their omitted defaults). Multi-prompt batches, prefill-only requests,
full-context prompts leaving zero generation capacity, prompt echo, and non-null `logprobs` (including `0`) return HTTP
`400` before HTTP `200` headers or SSE events are sent.

`stream_options` may be omitted or `null`. A non-null object is accepted only when `stream: true` and supports these
fields:

| Field | Type | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `include_usage` | `bool \| null` | `false` | Requests a final usage event before `data: [DONE]`. |
| `continuous_usage_stats` | `bool \| null` | `false` | Requests cumulative usage on each choice chunk when `include_usage` is also `true` (no effect otherwise). |

Omitted or null option values resolve to `false`; an empty object enables neither option. Non-null `stream_options`
with `stream: false`, unlisted keys inside the object, or values outside the listed types return HTTP `400`.

After request validation succeeds, `stream: true` on `POST /v1/completions` or `POST /v1/chat/completions` produces a
response on that same HTTP request with `200 OK` and `Content-Type: text/event-stream`:

* **Incremental delivery:** Output chunks are flushed without application-level buffering as pieces become available
  (multi-token UTF-8 characters, stop-sequence prefix holdback, and network framing mean one SSE event does not
  necessarily map 1:1 to a single model token).
* **Completion chunk schema (`POST /v1/completions`):**
  ```text
  data: {"id":"cmpl-01","object":"text_completion","created":1730000000,"model":"<served-model-id>","choices":[{"index":0,"text":" world","logprobs":null,"finish_reason":null}]}

  data: {"id":"cmpl-01","object":"text_completion","created":1730000000,"model":"<served-model-id>","choices":[{"index":0,"text":"","logprobs":null,"finish_reason":"stop"}]}

  data: {"id":"cmpl-01","object":"text_completion","created":1730000000,"model":"<served-model-id>","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}

  data: [DONE]
  ```
* **Chat chunk schema (`POST /v1/chat/completions`):** Uses `object: "chat.completion.chunk"`. The opening chunk sends
  `delta: {"role": "assistant", "content": ""}` with `finish_reason: null`, content chunks send
  `delta: {"content": "..."}`, and the closing choice chunk sends `delta: {}` with `finish_reason: "stop" | "length"`.
* **Usage reporting:** Matching vLLM's `should_include_usage` rule, `continuous_usage_stats` depends on `include_usage`:
  when both `stream_options.include_usage` and `stream_options.continuous_usage_stats` are `true`, each choice chunk
  (including the opening chat role chunk) includes cumulative `usage` (`prompt_tokens`, `completion_tokens`,
  `total_tokens`) through that event. When `stream_options.include_usage` is `true`, a final chunk with `choices: []`
  and total `usage` is emitted immediately before `data: [DONE]`.
* **Mid-stream failure and disconnect:** If backend execution fails after HTTP `200` headers are committed, the server
  emits an SSE `data: {"error": {"message": "...", "type": "server_error", "param": null, "code": "stream_interrupted"}}`
  event followed by `data: [DONE]` and closes the connection. If the client disconnects early, the API server cancels
  the local stream task and signals cancellation to the backend for that request.

---

## Tokenization and Context Limits

* **Checkpoint and renderer fidelity:** Tokenization, detokenization, special tokens, EOS IDs, and chat rendering
  follow the served checkpoint and the server's configured tokenizer/renderer mode (`hf` Jinja chat-template rendering
  vs. `mistral` native rendering via `mistral_common`). Context accounting uses the exact token IDs produced after chat
  rendering and special-token insertion, and pre-tokenized integer prompts validate every token ID against the served
  vocabulary (`0 <= id < vocab_size`, returning HTTP `400` otherwise).
* **Context capacity (no prompt truncation):** The deployment's effective limit `max_model_len` (configured at or below
  the checkpoint's native limit) is exposed via `GET /v1/models`. Prompts are never silently truncated: after
  tokenization, including template and special-token insertion, reject any prompt longer than `max_model_len` with HTTP
  `400` (`invalid_request_error`). Remaining capacity is `remaining = max_model_len - prompt_tokens`.
* **Budget resolution and capping:** Generation budget fields specify upper bounds rather than guaranteed output
  lengths. Reject any requested budget greater than `max_model_len` with HTTP `400`.
  * **Completions:** Omitted or `null` `max_tokens` resolves to a requested budget of `16`; an integer
    (`0 <= max_tokens <= max_model_len`) is used as requested. The effective budget is `min(requested, remaining)`.
  * **Chat:** Resolve `max_tokens` and `max_completion_tokens` (`1 <= N <= max_model_len`) before applying any default:
    unequal non-null values return HTTP `400`; otherwise a supplied non-null value sets the requested budget and
    resolves to `min(requested, remaining)`, while omitting or passing `null` for both fields resolves directly to
    `remaining`.
  * **Batches:** Resolve and validate each prompt separately in a completion batch; if any prompt fails validation,
    reject the entire request before generation.
* **Positive generation budget & prefill-only exception:** Generation requires a positive effective budget (`>= 1`). Any
  generative request whose prompt fills the context window (`remaining == 0`) resolves to an effective budget of `0` and
  returns HTTP `400` (`invalid_request_error`). The sole zero-budget exception is an explicit `max_tokens: 0` on
  `POST /v1/completions` with `stream: false`, which performs prefill only and may use the full context window.

For example, a completion prompt leaving eight tokens available with omitted or `null` `max_tokens` (`requested = 16`)
succeeds with its effective generation budget capped to `8`. A completion or chat prompt that fills the context window
(`remaining = 0`) returns HTTP `400` for any generative request, since its effective generation budget is `0`, whereas a
full-context completion prompt with explicit `max_tokens: 0` and `stream: false` remains valid.
