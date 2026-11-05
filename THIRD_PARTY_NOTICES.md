# Third-party notices

FrontierSplit includes portions of the projects identified below, modified for its serving interfaces and model experiments.
These portions are licensed under Apache License 2.0; the license text is included in [LICENSE](LICENSE).
The notices below identify incorporated source code, including attribution inherited through vLLM. They are not
an inventory of separately installed dependencies.

## vLLM

Copyright contributors to the vLLM project

- Project: [vLLM](https://github.com/vllm-project/vllm)
- License: [Apache License 2.0][vllm-license]

The following files contain adaptations. The attribution applies to the indicated portions, alongside
FrontierSplit's own implementation; it does not imply that entire files are unchanged upstream copies.

| FrontierSplit file | Adapted portions | Upstream sources |
| --- | --- | --- |
| `frontiersplit/openai_api/detokenizer.py` | Incremental decoding, prefix/read offsets, Metaspace restoration, stop holdback, and token truncation. | [Engine detokenizer][detokenizer], [tokenizer utilities][detokenizer-utils] |
| `frontiersplit/openai_api/errors.py` | Bearer-token authentication, validation-location filtering and bounded error reporting, disconnect cancellation. | [Authentication middleware][authenticate], [validation handler][validation], [API utilities][api-utils] |
| `frontiersplit/openai_api/generation.py` | Final versus delta output assembly, decoded text and score selection, and prompt-score emission. | [Output processor][output-processor] |
| `frontiersplit/openai_api/logprobs.py` | Prompt/sample score accumulation, recent-token context, and UTF-8 byte-fallback reconstruction. | [Log-probability processor][logprobs-processor] |
| `frontiersplit/openai_api/protocol.py` | Restricted request/response schemas, usage options, and sampling normalization. | [Completion protocol][completion-protocol], [chat protocol][chat-protocol], [engine protocol][engine-protocol], [generation protocol][generation-protocol], [API utilities][api-utils], [sampling parameters][sampling-params] |
| `frontiersplit/openai_api/renderer.py` | Jinja AST access/alias analysis, content-format detection, and the generation-tag parser. | [Hugging Face renderer][hf-renderer] |
| `frontiersplit/openai_api/server.py` | Completion/chat route dispatch to JSON or streaming responses, with request validation and cancellation. | [Completion router][completion-router], [chat router][chat-router] |
| `frontiersplit/openai_api/serving.py` | Completion/chat response assembly, prompt echo, log-probability formatting, and streaming usage output. | [Completion serving][completion-serving], [chat serving][chat-serving] |
| `frontiersplit/openai_api/types.py` | Reduced `CompletionOutput`, `RequestOutput`, and token log-probability records. | [Output types][outputs], [log-probability types][logprobs-types] |

FrontierSplit changes include a text-only API subset, stricter request validation, renamed and reorganized functions,
reduced schemas, an independent token-level backend contract, and modified cancellation and output handling.
The backend modules and test harnesses are project-specific implementations and are not included in this source mapping.

### Reference revisions

The links below identify revisions whose file contents were verified against the saved upstream sources during the
attribution review. The original download revisions were not recorded; these are content-verified references,
not a claim about the commit at which each adaptation was first made.

- Most referenced vLLM files match commit `900b8b08e2552c28a004e075866b5e1a39b99ac3`.
- `vllm/renderers/hf.py` matches commit `60a7d568f46851992a364074cbfd96e10ccd68a6`.
- `vllm/sampling_params.py` matches commit `e1e77a279d68a9601be062cb267a5f9911c422c7`.

## Hugging Face Transformers model implementations

The local baseline adapts model code from Hugging Face Transformers `5.20.2`:

| FrontierSplit file | Upstream source path |
| --- | --- |
| `frontiersplit/models/experiments/baseline/mistral.py` | `src/transformers/models/mistral/modeling_mistral.py` |
| `frontiersplit/models/experiments/baseline/mixtral.py` | `src/transformers/models/mixtral/modeling_mixtral.py` |

- Project: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- License: Apache License 2.0 (included in [LICENSE](LICENSE))
- Upstream copyright notice: Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.

Original distribution file paths and SHA-256 hashes are recorded in each file's header. FrontierSplit modifications
reorganize and rename symbols to match project conventions, remove unused task heads and optional configuration or
training scaffolding, and narrow execution to the project's target inference path while retaining the upstream model
computation.

## Attribution inherited through vLLM

### FastChat

vLLM's protocol modules credit
[FastChat's OpenAI API protocol definitions][fastchat-protocol] at commit
`168ccc29d3f7edc50823016105c024fe2282732a`. This attribution is retained for the reduced definitions in
`frontiersplit/openai_api/protocol.py`. FastChat's source at that revision is licensed under
[Apache License 2.0][fastchat-license].

### Hugging Face Transformers

Copyright 2018- The Hugging Face team. All rights reserved.

Copyright 2024 The HuggingFace Team. All rights reserved.

`_GenerationTagExtension.parse` in `frontiersplit/openai_api/renderer.py` is adapted through vLLM's
`AssistantTracker` from [Transformers' chat-template utilities at v4.56.2][transformers-template].
The parser is retained for template inspection. The upstream code is licensed under
[Apache License 2.0][transformers-license].

### Hugging Face Text Generation Inference

vLLM's incremental detokenization utility credits
[Text Generation Inference's `Model.decode_token` at v0.9.4][tgi-model]. That attribution is retained for the
slow decoding path in `frontiersplit/openai_api/detokenizer.py`. The referenced v0.9.4 source is licensed under
[Apache License 2.0][tgi-license].

## OpenAI API compatibility

The OpenAI-compatible interface describes supported request and response formats. The source adaptations identified
above come from vLLM and its credited upstream sources; this serving layer does not incorporate OpenAI SDK code.

[vllm-license]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/LICENSE
[detokenizer]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/v1/engine/detokenizer.py
[detokenizer-utils]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/tokenizers/detokenizer_utils.py
[authenticate]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/serve/middleware/authenticate.py
[validation]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/serve/exception_handling/handlers/validation.py
[api-utils]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/serve/utils/api_utils.py
[output-processor]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/v1/engine/output_processor.py
[logprobs-processor]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/v1/engine/logprobs.py
[completion-protocol]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/openai/completion/protocol.py
[chat-protocol]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/openai/chat_completion/protocol.py
[engine-protocol]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/serve/engine/protocol.py
[generation-protocol]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/generate/base/protocol.py
[sampling-params]: https://github.com/vllm-project/vllm/blob/e1e77a279d68a9601be062cb267a5f9911c422c7/vllm/sampling_params.py
[hf-renderer]: https://github.com/vllm-project/vllm/blob/60a7d568f46851992a364074cbfd96e10ccd68a6/vllm/renderers/hf.py
[completion-router]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/openai/completion/api_router.py
[chat-router]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/openai/chat_completion/api_router.py
[completion-serving]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/openai/completion/serving.py
[chat-serving]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/entrypoints/openai/chat_completion/serving.py
[outputs]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/outputs.py
[logprobs-types]: https://github.com/vllm-project/vllm/blob/900b8b08e2552c28a004e075866b5e1a39b99ac3/vllm/logprobs.py
[fastchat-protocol]: https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
[fastchat-license]: https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/LICENSE
[transformers-template]: https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/chat_template_utils.py
[transformers-license]: https://github.com/huggingface/transformers/blob/v4.56.2/LICENSE
[tgi-model]: https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py
[tgi-license]: https://github.com/huggingface/text-generation-inference/blob/v0.9.4/LICENSE
