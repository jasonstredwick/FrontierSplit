# Model experiments

| Experiment | Implementation |
| --- | --- |
| `hugging_face` | The installed Hugging Face Transformers model classes. |
| `baseline` | Local copies of those classes, retaining the same model computation and shared Transformers utilities. |

Both experiments support Mistral-7B-Instruct-v0.3 and Mixtral-8x7B-Instruct-v0.1 with BF16 or FP16 weights.
They require Transformers `5.20.2`, the version used for the baseline source. TPU execution is not implemented in
these experiments.

## Model interface

The public interface consists of `Model` and `TensorSpec` in `frontiersplit.models.interface` and `GetModel`, `GetModelExperiment`,
`ModelName`, and `ExperimentName` in `frontiersplit.models.selection`. All other code under `models/` is internal.
Package `__init__.py` files are empty.

`GetModelExperiment` selects an implementation by model name and experiment name, loads its
weights onto one requested device, and returns an initialized `Model`. Each call creates a separate model instance.
`GetModel()` is a parameterless placeholder and raises `NotImplementedError` until a primary implementation is chosen.

`Model.Execute(inputs=...)` accepts and returns dictionaries of named tensors. The model's `device`, `input_specs`, and
`output_specs` describe their placement, shapes, and dtypes. A string dimension such as `"batch"` denotes a shared size;
an integer denotes a fixed size. Input validation checks tensor metadata without reading device values back to the CPU.

Each current experiment represents the whole network, from token embeddings through the vocabulary projection.
All weights, inputs, computation, and outputs reside on one CPU or CUDA device. The model does not move inputs between
devices or offload layers. Model loading initializes weights on the selected device before execution begins.

| Tensor | Direction | Shape | Dtype | Meaning |
| --- | --- | --- | --- | --- |
| `input_ids` | Input | `[batch, sequence]` | `int64` | Token IDs in the checkpoint's vocabulary. |
| `attention_mask` | Input | `[batch, sequence]` | `int64` | `1` for included tokens, `0` for padding. |
| `position_ids` | Input | `[batch, sequence]` | `int64` | Explicit positions for rotary embeddings. |
| `logits` | Output | `[batch, sequence, vocab_size]` | Model precision | Unnormalized next-token scores. |

The caller supplies valid token IDs, binary masks, and positions within the configured context range. Model entry
validates names, shapes, dtypes, device placement, and sequence length. It does not scan tensor values on the host.
Execution uses evaluation mode with gradients disabled.

Each call evaluates the supplied sequence independently. These adapters disable retained Transformers session state
and expose logits only. Incremental decoding with retained intermediate tensors is not implemented through this
interface yet; generation currently resubmits the growing sequence. These runs must not be treated as cached-decoding
performance baselines.

## Runner interface

`frontiersplit.runners.interface.Runner` represents a local portion of coordination. It can connect Models and handle
device transfers and communication. It does not imply one centralized coordinator for a distributed deployment.

`frontiersplit.runners.local.LocalRunner` currently invokes one whole-model experiment. It moves inputs to the Model's
device and places outputs on its explicitly configured `output_device`. Tensors already on the target device are passed
through without a copy. Outputs remain on the model device when `output_device=model.device`.

This local runner has no generation policy, tokenizer, network transport, or session manager. Those responsibilities
can be added to Runner implementations without changing the tensor-only Model boundary.

```python
import torch

from frontiersplit.models import selection as fs_selection
from frontiersplit.runners import local as fs_local


model = fs_selection.GetModelExperiment(
    model_name=fs_selection.ModelName.MISTRAL,
    experiment_name=fs_selection.ExperimentName.HUGGING_FACE,
    checkpoint="mistralai/Mistral-7B-Instruct-v0.3",
    revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
    dtype=torch.float16,
    device=torch.device("cuda:0"),
    attention="sdpa",
)
runner = fs_local.LocalRunner(model=model, output_device=model.device)
output = runner.Execute(inputs={
    "attention_mask": torch.ones((1, 3), dtype=torch.int64),
    "input_ids": torch.tensor([[1, 4, 7]]),
    "position_ids": torch.tensor([[0, 1, 2]]),
})
logits = output["logits"]  # Remains on cuda:0.
```

Select `ExperimentName.BASELINE` with the same arguments to use the local implementation. For Mixtral, select
`ModelName.MIXTRAL` and checkpoint
`mistralai/Mixtral-8x7B-Instruct-v0.1`; the existing evaluation configuration uses revision
`eba92302a2861cdc0098cc54bc9f17cb2c47eb61`. Placement and attention implementation are explicit loader arguments.
The whole model must fit on the chosen device. Automatic multi-device placement is not supported; a future partitioned
experiment must define separate Models connected by Runners.

Checkpoints may also be local directories. Transformers ignores `revision` for a local directory; its contents must be
retained separately for reproducibility. Downloads use `.__pycache__/huggingface/hub/`. The loader rejects incompatible
architectures and missing, unexpected, or mismatched weights.

These are model implementations, not API servers. Tokenization, generation policy, and serving integration belong to
the calling code.

## Source and attribution

Each experiment's `mistral.py` and `mixtral.py` supplies an internal loader that returns an initialized `Model`.
The `hugging_face` loaders use installed Transformers classes; the `baseline` files contain the local implementations.
Those classes are internal and retain their upstream names for source comparison. Each baseline header records the
upstream path, version, and SHA-256 of the original file.
The copied computation retains upstream naming and structure for comparison; the selector, interface, and adapter
follow project conventions.

Local source changes replace relative imports, remove unused classification and question-answering heads and upstream
generated-file banners, add internal loading functions, remove public class exports, and add provenance headers.
The local Mixtral causal-LM class has a distinct name and registers
the upstream checkpoint conversion rules for that name. This enables loading older expert-weight layouts without
changing the upstream reference class's registration. See [third-party notices](../../../../THIRD_PARTY_NOTICES.md).

## Validation

`tests/test_model_experiments.py` loads identical small checkpoints through the public selector and compares `Execute`
outputs with direct Transformers execution. It covers BF16 and FP16, eager and SDPA attention, padded batches,
growing sequences, independent repeated calls, invalid input contracts, and greedy token selection. Temporary
checkpoints stay under `scratch/`. `tests/test_runners.py` checks same-device reference reuse between two Runners and
CPU/CUDA transfers when CUDA hardware is available.

These checks cover small CPU configurations. Full-size checkpoint evaluation, GPU measurements, and serving integration
remain to be performed.
