import pathlib
import pytest
import safetensors.torch
import tempfile
import torch
import transformers

from frontiersplit.models import interface as fs_interface
from frontiersplit.models import selection as fs_selection
from frontiersplit.runners import local as fs_local


REFERENCE_CLASSES = {
    fs_selection.ModelName.MISTRAL: transformers.MistralForCausalLM,
    fs_selection.ModelName.MIXTRAL: transformers.MixtralForCausalLM,
}


@pytest.fixture(name="workspace")
def Workspace():
    root = pathlib.Path.cwd() / "scratch"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix="model-experiment-") as directory:
        yield pathlib.Path(directory)


def Configuration(*, model_name: fs_selection.ModelName) -> transformers.PretrainedConfig:
    settings = {
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "intermediate_size": 48,
        "max_position_embeddings": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "sliding_window": 4,
        "vocab_size": 32,
    }
    if model_name == fs_selection.ModelName.MIXTRAL:
        return transformers.MixtralConfig(**settings, num_experts_per_tok=2, num_local_experts=4)
    return transformers.MistralConfig(**settings)


@pytest.mark.parametrize("model_name", [fs_selection.ModelName.MISTRAL, fs_selection.ModelName.MIXTRAL])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_reference_and_baseline_outputs(*, workspace, model_name, dtype, attention):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        original = REFERENCE_CLASSES[model_name](config=Configuration(model_name=model_name))
    checkpoint = workspace / "checkpoint"
    original.save_pretrained(checkpoint)
    del original
    reference, baseline = [fs_selection.GetModelExperiment(
        model_name=model_name, experiment_name=experiment_name, checkpoint=str(checkpoint), revision="local",
        dtype=dtype, device=torch.device("cpu"), attention=attention,
    ) for experiment_name in [fs_selection.ExperimentName.HUGGING_FACE, fs_selection.ExperimentName.BASELINE]]
    original = REFERENCE_CLASSES[model_name].from_pretrained(
        checkpoint, dtype=dtype, device_map="cpu", attn_implementation=attention,
    ).eval()
    assert isinstance(reference, fs_interface.Model)
    assert isinstance(baseline, fs_interface.Model)
    input_ids = torch.tensor([[1, 4, 7, 8, 9, 10], [0, 0, 1, 5, 6, 11]])
    attention_mask = input_ids.ne(0).long()
    inputs = {
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "position_ids": torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1),
    }
    runners = [fs_local.LocalRunner(model=model, output_device=torch.device("cpu")) for model in [reference, baseline]]
    with torch.inference_mode():
        expected = original(**inputs, use_cache=False).logits
    for model, runner in zip([reference, baseline], runners, strict=True):
        assert model.device == torch.device("cpu")
        assert model.input_specs["input_ids"].shape == ("batch", "sequence")
        assert model.output_specs["logits"].shape == ("batch", "sequence", 32)
        actual = model.Execute(inputs=inputs)["logits"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
        assert actual.device == model.device
        assert actual.dtype == dtype
        torch.testing.assert_close(runner.Execute(inputs=inputs)["logits"], expected, rtol=0, atol=0)

    # Retain the growing sequence outside the Model and replay the original input between calls.
    growing_ids = input_ids
    for step in range(4):
        growing_ids = torch.cat([growing_ids, torch.tensor([[12 + step], [16 + step]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((2, 1), dtype=torch.long)], dim=1)
        step_inputs = {
            "attention_mask": attention_mask,
            "input_ids": growing_ids,
            "position_ids": torch.arange(growing_ids.shape[1]).expand(growing_ids.shape[0], -1),
        }
        with torch.inference_mode():
            expected_step = original(**step_inputs, use_cache=False).logits
        for runner in runners:
            torch.testing.assert_close(runner.Execute(inputs=step_inputs)["logits"], expected_step, rtol=0, atol=0)
            torch.testing.assert_close(runner.Execute(inputs=inputs)["logits"], expected, rtol=0, atol=0)

    prompt = torch.tensor([[1, 4, 7], [1, 5, 6]])
    with torch.inference_mode():
        expected_tokens = original.generate(
            input_ids=prompt, generation_config=transformers.GenerationConfig(
                do_sample=False, eos_token_id=None, max_new_tokens=4, pad_token_id=0, use_cache=False,
            ),
        )
    for runner in runners:
        tokens = prompt
        for _ in range(4):
            result = runner.Execute(inputs={
                "attention_mask": torch.ones_like(tokens),
                "input_ids": tokens,
                "position_ids": torch.arange(tokens.shape[1]).expand(tokens.shape[0], -1),
            })
            tokens = torch.cat([tokens, result["logits"][:, -1].argmax(dim=-1, keepdim=True)], dim=1)
        assert torch.equal(tokens, expected_tokens)


def test_checkpoint_architecture_must_match(*, workspace):
    Configuration(model_name=fs_selection.ModelName.MISTRAL).save_pretrained(workspace)
    with pytest.raises(ValueError, match="does not match"):
        fs_selection.GetModelExperiment(
            model_name=fs_selection.ModelName.MIXTRAL, experiment_name=fs_selection.ExperimentName.BASELINE,
            checkpoint=str(workspace), revision="local",
            dtype=torch.bfloat16, device=torch.device("cpu"), attention="eager",
        )


def test_incomplete_checkpoint_is_rejected(*, workspace):
    model = REFERENCE_CLASSES[fs_selection.ModelName.MISTRAL](
        config=Configuration(model_name=fs_selection.ModelName.MISTRAL),
    )
    model.save_pretrained(workspace)
    path = workspace / "model.safetensors"
    weights = safetensors.torch.load_file(path)
    del weights["model.layers.0.self_attn.q_proj.weight"]
    safetensors.torch.save_file(weights, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="did not load completely"):
        fs_selection.GetModelExperiment(
            model_name=fs_selection.ModelName.MISTRAL, experiment_name=fs_selection.ExperimentName.BASELINE,
            checkpoint=str(workspace), revision="local",
            dtype=torch.bfloat16, device=torch.device("cpu"), attention="eager",
        )


@pytest.mark.parametrize("model_name, experiment_name", [
    ("unknown", fs_selection.ExperimentName.BASELINE),
    (fs_selection.ModelName.MISTRAL, "unknown"),
])
def test_unknown_experiment_is_rejected(*, model_name, experiment_name):
    with pytest.raises(ValueError, match="Unknown model experiment"):
        fs_selection.GetModelExperiment(
            model_name=model_name, experiment_name=experiment_name, checkpoint="unused", revision="unused",
            dtype=torch.bfloat16, device=torch.device("cpu"), attention="eager",
        )


@pytest.mark.parametrize("invalid", ["device", "dtype", "shape", "missing", "control"])
def test_model_rejects_invalid_inputs(*, workspace, invalid):
    original = transformers.MistralForCausalLM(config=Configuration(model_name=fs_selection.ModelName.MISTRAL))
    original.save_pretrained(workspace)
    model = fs_selection.GetModelExperiment(
        model_name=fs_selection.ModelName.MISTRAL, experiment_name=fs_selection.ExperimentName.HUGGING_FACE,
        checkpoint=str(workspace), revision="local", dtype=torch.bfloat16,
        device=torch.device("cpu"), attention="eager",
    )
    inputs = {
        "attention_mask": torch.ones((1, 3), dtype=torch.int64),
        "input_ids": torch.tensor([[1, 4, 7]]),
        "position_ids": torch.arange(3).unsqueeze(0),
    }
    if invalid == "device":
        inputs["input_ids"] = inputs["input_ids"].to(device="meta")
    elif invalid == "dtype":
        inputs["input_ids"] = inputs["input_ids"].float()
    elif invalid == "shape":
        inputs["attention_mask"] = torch.ones((1, 2), dtype=torch.int64)
    elif invalid == "missing":
        del inputs["position_ids"]
    else:
        inputs["use_cache"] = True
    with pytest.raises(ValueError):
        model.Execute(inputs=inputs)
