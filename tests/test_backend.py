import pytest

from frontiersplit.backend import types as fs_backend_types


def test_model_config_validation() -> None:
    with pytest.raises(ValueError, match="model_id"):
        fs_backend_types.ModelConfig(max_model_len=16, model_id="", vocab_size=32)
    with pytest.raises(ValueError, match="max_model_len"):
        fs_backend_types.ModelConfig(max_model_len=0, model_id="m", vocab_size=32)
    with pytest.raises(ValueError, match="vocab_size"):
        fs_backend_types.ModelConfig(max_model_len=16, model_id="m", vocab_size=0)
