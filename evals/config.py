import pathlib
import pydantic
import re
import urllib.parse

from typing import Literal


HARNESS_VERSION: str = "0.4.13"
ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class Configuration(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)

    display_name: str = pydantic.Field(min_length=1)
    id: str = pydantic.Field(pattern=ID_PATTERN)
    schema_version: Literal[1] = 1


class Dataset(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)

    local_files: dict[str, str] | None = None
    repository: str | None = None
    revision: str | None = pydantic.Field(default=None, pattern=r"^[a-f0-9]{40}$")

    @pydantic.model_validator(mode="after")
    def ValidateSource(self):
        if self.local_files is not None:
            if not self.local_files or self.repository is not None or self.revision is not None:
                raise ValueError("Use local_files or a repository and pinned revision")
        elif self.repository is None or self.revision is None:
            raise ValueError("A dataset repository requires a pinned revision")
        return self


class Backend(Configuration):
    add_bos_token: bool = False
    chat_template: str | None = None
    engine_settings: dict = pydantic.Field(default_factory=dict)
    max_model_len: int = pydantic.Field(ge=2)
    model: str = pydantic.Field(min_length=1)
    model_revision: str = pydantic.Field(min_length=1)
    precision: Literal["bf16", "fp16"]
    software: dict[str, str]
    tokenizer: str = pydantic.Field(min_length=1)
    tokenizer_revision: str = pydantic.Field(pattern=r"^[a-f0-9]{40}$")


class Deployment(Configuration):
    hardware: dict
    launch_instructions: str
    location: str
    provider: str
    software: dict[str, str]


class Evaluation(Configuration):
    adapter: Literal["local-completions", "local-chat-completions"] = "local-completions"
    apply_chat_template: bool = False
    batch_size: int = pydantic.Field(default=1, ge=1)
    dataset: Dataset
    fewshot_as_multiturn: bool = False
    generation: dict[str, int | float | str | bool | list[str]] = pydantic.Field(default_factory=dict)
    harness_version: str = HARNESS_VERSION
    limit: int | None = pydantic.Field(default=None, ge=1)
    max_gen_toks: int = pydantic.Field(default=256, ge=1)
    max_retries: int = pydantic.Field(default=3, ge=1)
    num_concurrent: int = pydantic.Field(default=1, ge=1)
    num_fewshot: int = pydantic.Field(ge=0)
    seeds: list[int] = pydantic.Field(default_factory=lambda: [0, 1234, 1234, 1234], min_length=4, max_length=4)
    task: str = pydantic.Field(min_length=1)
    task_files: list[str] = pydantic.Field(min_length=1)
    task_version: int | float
    timeout: int = pydantic.Field(default=300, ge=1)

    @pydantic.model_validator(mode="after")
    def ValidateAdapter(self):
        if self.adapter == "local-chat-completions" and (not self.apply_chat_template or self.batch_size != 1):
            raise ValueError("Chat evaluations require apply_chat_template=true and batch_size=1")
        if self.fewshot_as_multiturn and not self.apply_chat_template:
            raise ValueError("fewshot_as_multiturn requires apply_chat_template")
        return self

    @pydantic.field_validator("harness_version")
    @classmethod
    def ValidateHarnessVersion(cls, value: str) -> str:
        if value != HARNESS_VERSION:
            raise ValueError(f"Harness version must be {HARNESS_VERSION}")
        return value


class Suite(Configuration):
    evaluations: list[str] = pydantic.Field(min_length=1)

    @pydantic.field_validator("evaluations")
    @classmethod
    def ValidateEvaluations(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(re.fullmatch(ID_PATTERN, item) is None for item in value):
            raise ValueError("Suite evaluations must be distinct configuration IDs")
        return value


class Target(Configuration):
    api_base_url: str
    api_key_env: str | None = pydantic.Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    backend: str = pydantic.Field(pattern=ID_PATTERN)
    deployment: str = pydantic.Field(pattern=ID_PATTERN)
    served_model: str = pydantic.Field(min_length=1)

    @pydantic.field_validator("api_base_url")
    @classmethod
    def ValidateURL(cls, value: str) -> str:
        url = urllib.parse.urlsplit(value)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path.rstrip("/") != "/v1"):
            raise ValueError("api_base_url must be an HTTP(S) /v1 URL without credentials, query, or fragment")
        return value.rstrip("/")


def Load[T: Configuration](*, root: pathlib.Path, kind: str, name: str, schema: type[T]) -> T:
    if re.fullmatch(ID_PATTERN, name) is None:
        raise ValueError(f"Invalid configuration ID: {name}")
    path = root / kind / f"{name}.json"
    config = schema.model_validate_json(path.read_text())
    if config.id != name:
        raise ValueError(f"Configuration ID must match filename: {path}")
    return config
