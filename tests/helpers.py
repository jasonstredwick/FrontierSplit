import dataclasses
import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.processors
import transformers


DEFAULT_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if loop.index0 > 0 and message['role'] == messages[loop.index0 - 1]['role'] %}"
    "{{ raise_exception('Roles must alternate.') }}"
    "{%- endif %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}<|end|>"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>{%- endif %}"
)

OPENAI_CONTENT_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "<|{{ message['role'] }}|>"
    "{%- for part in message['content'] %}[{{ part['type'] }}:{{ part['text'] }}]{%- endfor %}"
    "<|end|>"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>{%- endif %}"
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenizerBundle:
    alt_world_id: int
    bos_id: int
    eos_id: int
    euro_byte_ids: tuple[int, int, int]
    hello_id: int
    stop_word_id: int
    tokenizer: transformers.PreTrainedTokenizerFast
    vocab_size: int
    world_id: int


def BuildTestTokenizer(
    *,
    chat_template: str | dict[str, str] | None = DEFAULT_CHAT_TEMPLATE,
) -> TokenizerBundle:
    entries: list[tuple[str, float]] = [
        ("<unk>", 0.0),
        ("<s>", 0.0),
        ("</s>", 0.0),
    ]
    for byte_val in range(256):
        entries.append((f"<0x{byte_val:02X}>", -20.0))

    lexical_pieces = [
        "▁",
        "▁Hello",
        "Hello",
        "▁world",
        "world",
        "▁stop",
        "stop",
        "▁END",
        "END",
        "▁assistant",
        "▁user",
        "▁system",
        "!",
        ".",
        ":",
        "\n",
        "[",
        "]",
        "<|",
        "|>",
        "system",
        "user",
        "assistant",
        "end",
        "text",
        "a",
        "b",
        "c",
        "1",
        "2",
        "3",
    ]
    for piece in lexical_pieces:
        entries.append((piece, -1.0))

    raw_tok = tokenizers.Tokenizer(
        tokenizers.models.Unigram(entries, unk_id=0, byte_fallback=True)
    )
    raw_tok.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace(replacement="▁")
    raw_tok.decoder = tokenizers.decoders.Sequence(
        [
            tokenizers.decoders.ByteFallback(),
            tokenizers.decoders.Metaspace(replacement="▁"),
        ]
    )
    raw_tok.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", 1)],
    )
    # Add a distinct non-special token that also decodes to " world" to exercise top_logprobs collisions.
    raw_tok.add_tokens([tokenizers.AddedToken(" world", special=False)])

    fast_tok = transformers.PreTrainedTokenizerFast(
        bos_token="<s>",
        eos_token="</s>",
        tokenizer_object=raw_tok,
        unk_token="<unk>",
    )
    fast_tok.chat_template = chat_template

    vocab = fast_tok.get_vocab()
    return TokenizerBundle(
        alt_world_id=vocab[" world"],
        bos_id=vocab["<s>"],
        eos_id=vocab["</s>"],
        euro_byte_ids=(vocab["<0xE2>"], vocab["<0x82>"], vocab["<0xAC>"]),
        hello_id=vocab["▁Hello"],
        stop_word_id=vocab["▁END"],
        tokenizer=fast_tok,
        vocab_size=len(vocab),
        world_id=vocab["▁world"],
    )
