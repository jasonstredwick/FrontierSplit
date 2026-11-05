# Adapted from Hugging Face Transformers 5.20.2.
# Upstream source: src/transformers/models/mixtral/modeling_mixtral.py
# Original file SHA-256: 96d97e94b6bccd3961fa3e0bf8e0d8b6d146eb51f8d56e1126e732ead02a6b5d

import torch
import transformers
import transformers.conversion_mapping
import transformers.initialization
import transformers.integrations.moe
import transformers.integrations.sdpa_attention
import transformers.masking_utils
import transformers.modeling_outputs


class Attention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: transformers.MixtralConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = torch.nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: transformers.Cache,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = ApplyRotaryPosEmb(q=query_states, k=key_states, cos=cos, sin=sin)

        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, _ = transformers.integrations.sdpa_attention.sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class CausalLM(transformers.PreTrainedModel):
    _supports_sdpa = True
    base_model_prefix = "model"
    config: transformers.MixtralConfig

    def __init__(self, config: transformers.MixtralConfig) -> None:
        super().__init__(config)
        self._no_split_modules = {"DecoderLayer"}
        self._skip_keys_device_placement = {"past_key_values"}
        self.model = Model(config=config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: torch.nn.Module) -> None:
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, Experts):
            transformers.initialization.normal_(module.gate_up_proj, mean=0.0, std=std)
            transformers.initialization.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, TopKRouter):
            transformers.initialization.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RotaryEmbedding):
            inv_freq, _ = module.ComputeDefaultRopeParameters(config=module.config)
            transformers.initialization.copy_(module.inv_freq, inv_freq)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: transformers.Cache | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> transformers.modeling_outputs.MoeCausalLMOutputWithPast:
        hidden_states: torch.Tensor
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return transformers.modeling_outputs.MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        was_registered = "mixtral" in transformers.conversion_mapping.USER_REGISTERED_MAPPINGS
        if not was_registered:
            transformers.conversion_mapping.USER_REGISTERED_MAPPINGS.add("mixtral")
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            if not was_registered:
                transformers.conversion_mapping.USER_REGISTERED_MAPPINGS.discard("mixtral")


class DecoderLayer(torch.nn.Module):
    def __init__(self, config: transformers.MixtralConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = SparseMoeBlock(config=config)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: transformers.Cache,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Experts(torch.nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    # Attributes read by transformers.integrations.moe.grouped_mm_experts_forward.
    _apply_gate = transformers.integrations.moe._default_apply_gate
    _is_expert_parallel = False
    has_bias = False
    has_gate = True
    has_post_expert_norm = False
    is_transposed = False

    def __init__(self, config: transformers.MixtralConfig) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = torch.nn.functional.silu

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return transformers.integrations.moe.grouped_mm_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        )


class Model(torch.nn.Module):
    def __init__(self, config: transformers.MixtralConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(config=config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: transformers.Cache | None = None,
    ) -> tuple[torch.Tensor, transformers.Cache]:
        inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = transformers.DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length()
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = transformers.masking_utils.create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                attention_mask=causal_mask,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, config: transformers.MixtralConfig) -> None:
        super().__init__()
        self.config = config
        inv_freq, self.attention_scaling = self.ComputeDefaultRopeParameters(config=config)
        self.inv_freq = torch.nn.Buffer(inv_freq, persistent=False)

    @staticmethod
    def ComputeDefaultRopeParameters(*, config: transformers.MixtralConfig) -> tuple[torch.Tensor, float]:
        rope_theta = config.rope_parameters["rope_theta"] if config.rope_parameters is not None else None
        base = rope_theta or config.default_theta
        dim = config.head_dim or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SparseMoeBlock(torch.nn.Module):
    def __init__(self, config: transformers.MixtralConfig) -> None:
        super().__init__()
        self.gate = TopKRouter(config=config)
        self.experts = Experts(config=config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class TopKRouter(torch.nn.Module):
    def __init__(self, config: transformers.MixtralConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_top_value, router_indices


def ApplyRotaryPosEmb(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (RotateHalf(x=q) * sin)
    k_embed = (k * cos) + (RotateHalf(x=k) * sin)
    return q_embed, k_embed


def RotateHalf(*, x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
