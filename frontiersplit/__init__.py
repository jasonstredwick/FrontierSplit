"""FrontierSplit: Distributed LLM Inference Engine.

Subpackages:
- `frontiersplit.context`: Disaggregated Long-Context Serving using Online Softmax RPCs.
- `frontiersplit.pipeline`: 1F1B Distributed Pipeline Parallel Slicing (TPU / multi-GPU).
- `frontiersplit.gateway`: Unified OpenAI-compatible API Ingress.
"""

from frontiersplit.context import (
    ConsistentHashRing,
    ContextClient,
    ContextServer,
    ContextShardNode,
    ContextStore,
    DecodeWorker,
    DisaggregatedAttention,
    DisaggregatedAttentionPatcher,
    DisaggregatedModel,
    LocalOutputKVCache,
    PartialAttentionChunk,
    PrefixRouter,
    RadixNode,
    RadixPrefixCache,
    ShardRoutingDecision,
    compute_partial_attention,
    finalize_attention,
    merge_partial_attentions,
    merge_two_partial_attentions,
    repeat_kv,
)
from frontiersplit.gateway import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    create_gateway_app,
)
from frontiersplit.pipeline import (
    ActivationPacket,
    BatchedActivationPacket,
    BinaryTransportClient,
    BinaryTransportServer,
    ModelSlicer,
    ModelSpec,
    PipelineScheduler,
    ReleaseSessionPacket,
    ScheduledRequest,
    ShardPlanner,
    WorkerKVCacheStore,
    create_worker_app,
    resolve_model_spec,
)

__version__ = "0.2.0.dev0"

__all__ = [
    # Context Engine
    "ConsistentHashRing",
    "ContextClient",
    "ContextServer",
    "ContextShardNode",
    "ContextStore",
    "DecodeWorker",
    "DisaggregatedAttention",
    "DisaggregatedAttentionPatcher",
    "DisaggregatedModel",
    "LocalOutputKVCache",
    "PartialAttentionChunk",
    "PrefixRouter",
    "RadixNode",
    "RadixPrefixCache",
    "ShardRoutingDecision",
    "compute_partial_attention",
    "finalize_attention",
    "merge_partial_attentions",
    "merge_two_partial_attentions",
    "repeat_kv",
    # Gateway
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "create_gateway_app",
    # Pipeline Engine
    "ActivationPacket",
    "BatchedActivationPacket",
    "BinaryTransportClient",
    "BinaryTransportServer",
    "ModelSlicer",
    "ModelSpec",
    "PipelineScheduler",
    "ReleaseSessionPacket",
    "ScheduledRequest",
    "ShardPlanner",
    "WorkerKVCacheStore",
    "create_worker_app",
    "resolve_model_spec",
]
