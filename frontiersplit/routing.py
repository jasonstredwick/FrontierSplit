"""FrontierSplit Prefix-Aware Intelligent Router and Consistent Hash Ring.

Routes incoming inference requests to the optimal Context Server shard using:
1. Explicit Session Affinity (for multi-turn conversational agents with session IDs).
2. Hierarchical Prefix Fingerprint Hashing (for automatic shared-prefix cache hits).
3. Consistent Hashing with Virtual Nodes for even distribution and fault-tolerant failover.
"""

from __future__ import annotations

import bisect
import hashlib
import struct
from typing import Any

from pydantic import BaseModel, Field

from frontiersplit.context_server import ContextClient


class ContextShardNode(BaseModel):
    """Represents a physical or containerized Context Server instance."""

    shard_id: str
    host: str
    port: int
    weight: int = 1
    is_healthy: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    _client: ContextClient | None = None

    model_config = {"arbitrary_types_allowed": True}

    def get_client(self) -> ContextClient:
        """Returns or lazily creates a persistent ContextClient for this shard."""
        if self._client is None:
            self._client = ContextClient(host=self.host, port=self.port)
        return self._client

    def close(self) -> None:
        """Closes active client connections."""
        if self._client is not None:
            self._client.close_sync()
            self._client = None


class ShardRoutingDecision(BaseModel):
    """Routing resolution for an incoming request."""

    shard: ContextShardNode
    replicas: list[ContextShardNode] = Field(default_factory=list)
    route_type: str  # 'session', 'prefix', or 'fallback'
    fingerprint: str
    token_count: int = 0


def hash_key_to_uint32(key: str | bytes) -> int:
    """Computes a deterministic 32-bit unsigned integer hash using BLAKE2b."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return struct.unpack(">I", digest)[0]


class ConsistentHashRing:
    """Consistent Hash Ring with virtual nodes for shard distribution and failover."""

    def __init__(self, vnodes_per_shard: int = 128) -> None:
        """Initializes an empty consistent hash ring.

        Args:
            vnodes_per_shard: Number of virtual node points per physical shard.
        """
        self.vnodes_per_shard = max(1, vnodes_per_shard)
        self.shards: dict[str, ContextShardNode] = {}
        self._ring: list[tuple[int, str]] = []  # sorted list of (hash_val, shard_id)

    def add_shard(self, shard: ContextShardNode) -> None:
        """Adds a shard and distributes its virtual nodes across the ring."""
        self.shards[shard.shard_id] = shard
        num_vnodes = self.vnodes_per_shard * max(1, shard.weight)
        for i in range(num_vnodes):
            vnode_key = f"{shard.shard_id}#vn{i}"
            h = hash_key_to_uint32(vnode_key)
            bisect.insort(self._ring, (h, shard.shard_id))

    def remove_shard(self, shard_id: str) -> None:
        """Removes a shard and all its virtual nodes from the ring."""
        if shard_id in self.shards:
            self.shards[shard_id].close()
            del self.shards[shard_id]
            self._ring = [(h, s_id) for h, s_id in self._ring if s_id != shard_id]

    def set_shard_health(self, shard_id: str, is_healthy: bool) -> None:
        """Updates the health status of a shard."""
        if shard_id in self.shards:
            self.shards[shard_id].is_healthy = is_healthy

    def get_node(self, key: str | bytes) -> ContextShardNode:
        """Resolves the primary healthy ContextShardNode for a key."""
        if not self.shards:
            raise RuntimeError("ConsistentHashRing is empty: no shards configured")

        healthy_shards = [s for s in self.shards.values() if s.is_healthy]
        if not healthy_shards:
            raise RuntimeError("All Context Server shards are marked unhealthy")

        h = hash_key_to_uint32(key)
        # Binary search for the first virtual node >= h
        idx = bisect.bisect_right(self._ring, (h, ""))

        # Search clockwise for the first healthy shard
        n = len(self._ring)
        for step in range(n):
            ring_idx = (idx + step) % n
            _, target_shard_id = self._ring[ring_idx]
            target_shard = self.shards[target_shard_id]
            if target_shard.is_healthy:
                return target_shard

        return healthy_shards[0]

    def get_replicas(self, key: str | bytes, count: int = 2) -> list[ContextShardNode]:
        """Resolves the primary node and successor replica nodes along the ring."""
        if not self.shards:
            raise RuntimeError("ConsistentHashRing is empty")

        h = hash_key_to_uint32(key)
        idx = bisect.bisect_right(self._ring, (h, ""))

        seen_shard_ids: set[str] = set()
        replicas: list[ContextShardNode] = []
        n = len(self._ring)

        for step in range(n):
            ring_idx = (idx + step) % n
            _, s_id = self._ring[ring_idx]
            if s_id not in seen_shard_ids:
                seen_shard_ids.add(s_id)
                shard = self.shards[s_id]
                if shard.is_healthy:
                    replicas.append(shard)
                    if len(replicas) >= count:
                        break

        return replicas


class PrefixRouter:
    """Prefix-aware intelligent router mapping chat requests to Context Shards."""

    def __init__(
        self,
        ring: ConsistentHashRing | None = None,
        prefix_window_tokens: int = 64,
    ) -> None:
        """Initializes the PrefixRouter.

        Args:
            ring: Optional configured ConsistentHashRing instance.
            prefix_window_tokens: Leading tokens to include in the prefix fingerprint.
        """
        self.ring = ring if ring is not None else ConsistentHashRing()
        self.prefix_window_tokens = prefix_window_tokens

    def add_shard(
        self, shard_id: str, host: str, port: int, weight: int = 1
    ) -> ContextShardNode:
        """Adds a Context Server shard to the router's hash ring."""
        node = ContextShardNode(
            shard_id=shard_id,
            host=host,
            port=port,
            weight=weight,
            is_healthy=True,
        )
        self.ring.add_shard(node)
        return node

    def remove_shard(self, shard_id: str) -> None:
        """Removes a Context Server shard from the router."""
        self.ring.remove_shard(shard_id)

    def route_request(
        self,
        messages: list[Any],
        session_id: str | None = None,
        tokenizer: Any | None = None,
    ) -> ShardRoutingDecision:
        """Routes an incoming chat request to the optimal Context Server shard.

        Routing Logic:
        1. If session_id is provided, routes via explicit session affinity.
        2. Validates non-empty message list (raises ValueError if messages is empty).
        3. Extracts initial prompt prefix (up to prefix_window_tokens) and computes fingerprint.
        4. Queries ConsistentHashRing to select primary shard and backup replicas.

        Args:
            messages: List of message dicts or objects with 'role' and 'content'.
            session_id: Optional explicit session ID from client header.
            tokenizer: Optional tokenizer for token-level prefix extraction.

        Returns:
            ShardRoutingDecision specifying the target shard, replicas, and fingerprint.
        """
        # 1. Explicit Session Affinity (Overrides Prefix Fingerprinting)
        if session_id:
            key = f"session:{session_id}"
            primary = self.ring.get_node(key)
            replicas = self.ring.get_replicas(key, count=2)
            backup_replicas = [r for r in replicas if r.shard_id != primary.shard_id]
            return ShardRoutingDecision(
                shard=primary,
                replicas=backup_replicas,
                route_type="session",
                fingerprint=session_id,
                token_count=0,
            )

        # 2. Ingress Validation
        if not messages:
            raise ValueError("Incoming message list cannot be empty")

        # 3. Extract and Tokenize Leading Prefix
        all_text_parts: list[str] = []
        for m in messages:
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content", "")
            role = getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role", "user")
            all_text_parts.append(f"{role}: {content or ''}")

        full_prompt_text = "\n".join(all_text_parts).strip()

        # Handle empty/whitespace content gracefully
        if not full_prompt_text or full_prompt_text in ("user:", "system:"):
            fallback_key = "prefix:bos_canonical_fallback"
            primary = self.ring.get_node(fallback_key)
            replicas = self.ring.get_replicas(fallback_key, count=2)
            backup_replicas = [r for r in replicas if r.shard_id != primary.shard_id]
            return ShardRoutingDecision(
                shard=primary,
                replicas=backup_replicas,
                route_type="fallback",
                fingerprint="bos_fallback",
                token_count=1,
            )

        # Check if an explicit system message is present
        system_content = ""
        for m in messages:
            role = getattr(m, "role", None) or (
                m.get("role") if isinstance(m, dict) else None
            )
            content = getattr(m, "content", None) or (
                m.get("content") if isinstance(m, dict) else None
            )
            if role == "system" and content and content.strip():
                system_content = f"system: {content.strip()}"
                break

        prefix_source = system_content if system_content else full_prompt_text

        # Extract tokens or character window
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                tokens = tokenizer.encode(prefix_source)
                prefix_tokens = tokens[: self.prefix_window_tokens]
                fingerprint_str = ",".join(str(t) for t in prefix_tokens)
                token_count = len(prefix_tokens)
            except Exception:
                # Fallback to character prefix
                prefix_text = prefix_source[: self.prefix_window_tokens * 4]
                fingerprint_str = prefix_text
                token_count = len(prefix_text.split())
        else:
            # Tokenize by whitespace approximation
            words = prefix_source.split()
            prefix_words = words[: self.prefix_window_tokens]
            fingerprint_str = " ".join(prefix_words)
            token_count = len(prefix_words)

        prefix_key = f"prefix:{fingerprint_str}"
        primary = self.ring.get_node(prefix_key)
        replicas = self.ring.get_replicas(prefix_key, count=2)
        backup_replicas = [r for r in replicas if r.shard_id != primary.shard_id]

        return ShardRoutingDecision(
            shard=primary,
            replicas=backup_replicas,
            route_type="prefix",
            fingerprint=fingerprint_str,
            token_count=token_count,
        )

    async def query_shard_prefix(
        self,
        decision: ShardRoutingDecision,
        token_ids: list[int],
    ) -> int:
        """Queries the assigned Context Server shard over TCP for matching prefix length.

        Falls over to replica shards if the primary shard is unreachable.

        Args:
            decision: ShardRoutingDecision resolved for this request.
            token_ids: Full sequence token IDs.

        Returns:
            Length of the longest matching cached prefix tokens.
        """
        targets = [decision.shard] + decision.replicas

        for target in targets:
            try:
                client = target.get_client()
                matched_len, _ = await client.match_prefix(token_ids)
                return matched_len
            except Exception:
                # Mark as unhealthy and attempt replica
                target.is_healthy = False
                continue

        return 0
