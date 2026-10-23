"""Radix Tree Prefix Cache for FrontierSplit Context Server.

Implements a compressed trie (Radix Tree) that stores prompt Key-Value (KV)
caches indexed by token sequence IDs. Allows arbitrary queries sharing common
prefixes (system prompts, document context, multi-turn chat history) to reuse
precomputed KV cache on the Context Server with zero redundant prefill compute.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch


class RadixNode:
    """A node in the Radix prefix tree representing a contiguous sequence of tokens."""

    def __init__(
        self,
        token_ids: tuple[int, ...] = (),
        k_cache: dict[int, torch.Tensor] | None = None,
        v_cache: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Initializes a Radix tree node.

        Args:
            token_ids: Tuple of token IDs stored on this edge/node.
            k_cache: Dictionary mapping layer_idx to Key tensor of shape
                `[batch, num_kv_heads, len(token_ids), head_dim]`.
            v_cache: Dictionary mapping layer_idx to Value tensor of shape
                `[batch, num_kv_heads, len(token_ids), head_dim]`.
        """
        self.token_ids = tuple(token_ids)
        self.k_cache: dict[int, torch.Tensor] = k_cache or {}
        self.v_cache: dict[int, torch.Tensor] = v_cache or {}
        self.children: dict[int, RadixNode] = {}  # first_token -> RadixNode
        self.last_access: float = time.time()
        self.session_count: int = 0

    @property
    def num_tokens(self) -> int:
        """Returns the number of tokens represented by this node."""
        return len(self.token_ids)

    def compact(self) -> bool:
        """Selectively compactor to isolate tensor storage when backing storage exceeds logical slice.

        Returns:
            True if compaction occurred (releasing a larger underlying slab), False otherwise.
        """
        compacted = False
        for layer, k in list(self.k_cache.items()):
            if k is not None and k.untyped_storage().nbytes() > (
                k.numel() * k.element_size()
            ):
                self.k_cache[layer] = k.clone()
                compacted = True
        for layer, v in list(self.v_cache.items()):
            if v is not None and v.untyped_storage().nbytes() > (
                v.numel() * v.element_size()
            ):
                self.v_cache[layer] = v.clone()
                compacted = True
        return compacted

    def split(self, split_idx: int) -> RadixNode:
        """Splits this node at `split_idx` into a prefix parent and a suffix child.

        Args:
            split_idx: Slicing index where `0 < split_idx < len(self.token_ids)`.

        Returns:
            The newly created child RadixNode holding the suffix tokens and KV slices.
        """
        if not (0 < split_idx < len(self.token_ids)):
            raise ValueError(
                f"Split index {split_idx} out of range (0, {len(self.token_ids)})"
            )

        suffix_tokens = self.token_ids[split_idx:]
        prefix_tokens = self.token_ids[:split_idx]

        # Preserve canonical immutable tensor storage via slice views (zero new memory allocation)
        suffix_k = {
            layer: k[..., split_idx:, :]
            for layer, k in self.k_cache.items()
        }
        suffix_v = {
            layer: v[..., split_idx:, :]
            for layer, v in self.v_cache.items()
        }

        prefix_k = {
            layer: k[..., :split_idx, :]
            for layer, k in self.k_cache.items()
        }
        prefix_v = {
            layer: v[..., :split_idx, :]
            for layer, v in self.v_cache.items()
        }

        child = RadixNode(suffix_tokens, suffix_k, suffix_v)
        child.children = self.children
        child.session_count = self.session_count
        child.last_access = self.last_access

        self.token_ids = prefix_tokens
        self.k_cache = prefix_k
        self.v_cache = prefix_v
        self.children = {child.token_ids[0]: child}

        return child


class RadixPrefixCache:
    """Compressed trie managing token-prefix KV caches with LRU eviction."""

    def __init__(self, max_cached_tokens: int = 1_000_000) -> None:
        """Initializes the Radix Prefix Cache.

        Args:
            max_cached_tokens: Maximum total tokens to retain across all cached prefixes.
        """
        self.root = RadixNode()
        self.max_cached_tokens = max_cached_tokens
        self.on_compact: Callable[[], None] | None = None

    @property
    def total_tokens(self) -> int:
        """Calculates total number of tokens stored across all nodes in the tree."""
        count = 0

        def _traverse(node: RadixNode) -> None:
            nonlocal count
            count += node.num_tokens
            for child in node.children.values():
                _traverse(child)

        for child in self.root.children.values():
            _traverse(child)

        return count

    @property
    def total_backing_tokens(self) -> int:
        """Calculates total number of sequence tokens physically allocated in unique backing tensor storages.

        Normalizes backing sequence capacity per layer to compare consistently against
        the sequence-token budget, rather than summing across all model layers.
        """
        seen_ptrs_by_layer: dict[int, set[int]] = {}
        layer_backing_tokens: dict[int, int] = {}

        def _traverse(node: RadixNode) -> None:
            for l_idx, t in node.k_cache.items():
                if t is not None:
                    seen = seen_ptrs_by_layer.setdefault(l_idx, set())
                    ptr = t.untyped_storage().data_ptr()
                    if ptr not in seen:
                        seen.add(ptr)
                        token_dim = t.shape[-2]
                        logical_bytes = t.numel() * t.element_size()
                        storage_bytes = t.untyped_storage().nbytes()
                        if logical_bytes > 0:
                            backing_tokens = int(
                                token_dim * (storage_bytes / logical_bytes)
                            )
                            layer_backing_tokens[l_idx] = (
                                layer_backing_tokens.get(l_idx, 0) + backing_tokens
                            )
            for child in node.children.values():
                _traverse(child)

        for child in self.root.children.values():
            _traverse(child)

        if not layer_backing_tokens:
            return 0
        return max(layer_backing_tokens.values())

    def compact_all(self, node: RadixNode | None = None) -> int:
        """Compacts all nodes in the tree whose backing allocation exceeds logical shape.

        Returns:
            Number of nodes compacted.
        """
        is_top_level = node is None or node is self.root
        target = node or self.root
        compacted_count = 1 if target.compact() else 0
        for child in list(target.children.values()):
            compacted_count += self._compact_subtree(child)
        if is_top_level and compacted_count > 0 and self.on_compact is not None:
            self.on_compact()
        return compacted_count

    def _compact_subtree(self, node: RadixNode) -> int:
        count = 1 if node.compact() else 0
        for child in list(node.children.values()):
            count += self._compact_subtree(child)
        return count

    def match(
        self,
        tokens: list[int] | tuple[int, ...],
        required_layers: set[int] | None = None,
    ) -> tuple[list[RadixNode], int]:
        """Finds the longest cached prefix matching the query token sequence.

        If a match ends midway through a node, that node is automatically split
        so that the exact matched prefix is represented as a clean node.

        Args:
            tokens: Sequence of token IDs to match.
            required_layers: Optional set of layer indices that a node must contain
                to be considered a valid complete match.

        Returns:
            Tuple of (matched_node_chain, total_matched_tokens_length).
        """
        seq = tuple(tokens)
        curr = self.root
        matched_len = 0
        nodes: list[RadixNode] = []

        while matched_len < len(seq):
            first_tok = seq[matched_len]
            if first_tok not in curr.children:
                break

            child = curr.children[first_tok]
            if required_layers and not (
                required_layers.issubset(child.k_cache.keys())
                and required_layers.issubset(child.v_cache.keys())
            ):
                break
            rem_tokens = seq[matched_len:]

            common_len = 0
            for a, b in zip(rem_tokens, child.token_ids, strict=False):
                if a != b:
                    break
                common_len += 1

            if common_len == child.num_tokens:
                # Full match of this child node
                nodes.append(child)
                matched_len += common_len
                curr = child
                curr.last_access = time.time()
            elif common_len > 0:
                # Partial match inside this child node -> split it!
                child.split(common_len)
                nodes.append(child)
                matched_len += common_len
                curr = child
                curr.last_access = time.time()
                break
            else:
                break

        return nodes, matched_len

    def insert(
        self,
        tokens: list[int] | tuple[int, ...],
        k_by_layer: dict[int, torch.Tensor],
        v_by_layer: dict[int, torch.Tensor],
    ) -> list[RadixNode]:
        """Inserts a token sequence and its KV caches into the Radix Tree.

        Reuses any existing prefix matches and only allocates new nodes
        for newly seen suffix tokens.

        Args:
            tokens: Full token ID sequence.
            k_by_layer: Mapping of layer_idx to Key tensor of shape
                `[batch, num_kv_heads, len(tokens), head_dim]`.
            v_by_layer: Mapping of layer_idx to Value tensor of shape
                `[batch, num_kv_heads, len(tokens), head_dim]`.

        Returns:
            List of RadixNodes representing the complete token sequence.
        """
        seq = tuple(tokens)
        nodes, matched_len = self.match(seq)
        if matched_len > 0:
            offset = 0
            for node in nodes:
                node_len = node.num_tokens
                for layer, k in k_by_layer.items():
                    if layer not in node.k_cache:
                        node.k_cache[layer] = (
                            k[..., offset : offset + node_len, :].contiguous().clone()
                        )
                    if layer not in node.v_cache:
                        node.v_cache[layer] = (
                            v_by_layer[layer][..., offset : offset + node_len, :]
                            .contiguous()
                            .clone()
                        )
                offset += node_len

        if matched_len == len(seq):
            return nodes

        curr = nodes[-1] if nodes else self.root
        rem_tokens = seq[matched_len:]

        rem_k = {
            layer: k[..., matched_len:, :].contiguous().clone()
            for layer, k in k_by_layer.items()
        }
        rem_v = {
            layer: v[..., matched_len:, :].contiguous().clone()
            for layer, v in v_by_layer.items()
        }

        new_node = RadixNode(rem_tokens, rem_k, rem_v)
        curr.children[rem_tokens[0]] = new_node
        nodes.append(new_node)

        self.evict_if_needed()
        return nodes

    def insert_delta(
        self,
        base_tokens: list[int] | tuple[int, ...],
        suffix_tokens: list[int] | tuple[int, ...],
        k_delta_by_layer: dict[int, torch.Tensor],
        v_delta_by_layer: dict[int, torch.Tensor],
    ) -> list[RadixNode]:
        """Extends an existing cached prefix with suffix tokens and delta KV slices.

        Avoids full-history concatenation by directly attaching suffix nodes
        under the matched base prefix node and preserving existing canonical
        prefix tensors.

        Args:
            base_tokens: Base prefix token sequence already in the cache.
            suffix_tokens: Suffix token IDs being appended.
            k_delta_by_layer: Mapping of layer_idx to delta Key tensor of shape
                `[batch, num_kv_heads, len(suffix_tokens), head_dim]`.
            v_delta_by_layer: Mapping of layer_idx to delta Value tensor of shape
                `[batch, num_kv_heads, len(suffix_tokens), head_dim]`.

        Returns:
            List of RadixNodes representing the complete token sequence.
        """
        full_seq = tuple(base_tokens) + tuple(suffix_tokens)
        nodes, matched_len = self.match(full_seq)

        if matched_len == len(full_seq):
            return nodes

        base_len = len(base_tokens)
        curr = nodes[-1] if nodes else self.root

        if matched_len >= base_len:
            delta_offset = matched_len - base_len
            rem_tokens = full_seq[matched_len:]
            if rem_tokens:
                rem_k = {
                    l_i: k[..., delta_offset:, :].contiguous().clone()
                    for l_i, k in k_delta_by_layer.items()
                }
                rem_v = {
                    l_i: v[..., delta_offset:, :].contiguous().clone()
                    for l_i, v in v_delta_by_layer.items()
                }
                new_node = RadixNode(rem_tokens, rem_k, rem_v)
                curr.children[rem_tokens[0]] = new_node
                nodes.append(new_node)
                self.evict_if_needed()
            return nodes

        # If base prefix was evicted from the Radix tree, NEVER attach suffix to curr or root!
        # Those KV tensors were computed with the missing preceding tokens and their original positions,
        # so publishing the suffix as an orphaned prefix would corrupt the cache for unrelated queries.
        return None

    def _find_unpinned_leaves(
        self,
        parent: RadixNode,
        node: RadixNode,
        out: list[tuple[float, RadixNode, RadixNode]],
    ) -> None:
        """Helper to collect unpinned leaf nodes in the tree."""
        if not node.children and node.session_count == 0:
            out.append((node.last_access, parent, node))
        for child in node.children.values():
            self._find_unpinned_leaves(node, child, out)

    def evict_if_needed(self) -> int:
        """Evicts least recently used (LRU) leaf nodes if token count or backing storage exceeds quota.

        Only leaf nodes with `session_count == 0` (unpinned) are eligible for eviction.

        Returns:
            Number of tokens evicted.
        """
        evicted_tokens = 0
        while (
            self.total_tokens > self.max_cached_tokens
            or self.total_backing_tokens > self.max_cached_tokens
        ):
            candidates: list[tuple[float, RadixNode, RadixNode]] = []
            for child in self.root.children.values():
                self._find_unpinned_leaves(self.root, child, candidates)

            if not candidates:
                # No unpinned leaves eligible for eviction
                break

            candidates.sort(key=lambda item: item[0])
            _, parent, victim = candidates[0]
            del parent.children[victim.token_ids[0]]
            evicted_tokens += victim.num_tokens

            # Selective compaction on eviction: compact surviving nodes to release dead backing slabs
            self.compact_all()

        return evicted_tokens

    @staticmethod
    def assemble_kv(
        nodes: list[RadixNode],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Assembles concatenated Key and Value tensors for all layers from a node chain.

        Args:
            nodes: Sequential chain of matched RadixNodes.

        Returns:
            Tuple of (k_by_layer, v_by_layer) concatenated along the sequence dimension (-2).
        """
        if not nodes:
            return {}, {}

        layers = [
            l_idx
            for l_idx in nodes[0].k_cache
            if all(l_idx in node.k_cache and l_idx in node.v_cache for node in nodes)
        ]
        k_out: dict[int, torch.Tensor] = {}
        v_out: dict[int, torch.Tensor] = {}

        for l_idx in layers:
            k_out[l_idx] = torch.cat([node.k_cache[l_idx] for node in nodes], dim=-2)
            v_out[l_idx] = torch.cat([node.v_cache[l_idx] for node in nodes], dim=-2)

        return k_out, v_out


__all__ = [
    "RadixNode",
    "RadixPrefixCache",
]
