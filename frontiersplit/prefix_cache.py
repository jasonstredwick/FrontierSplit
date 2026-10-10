"""Radix Tree Prefix Cache for FrontierSplit Context Server.

Implements a compressed trie (Radix Tree) that stores prompt Key-Value (KV)
caches indexed by token sequence IDs. Allows arbitrary queries sharing common
prefixes (system prompts, document context, multi-turn chat history) to reuse
precomputed KV cache on the Context Server with zero redundant prefill compute.
"""

from __future__ import annotations

import time

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

        suffix_k = {
            layer: k[..., split_idx:, :].contiguous()
            for layer, k in self.k_cache.items()
        }
        suffix_v = {
            layer: v[..., split_idx:, :].contiguous()
            for layer, v in self.v_cache.items()
        }

        prefix_k = {
            layer: k[..., :split_idx, :].contiguous()
            for layer, k in self.k_cache.items()
        }
        prefix_v = {
            layer: v[..., :split_idx, :].contiguous()
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

    def match(self, tokens: list[int] | tuple[int, ...]) -> tuple[list[RadixNode], int]:
        """Finds the longest cached prefix matching the query token sequence.

        If a match ends midway through a node, that node is automatically split
        so that the exact matched prefix is represented as a clean node.

        Args:
            tokens: Sequence of token IDs to match.

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
                    node.k_cache[layer] = k[
                        ..., offset : offset + node_len, :
                    ].contiguous()
                    node.v_cache[layer] = v_by_layer[layer][
                        ..., offset : offset + node_len, :
                    ].contiguous()
                offset += node_len

        if matched_len == len(seq):
            return nodes

        curr = nodes[-1] if nodes else self.root
        rem_tokens = seq[matched_len:]

        rem_k = {
            layer: k[..., matched_len:, :].contiguous()
            for layer, k in k_by_layer.items()
        }
        rem_v = {
            layer: v[..., matched_len:, :].contiguous()
            for layer, v in v_by_layer.items()
        }

        new_node = RadixNode(rem_tokens, rem_k, rem_v)
        curr.children[rem_tokens[0]] = new_node
        nodes.append(new_node)

        self.evict_if_needed()
        return nodes

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
        """Evicts least recently used (LRU) leaf nodes if token count exceeds quota.

        Only leaf nodes with `session_count == 0` (unpinned) are eligible for eviction.

        Returns:
            Number of tokens evicted.
        """
        evicted_tokens = 0
        while self.total_tokens > self.max_cached_tokens:
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

        layers = list(nodes[0].k_cache.keys())
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
