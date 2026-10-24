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
        self.ragged_children: dict[tuple[int, ...], RadixNode] = {}
        self.last_access: float = time.time()
        self.session_count: int = 0

    def iter_all_children(self) -> list[RadixNode]:
        """Returns all child nodes (both primary children and preserved ragged children)."""
        out = list(self.children.values())
        seen = {id(n) for n in out}
        for rc in self.ragged_children.values():
            if id(rc) not in seen:
                out.append(rc)
                seen.add(id(rc))
        return out

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
        child.ragged_children = self.ragged_children
        child.session_count = self.session_count
        child.last_access = self.last_access

        self.token_ids = prefix_tokens
        self.k_cache = prefix_k
        self.v_cache = prefix_v
        self.children = {child.token_ids[0]: child}
        self.ragged_children = {}

        return child


class RadixPrefixCache:
    """Compressed trie managing token-prefix KV caches with LRU eviction."""

    def __init__(
        self,
        max_cached_tokens: int = 1_000_000,
        chunk_size: int = 128,
    ) -> None:
        """Initializes the Radix Prefix Cache.

        Args:
            max_cached_tokens: Maximum total tokens to retain across all cached prefixes.
            chunk_size: Target token chunk alignment size (e.g., 128). Ragged prompt
                edges smaller than `chunk_size` are preserved for prompt-boundary
                branches while output continuations coalesce starting at the ragged edge.
        """
        self.root = RadixNode()
        self.max_cached_tokens = max_cached_tokens
        self.chunk_size = chunk_size
        self.on_compact: Callable[[], None] | None = None

    @staticmethod
    def _is_prefix_shadowed(parent: RadixNode, node: RadixNode) -> bool:
        """Returns True if `node` is a leaf ragged-edge snapshot shadowed by a longer coalesced sibling."""
        if node.children or node.ragged_children:
            return False
        n_len = node.num_tokens
        for other in parent.iter_all_children():
            if (
                other is not node
                and other.num_tokens > n_len
                and other.token_ids[:n_len] == node.token_ids
            ):
                return True
        return False

    @property
    def total_tokens(self) -> int:
        """Calculates total number of tokens stored across all nodes in the tree."""
        count = 0

        def _traverse(parent: RadixNode, node: RadixNode) -> None:
            nonlocal count
            if not self._is_prefix_shadowed(parent, node):
                count += node.num_tokens
            for child in node.iter_all_children():
                _traverse(node, child)

        for child in self.root.iter_all_children():
            _traverse(self.root, child)

        return count

    @property
    def total_backing_tokens(self) -> int:
        """Calculates total number of sequence tokens physically allocated in unique backing tensor storages.

        Normalizes backing sequence capacity per layer to compare consistently against
        the sequence-token budget, rather than summing across all model layers.
        """
        seen_ptrs_by_layer: dict[int, set[int]] = {}
        layer_backing_tokens: dict[int, int] = {}

        def _traverse(parent: RadixNode, node: RadixNode) -> None:
            if not self._is_prefix_shadowed(parent, node):
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
            for child in node.iter_all_children():
                _traverse(node, child)

        for child in self.root.iter_all_children():
            _traverse(self.root, child)

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
        for child in target.iter_all_children():
            compacted_count += self._compact_subtree(child)
        if is_top_level and compacted_count > 0 and self.on_compact is not None:
            self.on_compact()
        return compacted_count

    def _compact_subtree(self, node: RadixNode) -> int:
        count = 1 if node.compact() else 0
        for child in node.iter_all_children():
            count += self._compact_subtree(child)
        return count

    def match(
        self,
        tokens: list[int] | tuple[int, ...],
        required_layers: set[int] | None = None,
    ) -> tuple[list[RadixNode], int]:
        """Finds the longest cached prefix matching the query token sequence.

        If a match ends midway through a node, that node is automatically split
        so that the exact matched prefix is represented as a clean node. When both
        a standalone ragged-edge node and a coalesced output chunk exist under the
        same parent, selects the coalesced chunk for output continuations and the
        standalone ragged edge when branching at the prompt boundary.

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
            candidates: list[RadixNode] = []
            primary = curr.children.get(first_tok)
            if primary is not None:
                candidates.append(primary)
            for rc in curr.ragged_children.values():
                if rc.token_ids and rc.token_ids[0] == first_tok and rc is not primary:
                    candidates.append(rc)

            if required_layers:
                candidates = [
                    c
                    for c in candidates
                    if required_layers.issubset(c.k_cache.keys())
                    and required_layers.issubset(c.v_cache.keys())
                ]

            if not candidates:
                break

            rem_tokens = seq[matched_len:]
            best_child: RadixNode | None = None
            best_common = 0
            best_key: tuple[int, int, int] = (-1, -1, 0)

            for c in candidates:
                common_len = 0
                for a, b in zip(rem_tokens, c.token_ids, strict=False):
                    if a != b:
                        break
                    common_len += 1
                if common_len > 0:
                    cand_key = (
                        common_len,
                        1 if common_len == c.num_tokens else 0,
                        -c.num_tokens,
                    )
                    if cand_key > best_key:
                        best_key = cand_key
                        best_child = c
                        best_common = common_len

            if best_child is None or best_common == 0:
                break

            if best_common == best_child.num_tokens:
                # Full match of this child node
                nodes.append(best_child)
                matched_len += best_common
                curr = best_child
                curr.last_access = time.time()
            else:
                # Partial match inside best_child -> split it!
                old_tokens = best_child.token_ids
                best_child.split(best_common)
                if (
                    old_tokens in curr.ragged_children
                    and curr.ragged_children[old_tokens] is best_child
                ):
                    del curr.ragged_children[old_tokens]
                    curr.ragged_children[best_child.token_ids] = best_child
                nodes.append(best_child)
                matched_len += best_common
                curr = best_child
                curr.last_access = time.time()
                break

        return nodes, matched_len

    def _attach_child(self, parent: RadixNode, new_node: RadixNode) -> None:
        """Attaches `new_node` to `parent`, preserving any existing child with the same first token in `ragged_children`."""
        first_tok = new_node.token_ids[0]
        existing = parent.children.get(first_tok)
        if existing is not None and existing is not new_node:
            parent.ragged_children[existing.token_ids] = existing
        parent.ragged_children.pop(new_node.token_ids, None)
        parent.children[first_tok] = new_node

    def _append_chunked_nodes(
        self,
        curr: RadixNode,
        rem_tokens: tuple[int, ...],
        rem_k: dict[int, torch.Tensor],
        rem_v: dict[int, torch.Tensor],
        nodes: list[RadixNode],
    ) -> None:
        """Slices `rem_tokens` and its KV tensors into `chunk_size` blocks and attaches them under `curr`."""
        if not rem_tokens:
            return
        step = (
            self.chunk_size
            if self.chunk_size is not None and self.chunk_size > 0
            else len(rem_tokens)
        )
        for offset in range(0, len(rem_tokens), step):
            chunk_toks = rem_tokens[offset : offset + step]
            c_len = len(chunk_toks)
            chunk_k = {
                layer: k[..., offset : offset + c_len, :].contiguous().clone()
                for layer, k in rem_k.items()
            }
            chunk_v = {
                layer: v[..., offset : offset + c_len, :].contiguous().clone()
                for layer, v in rem_v.items()
            }
            new_node = RadixNode(chunk_toks, chunk_k, chunk_v)
            self._attach_child(curr, new_node)
            nodes.append(new_node)
            curr = new_node

    def _coalesce_or_append(
        self,
        nodes: list[RadixNode],
        rem_tokens: tuple[int, ...],
        rem_k: dict[int, torch.Tensor],
        rem_v: dict[int, torch.Tensor],
    ) -> None:
        """Coalesces `rem_tokens` starting at a trailing ragged edge if applicable, or appends after `nodes[-1]`."""
        curr = nodes[-1] if nodes else self.root
        if (
            nodes
            and self.chunk_size is not None
            and self.chunk_size > 0
            and 0 < curr.num_tokens < self.chunk_size
            and not curr.children
            and not curr.ragged_children
            and all(
                l_i in curr.k_cache and l_i in curr.v_cache for l_i in rem_k
            )
        ):
            parent = nodes[-2] if len(nodes) >= 2 else self.root
            parent.ragged_children[curr.token_ids] = curr
            nodes.pop()
            coalesced_tokens = curr.token_ids + rem_tokens
            coalesced_k = {
                l_i: torch.cat([curr.k_cache[l_i], k], dim=-2)
                for l_i, k in rem_k.items()
            }
            coalesced_v = {
                l_i: torch.cat([curr.v_cache[l_i], v], dim=-2)
                for l_i, v in rem_v.items()
            }
            self._append_chunked_nodes(
                parent, coalesced_tokens, coalesced_k, coalesced_v, nodes
            )
        else:
            self._append_chunked_nodes(curr, rem_tokens, rem_k, rem_v, nodes)

    def insert(
        self,
        tokens: list[int] | tuple[int, ...],
        k_by_layer: dict[int, torch.Tensor],
        v_by_layer: dict[int, torch.Tensor],
    ) -> list[RadixNode]:
        """Inserts a token sequence and its KV caches into the Radix Tree.

        Reuses any existing prefix matches and only allocates new nodes
        for newly seen suffix tokens. When extending a ragged prompt edge,
        preserves the standalone ragged edge and coalesces the continuation
        starting at the ragged edge boundary.

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

        rem_tokens = seq[matched_len:]
        rem_k = {
            layer: k[..., matched_len:, :]
            for layer, k in k_by_layer.items()
        }
        rem_v = {
            layer: v[..., matched_len:, :]
            for layer, v in v_by_layer.items()
        }

        self._coalesce_or_append(nodes, rem_tokens, rem_k, rem_v)
        self.evict_if_needed()
        return nodes

    def insert_delta(
        self,
        base_tokens: list[int] | tuple[int, ...],
        suffix_tokens: list[int] | tuple[int, ...],
        k_delta_by_layer: dict[int, torch.Tensor],
        v_delta_by_layer: dict[int, torch.Tensor],
    ) -> list[RadixNode] | None:
        """Extends an existing cached prefix with suffix tokens and delta KV slices.

        Preserves existing aligned canonical prefix tensors. When `base_tokens` ends
        in a ragged edge (`< chunk_size` tokens), preserves that standalone ragged edge
        for future prompt-boundary branches while coalescing the suffix starting at
        the ragged edge into `chunk_size`-aligned chunks.

        Args:
            base_tokens: Base prefix token sequence already in the cache.
            suffix_tokens: Suffix token IDs being appended.
            k_delta_by_layer: Mapping of layer_idx to delta Key tensor of shape
                `[batch, num_kv_heads, len(suffix_tokens), head_dim]`.
            v_delta_by_layer: Mapping of layer_idx to delta Value tensor of shape
                `[batch, num_kv_heads, len(suffix_tokens), head_dim]`.

        Returns:
            List of RadixNodes representing the complete token sequence, or None if base was evicted.
        """
        full_seq = tuple(base_tokens) + tuple(suffix_tokens)
        nodes, matched_len = self.match(full_seq)

        if matched_len == len(full_seq):
            return nodes

        base_len = len(base_tokens)

        if matched_len >= base_len:
            delta_offset = matched_len - base_len
            rem_tokens = full_seq[matched_len:]
            if rem_tokens:
                rem_k = {
                    l_i: k[..., delta_offset:, :]
                    for l_i, k in k_delta_by_layer.items()
                }
                rem_v = {
                    l_i: v[..., delta_offset:, :]
                    for l_i, v in v_delta_by_layer.items()
                }
                self._coalesce_or_append(nodes, rem_tokens, rem_k, rem_v)
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
        if not node.children and not node.ragged_children and node.session_count == 0:
            out.append((node.last_access, parent, node))
        for child in node.iter_all_children():
            self._find_unpinned_leaves(node, child, out)

    def _remove_child(self, parent: RadixNode, victim: RadixNode) -> None:
        """Removes `victim` from `parent.children` or `parent.ragged_children`."""
        if (
            victim.token_ids in parent.ragged_children
            and parent.ragged_children[victim.token_ids] is victim
        ):
            del parent.ragged_children[victim.token_ids]
        first_tok = victim.token_ids[0]
        if parent.children.get(first_tok) is victim:
            del parent.children[first_tok]
            # Promote the longest remaining ragged child with the same first_tok if one exists
            same_first = [
                rc
                for rc in parent.ragged_children.values()
                if rc.token_ids and rc.token_ids[0] == first_tok
            ]
            if same_first:
                same_first.sort(key=lambda n: n.num_tokens, reverse=True)
                promoted = same_first[0]
                del parent.ragged_children[promoted.token_ids]
                parent.children[first_tok] = promoted

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
            for child in self.root.iter_all_children():
                self._find_unpinned_leaves(self.root, child, candidates)

            if not candidates:
                # No unpinned leaves eligible for eviction
                break

            candidates.sort(
                key=lambda item: (
                    1 if self._is_prefix_shadowed(item[1], item[2]) else 0,
                    item[0],
                )
            )
            _, parent, victim = candidates[0]
            was_shadowed = self._is_prefix_shadowed(parent, victim)
            self._remove_child(parent, victim)
            if not was_shadowed:
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
