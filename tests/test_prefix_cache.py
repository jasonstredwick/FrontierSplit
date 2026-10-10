"""Unit tests for RadixPrefixCache.

Tests compressed trie matching, node splitting, branching, KV tensor assembly,
and LRU eviction.
"""

from __future__ import annotations

import time

import torch

from frontiersplit.prefix_cache import RadixPrefixCache


def test_exact_prefix_match():
    """Verify that an exact token sequence matches completely and returns exact KV tensors."""
    cache = RadixPrefixCache(max_cached_tokens=10_000)
    tokens = [10, 20, 30, 40, 50, 60]

    k_by_layer = {
        0: torch.randn(1, 4, len(tokens), 32),
        1: torch.randn(1, 4, len(tokens), 32),
    }
    v_by_layer = {
        0: torch.randn(1, 4, len(tokens), 32),
        1: torch.randn(1, 4, len(tokens), 32),
    }

    cache.insert(tokens, k_by_layer, v_by_layer)

    nodes, matched_len = cache.match(tokens)
    assert matched_len == len(tokens)
    assert len(nodes) == 1

    assembled_k, assembled_v = cache.assemble_kv(nodes)
    assert torch.equal(assembled_k[0], k_by_layer[0])
    assert torch.equal(assembled_k[1], k_by_layer[1])
    assert torch.equal(assembled_v[0], v_by_layer[0])
    assert torch.equal(assembled_v[1], v_by_layer[1])


def test_branching_and_splitting():
    """Verify that overlapping prefixes trigger clean node splitting and branching."""
    cache = RadixPrefixCache(max_cached_tokens=10_000)

    # Insert sequence A: [1, 2, 3, 4, 5, 10, 20]
    tokens_a = [1, 2, 3, 4, 5, 10, 20]
    k_a = {0: torch.randn(1, 2, len(tokens_a), 16)}
    v_a = {0: torch.randn(1, 2, len(tokens_a), 16)}
    cache.insert(tokens_a, k_a, v_a)

    # Insert sequence B: [1, 2, 3, 4, 5, 30, 40] sharing first 5 tokens
    tokens_b = [1, 2, 3, 4, 5, 30, 40]
    k_b = {0: torch.cat([k_a[0][..., :5, :], torch.randn(1, 2, 2, 16)], dim=-2)}
    v_b = {0: torch.cat([v_a[0][..., :5, :], torch.randn(1, 2, 2, 16)], dim=-2)}
    cache.insert(tokens_b, k_b, v_b)

    # 1. Match common prefix [1, 2, 3, 4, 5]
    common_nodes, common_len = cache.match([1, 2, 3, 4, 5])
    assert common_len == 5
    assert len(common_nodes) == 1
    assert common_nodes[0].token_ids == (1, 2, 3, 4, 5)

    # 2. Match sequence A completely
    nodes_a, len_a = cache.match(tokens_a)
    assert len_a == len(tokens_a)
    assert len(nodes_a) == 2  # Prefix node + suffix A node
    assembled_ka, assembled_va = cache.assemble_kv(nodes_a)
    assert torch.equal(assembled_ka[0], k_a[0])
    assert torch.equal(assembled_va[0], v_a[0])

    # 3. Match sequence B completely
    nodes_b, len_b = cache.match(tokens_b)
    assert len_b == len(tokens_b)
    assert len(nodes_b) == 2  # Prefix node + suffix B node
    assembled_kb, assembled_vb = cache.assemble_kv(nodes_b)
    assert torch.equal(assembled_kb[0], k_b[0])
    assert torch.equal(assembled_vb[0], v_b[0])


def test_partial_match_auto_split():
    """Verify that querying a sub-slice of an unsplit node auto-splits the node."""
    cache = RadixPrefixCache(max_cached_tokens=10_000)

    tokens = [100, 200, 300, 400, 500, 600, 700, 800]
    k = {0: torch.randn(1, 1, len(tokens), 8)}
    v = {0: torch.randn(1, 1, len(tokens), 8)}
    cache.insert(tokens, k, v)

    # Query first 4 tokens
    query_sub = [100, 200, 300, 400]
    nodes, matched_len = cache.match(query_sub)
    assert matched_len == 4
    assert len(nodes) == 1
    assert nodes[0].token_ids == (100, 200, 300, 400)

    # Verify child node holds the remaining 4 tokens
    child = nodes[0].children[500]
    assert child.token_ids == (500, 600, 700, 800)

    # Reconstructed full match
    full_nodes, full_len = cache.match(tokens)
    assert full_len == len(tokens)
    assert len(full_nodes) == 2
    rec_k, rec_v = cache.assemble_kv(full_nodes)
    assert torch.equal(rec_k[0], k[0])
    assert torch.equal(rec_v[0], v[0])


def test_lru_eviction_and_pinning():
    """Verify LRU eviction prunes oldest unpinned leaves and preserves pinned nodes."""
    # Max quota of 15 tokens
    cache = RadixPrefixCache(max_cached_tokens=15)

    # Insert sequence 1 (8 tokens)
    t1 = [1, 2, 3, 4, 5, 6, 7, 8]
    k1 = {0: torch.zeros(1, 1, 8, 4)}
    v1 = {0: torch.zeros(1, 1, 8, 4)}
    cache.insert(t1, k1, v1)
    assert cache.total_tokens == 8

    time.sleep(0.01)

    # Insert sequence 2 (8 tokens, brings total to 16 tokens -> triggers eviction)
    t2 = [10, 20, 30, 40, 50, 60, 70, 80]
    k2 = {0: torch.zeros(1, 1, 8, 4)}
    v2 = {0: torch.zeros(1, 1, 8, 4)}
    cache.insert(t2, k2, v2)

    # Oldest sequence (t1) should have been evicted
    assert cache.total_tokens == 8
    _, m1 = cache.match(t1)
    assert m1 == 0  # t1 was evicted

    _, m2 = cache.match(t2)
    assert m2 == 8  # t2 remains

    # Now pin t2
    nodes2, _ = cache.match(t2)
    nodes2[0].session_count = 1  # Pinned!

    time.sleep(0.01)
    # Insert sequence 3 (8 tokens, brings total to 16 tokens)
    t3 = [100, 200, 300, 400, 500, 600, 700, 800]
    k3 = {0: torch.zeros(1, 1, 8, 4)}
    v3 = {0: torch.zeros(1, 1, 8, 4)}
    cache.insert(t3, k3, v3)

    # Since t2 is pinned, t3 (even though newer) or unpinned nodes will be evicted
    # t2 MUST NOT be evicted because session_count > 0!
    _, m2_after = cache.match(t2)
    assert m2_after == 8  # t2 was protected by pin!
