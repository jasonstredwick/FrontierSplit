"""Unit tests for RadixPrefixCache.

Tests compressed trie matching, node splitting, branching, KV tensor assembly,
and LRU eviction.
"""

from __future__ import annotations

import time

import torch

from frontiersplit.context.prefix_cache import RadixPrefixCache


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


def test_insert_delta_preserves_canonical_prefix_storage_sharing():
    """Verify that insert_delta and insert preserve canonical prefix tensor identity without clones.

    Review finding P2.4: Appending triggered full concatenation and insert replaced existing
    prefix tensors with clones, breaking memory sharing between readers bound before and after publication.
    """
    cache = RadixPrefixCache(max_cached_tokens=10_000)

    # 1. Register canonical prefix: [1, 2, 3, 4, 5]
    prefix_tokens = [1, 2, 3, 4, 5]
    k_prefix = {0: torch.randn(1, 2, len(prefix_tokens), 16)}
    v_prefix = {0: torch.randn(1, 2, len(prefix_tokens), 16)}
    cache.insert(prefix_tokens, k_prefix, v_prefix)

    # Reader 1 binds to this prefix
    nodes_before, len_before = cache.match(prefix_tokens)
    assert len_before == 5
    assert len(nodes_before) == 1
    canonical_k_node0 = nodes_before[0].k_cache[0]
    canonical_v_node0 = nodes_before[0].v_cache[0]

    # 2. Append suffix tokens [6, 7] using insert_delta (delta tensors only)
    suffix_tokens = [6, 7]
    k_delta = {0: torch.randn(1, 2, len(suffix_tokens), 16)}
    v_delta = {0: torch.randn(1, 2, len(suffix_tokens), 16)}

    cache.insert_delta(
        base_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        k_delta_by_layer=k_delta,
        v_delta_by_layer=v_delta,
    )

    # Reader 2 binds to the full sequence [1, 2, 3, 4, 5, 6, 7]
    full_tokens = prefix_tokens + suffix_tokens
    nodes_after, len_after = cache.match(full_tokens)
    assert len_after == 7
    assert len(nodes_after) == 2  # Prefix node + suffix node

    # The prefix node MUST retain the exact same canonical tensor instance (zero clone / zero copy)
    assert nodes_after[0].k_cache[0] is canonical_k_node0
    assert nodes_after[0].v_cache[0] is canonical_v_node0

    # Also verify that subsequent insert() calls on an overlapping sequence do not overwrite existing prefix tensors
    k_full = {
        0: torch.cat([k_prefix[0], torch.randn(1, 2, 2, 16)], dim=-2),
    }
    v_full = {
        0: torch.cat([v_prefix[0], torch.randn(1, 2, 2, 16)], dim=-2),
    }
    cache.insert(prefix_tokens + [8, 9], k_full, v_full)

    nodes_branch, _ = cache.match(prefix_tokens)
    assert nodes_branch[0].k_cache[0] is canonical_k_node0
    assert nodes_branch[0].v_cache[0] is canonical_v_node0


def test_insert_delta_evicted_base_does_not_publish_orphan_suffix():
    """Verify that when base prefix is evicted, insert_delta never publishes suffix under root."""
    cache = RadixPrefixCache(max_cached_tokens=5)

    # 1. Insert session A: [6, 7, 8] (3 tokens)
    cache.insert(
        [6, 7, 8],
        {0: torch.randn(1, 2, 3, 16)},
        {0: torch.randn(1, 2, 3, 16)},
    )
    assert cache.total_tokens == 3

    # 2. Insert session B: [1, 2, 3, 4, 5] (5 tokens). This evicts [6, 7, 8].
    cache.insert(
        [1, 2, 3, 4, 5],
        {0: torch.randn(1, 2, 5, 16)},
        {0: torch.randn(1, 2, 5, 16)},
    )
    # [6, 7, 8] should now be evicted
    _, match_a = cache.match([6, 7, 8])
    assert match_a == 0

    # 3. Now try to append suffix [9] to evicted base [6, 7, 8]
    res = cache.insert_delta(
        base_tokens=[6, 7, 8],
        suffix_tokens=[9],
        k_delta_by_layer={0: torch.randn(1, 2, 1, 16)},
        v_delta_by_layer={0: torch.randn(1, 2, 1, 16)},
    )
    # MUST return None and NEVER attach [9] to root
    assert res is None

    # Querying unrelated prompt [9] MUST NOT produce a cache hit!
    _, match_orphan = cache.match([9])
    assert match_orphan == 0


def test_radix_node_split_preserves_tensor_storage_view():
    """Verify that RadixNode.split() creates zero-copy slice views sharing the underlying storage pointer."""
    cache = RadixPrefixCache(max_cached_tokens=10_000)
    tokens_initial = [10, 20, 30, 40, 50, 60]
    k = {0: torch.randn(1, 2, 6, 16)}
    v = {0: torch.randn(1, 2, 6, 16)}
    cache.insert(tokens_initial, k, v)

    nodes, matched = cache.match(tokens_initial)
    assert matched == 6
    assert len(nodes) == 1
    orig_node = nodes[0]
    orig_k_ptr = orig_node.k_cache[0].untyped_storage().data_ptr()
    orig_v_ptr = orig_node.v_cache[0].untyped_storage().data_ptr()

    # Split the node at index 3 (branching with a new sequence [10, 20, 30, 77, 88])
    tokens_branch = [10, 20, 30, 77, 88]
    k_branch = {0: torch.randn(1, 2, 2, 16)}
    v_branch = {0: torch.randn(1, 2, 2, 16)}
    cache.insert_delta(
        base_tokens=[10, 20, 30],
        suffix_tokens=[77, 88],
        k_delta_by_layer=k_branch,
        v_delta_by_layer=v_branch,
    )

    # Now match the prefix [10, 20, 30]
    nodes_split, matched_split = cache.match([10, 20, 30])
    assert matched_split == 3
    assert len(nodes_split) == 1
    prefix_node = nodes_split[0]
    suffix_node = prefix_node.children[40]

    # Verify underlying storage pointer is identical for both prefix and suffix views
    assert prefix_node.k_cache[0].untyped_storage().data_ptr() == orig_k_ptr
    assert prefix_node.v_cache[0].untyped_storage().data_ptr() == orig_v_ptr
    assert suffix_node.k_cache[0].untyped_storage().data_ptr() == orig_k_ptr
    assert suffix_node.v_cache[0].untyped_storage().data_ptr() == orig_v_ptr


def test_eviction_compacts_split_nodes_and_frees_backing_allocation():
    """Verify that evicting split nodes reclaims dead backing slabs via selective compaction."""
    # Allow enough capacity so insert_delta does not trigger early eviction before pinning
    cache = RadixPrefixCache(max_cached_tokens=200)

    # 1. Insert 100 tokens
    tokens_100 = list(range(1, 101))
    k100 = {0: torch.randn(1, 2, 100, 16)}  # 1 * 2 * 100 * 16 * 4 = 12,800 bytes
    v100 = {0: torch.randn(1, 2, 100, 16)}
    cache.insert(tokens_100, k100, v100)

    assert cache.total_tokens == 100
    assert cache.total_backing_tokens == 100

    # 2. Branch at token 1: insert [1, 999] so that node splits into 1 token and 99-token suffix
    k_branch = {0: torch.randn(1, 2, 1, 16)}
    v_branch = {0: torch.randn(1, 2, 1, 16)}
    cache.insert_delta(
        base_tokens=[1],
        suffix_tokens=[999],
        k_delta_by_layer=k_branch,
        v_delta_by_layer=v_branch,
    )

    nodes, matched = cache.match([1])
    assert matched == 1
    prefix_node = nodes[0]
    assert prefix_node.num_tokens == 1

    # Pin the 1-token prefix node and branch node so they survive eviction
    prefix_node.session_count = 1
    branch_node = prefix_node.children[999]
    branch_node.session_count = 1

    # Before eviction: prefix node shares 100-token backing slab (12,800 bytes)
    assert prefix_node.k_cache[0].untyped_storage().nbytes() == 12800

    # 3. Change quota to force eviction of the unpinned 99-token suffix
    cache.max_cached_tokens = 2
    evicted_tokens = cache.evict_if_needed()
    assert evicted_tokens == 99

    # 4. Compaction must have been applied to surviving nodes:
    # Surviving 1-token prefix node now only holds its own 1-token allocation (128 bytes)!
    # The 12,800-byte backing slab has been freed!
    assert prefix_node.k_cache[0].untyped_storage().nbytes() == 128
    assert prefix_node.v_cache[0].untyped_storage().nbytes() == 128
    assert cache.total_tokens == 2  # 1 token prefix + 1 token branch
    assert cache.total_backing_tokens == 2



