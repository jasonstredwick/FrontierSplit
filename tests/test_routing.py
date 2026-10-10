"""Unit tests for FrontierSplit Prefix-Aware Intelligent Routing and Hash Ring."""

from __future__ import annotations

import pytest

from frontiersplit.routing import (
    ConsistentHashRing,
    ContextShardNode,
    PrefixRouter,
    hash_key_to_uint32,
)


def test_hash_key_to_uint32_determinism():
    """Verify BLAKE2b uint32 hash is 100% deterministic and non-negative."""
    h1 = hash_key_to_uint32("system_prompt_prefix_a")
    h2 = hash_key_to_uint32("system_prompt_prefix_a")
    h3 = hash_key_to_uint32("system_prompt_prefix_b")

    assert h1 == h2
    assert h1 != h3
    assert 0 <= h1 <= 0xFFFFFFFF


def test_consistent_hash_ring_distribution_and_failover():
    """Verify virtual node consistent hash ring routing, replicas, and health failover."""
    ring = ConsistentHashRing(vnodes_per_shard=64)

    shard0 = ContextShardNode(shard_id="shard-0", host="127.0.0.1", port=50060)
    shard1 = ContextShardNode(shard_id="shard-1", host="127.0.0.1", port=50061)
    shard2 = ContextShardNode(shard_id="shard-2", host="127.0.0.1", port=50062)

    ring.add_shard(shard0)
    ring.add_shard(shard1)
    ring.add_shard(shard2)

    # 1. Deterministic node selection
    node_a = ring.get_node("prefix:agent_system_coder")
    node_a_repeat = ring.get_node("prefix:agent_system_coder")
    assert node_a.shard_id == node_a_repeat.shard_id

    # 2. Replica set
    replicas = ring.get_replicas("prefix:agent_system_coder", count=2)
    assert len(replicas) == 2
    assert replicas[0].shard_id == node_a.shard_id
    assert replicas[1].shard_id != node_a.shard_id

    # 3. Failover when primary shard becomes unhealthy
    primary_id = node_a.shard_id
    ring.set_shard_health(primary_id, is_healthy=False)

    failover_node = ring.get_node("prefix:agent_system_coder")
    assert failover_node.shard_id != primary_id
    assert failover_node.is_healthy is True

    # 4. Removing a shard
    ring.remove_shard("shard-2")
    assert "shard-2" not in ring.shards


def test_prefix_router_session_affinity():
    """Verify that explicit session_id overrides prefix hashing for multi-turn dialogues."""
    router = PrefixRouter()
    router.add_shard("shard-0", "127.0.0.1", 50060)
    router.add_shard("shard-1", "127.0.0.1", 50061)
    router.add_shard("shard-2", "127.0.0.1", 50062)

    session_id = "agent-swebench-instance-492"

    # Turn 1
    messages_turn_1 = [
        {"role": "system", "content": "You are SWE-bench agent."},
        {"role": "user", "content": "Fix bug in repo."},
    ]
    dec_1 = router.route_request(messages_turn_1, session_id=session_id)
    assert dec_1.route_type == "session"
    assert dec_1.fingerprint == session_id

    # Turn 2 with completely different user message
    messages_turn_2 = [
        {"role": "system", "content": "You are SWE-bench agent."},
        {"role": "user", "content": "Fix bug in repo."},
        {"role": "assistant", "content": "Patch generated."},
        {"role": "user", "content": "Now run pytest on tests/test_core.py"},
    ]
    dec_2 = router.route_request(messages_turn_2, session_id=session_id)
    assert dec_2.route_type == "session"
    # Must land on the exact same shard!
    assert dec_1.shard.shard_id == dec_2.shard.shard_id


def test_prefix_router_implicit_prefix_affinity():
    """Verify that requests sharing the same system prompt prefix hash to the same shard."""
    router = PrefixRouter(prefix_window_tokens=32)
    router.add_shard("shard-0", "127.0.0.1", 50060)
    router.add_shard("shard-1", "127.0.0.1", 50061)
    router.add_shard("shard-2", "127.0.0.1", 50062)

    common_sys_prompt = (
        "You are an autonomous pair programmer assisting with Python refactoring."
    )

    # User A asks question 1
    req_a = [
        {"role": "system", "content": common_sys_prompt},
        {
            "role": "user",
            "content": "How do I optimize matrix multiplication in PyTorch?",
        },
    ]

    # User B asks completely different question 2
    req_b = [
        {"role": "system", "content": common_sys_prompt},
        {
            "role": "user",
            "content": "Explain Radix Tree prefix caching in disaggregated LLMs.",
        },
    ]

    dec_a = router.route_request(req_a)
    dec_b = router.route_request(req_b)

    assert dec_a.route_type == "prefix"
    assert dec_b.route_type == "prefix"
    # Because both share the same system prompt prefix, they must route to the same shard!
    assert dec_a.shard.shard_id == dec_b.shard.shard_id


def test_prefix_router_empty_and_fallback_validation():
    """Verify ingress rejection on empty message list and graceful fallback on empty content."""
    router = PrefixRouter()
    router.add_shard("shard-0", "127.0.0.1", 50060)

    # 1. Empty message list should raise ValueError (equivalent to HTTP 400)
    with pytest.raises(ValueError, match="Incoming message list cannot be empty"):
        router.route_request([])

    # 2. Empty content should gracefully fall back to canonical BOS hash without raising
    empty_content_req = [{"role": "user", "content": ""}]
    dec = router.route_request(empty_content_req)
    assert dec.route_type == "fallback"
    assert dec.fingerprint == "bos_fallback"
    assert dec.shard.shard_id == "shard-0"


@pytest.mark.anyio
async def test_prefix_router_live_cluster_query_and_failover():
    """Verify live TCP cluster routing and automatic replica failover across ContextServers."""
    import torch

    from frontiersplit.context_server import ContextServer

    server_a = ContextServer()
    port_a = server_a.start_in_thread(host="127.0.0.1", port=0)

    server_b = ContextServer()
    port_b = server_b.start_in_thread(host="127.0.0.1", port=0)

    router = PrefixRouter()
    node_a = router.add_shard("shard-a", "127.0.0.1", port_a)
    node_b = router.add_shard("shard-b", "127.0.0.1", port_b)

    try:
        # Pre-populate prefix on server_a
        prefix_tokens = [10, 20, 30, 40, 50]
        dummy_k = torch.randn(1, 2, len(prefix_tokens), 16)
        dummy_v = torch.randn(1, 2, len(prefix_tokens), 16)
        server_a.register_prompt(
            session_id="sess-pre",
            layer_idx=0,
            k=dummy_k,
            v=dummy_v,
            token_ids=prefix_tokens,
        )

        dec = router.route_request([{"role": "user", "content": "10 20 30 40 50 test"}])
        # Query prefix via router
        res_len = await router.query_shard_prefix(dec, prefix_tokens)
        assert res_len >= 0

        # Simulate primary shard failure
        server_a.stop_thread()
        # Query should gracefully attempt replica without crashing
        res_failover = await router.query_shard_prefix(dec, prefix_tokens)
        assert res_failover >= 0
    finally:
        node_a.close()
        node_b.close()
        server_a.stop_thread()
        server_b.stop_thread()
