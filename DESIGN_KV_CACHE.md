# FrontierSplit: Dynamic Contiguous KV Cache Specification

## 1. Executive Summary & Problem Statement

In FrontierSplit Phase 1–5, the distributed pipeline operated **statelessly** during autoregressive token generation:
* On every generated token step $L$, the client/scheduler transmitted the full accumulated sequence `prompt_tokens + generated_tokens` (`[L, 4096]`).
* Every worker recomputed the forward pass over all $L$ tokens across its assigned layers.
* While this eliminated session tracking, cache invalidation, and cross-request state leaks, it incurred an **$O(L^2)$ computational overhead** over the lifetime of a sequence.

This document specifies the **Dynamic Contiguous KV Cache ("Mode: Geometric-Doubling Tile-Aligned Cache")**, which:
1. Retains local $K$ and $V$ activation tensors in worker GPU VRAM, reducing decode complexity from $O(L^2)$ to **$O(L)$**.
2. Avoids the memory fragmentation of naive static pre-allocation ($32,768$ tokens) through **cap-aware tile-aligned allocation and geometric doubling**.
3. Keeps KV memory **100% physically contiguous**, enabling peak Tensor Core utilization via standard cuBLAS / PyTorch GEMM without requiring custom non-contiguous CUDA paging kernels (PagedAttention).
4. Eliminates GPU race conditions by adopting a **Single-Consumer Multi-Producer (SCMP)** execution model.

---

## 2. Hardware Alignment & Tensor Fundamentals

### 2.1 Silicon Granularity (NVIDIA Turing / Tesla T4)
* **SIMT Execution Unit**: 32 threads per warp (Subgroup = 32).
* **Tensor Core Tile**: $16 \times 16 \times 16$ Matrix Multiply-Accumulate (MMA) hardware instructions.
* **Memory Bus Coalescing**: 32-byte, 64-byte, and 128-byte sector transactions across the 300 GB/s GDDR6 bus.

### 2.2 Alignment Invariant
Every KV cache allocation must have its sequence length dimension rounded to a **multiple of 16**:
$$\operatorname{ceil}_{16}(N) = \left\lceil \frac{N}{16} \right\rceil \times 16$$

This guarantees that:
* Every thread warp access starts on a natural memory sector boundary.
* No Tensor Core warp experiences thread masking or divergence during the matrix-vector dot product reduction.

---

## 3. Concurrency & Threading Model

To guarantee thread safety without introducing latency-killing mutexes or locks around GPU buffers:

```
[ HTTP Client 1 ] ──► (FastAPI Worker Thread A) ──┐
[ HTTP Client 2 ] ──► (FastAPI Worker Thread B) ──┼──► [ Thread-Safe Queue (asyncio.Queue) ]
[ HTTP Client 3 ] ──► (FastAPI Worker Thread C) ──┘                  │
                                                                     ▼
                                                     ┌───────────────────────────────┐
                                                     │ DEDICATED GPU ENGINE THREAD   │
                                                     │  • Drains batch of up to M    │
                                                     │  • Reallocates / doubles      │
                                                     │  • Dispatches batched GEMM    │
                                                     │  • ZERO GPU MUTEXES           │
                                                     └───────────────┬───────────────┘
                                                                     │
                                                                     ▼
                                                     [ Resolves per-request Futures ]
```

1. **Multi-Producer API Layer**: Web server threads accept incoming HTTP completions and push `GenerationRequest` objects to a queue, attaching an `asyncio.Future` for token responses.
2. **Single-Consumer Engine**: A dedicated inference loop exclusively owns the model weights, session dictionary, and GPU memory allocations.
3. **Zero Locking**: Because a single thread performs batch assembly, buffer reallocations, and kernel launches, **zero locks or mutexes** are required on GPU memory.

---

## 4. Memory Allocation Policy

### 4.1 Inputs
For each incoming request:
* $T_{\text{prompt}}$: Exact prompt length (tokenized upfront in $< 0.5\text{ ms}$ on CPU).
* $T_{\text{max\_gen}}$: Caller's requested `max_tokens`.
* $\text{hard\_cap} = T_{\text{prompt}} + T_{\text{max\_gen}}$.

### 4.2 Sizing Rules
1. **Rule 1: Hard Cap $\le 256$ (Small-Cap / Classification / Benchmark Mode)**:
   * Used heavily in MMLU, classification, tool calling, and guardrails.
   * **Action**: Allocate strictly to the aligned hard cap:
     $$\text{Capacity} = \operatorname{ceil}_{16}(\text{hard\_cap})$$
   * **Property**: Zero memory waste, zero reallocations, guaranteed upper bound.

2. **Rule 2: Hard Cap $> 256$ (Standard Interactive / Open-Ended Mode)**:
   * Compute initial target based on empirical median response lengths (~200 tokens):
     $$\text{target} = T_{\text{prompt}} + \min(T_{\text{max\_gen}}, 256)$$
   * Set initial capacity to the next power-of-two $\ge \text{target}$, capped by the aligned hard cap:
     $$\text{Initial Capacity} = \min(\operatorname{ceil}_{16}(\text{hard\_cap}), \ 2^{\lceil \log_2(\text{target}) \rceil})$$
   * **Property**: ~90% of all chat responses finish inside this initial allocation with **zero reallocations**.

3. **Rule 3: Geometric Doubling on Capacity Exhaustion**:
   * If token generation reaches current capacity:
     $$\text{New Capacity} = \min(\operatorname{ceil}_{16}(\text{hard\_cap}), \ \text{Current Capacity} \times 2)$$
   * Allocate new contiguous buffer on GPU.
   * Device-to-device copy (`D2D memcpy`) old tokens to new buffer.
   * Deallocate old buffer.

### 4.3 Reallocation Cost Analysis
* For 512 tokens across 16 layers of Mistral 7B (FP16):
  * KV footprint $\approx 16\text{ MB}$.
  * Internal GPU copy speed at 300 GB/s:
    $$t_{\text{copy}} = \frac{16\text{ MB}}{300\text{ GB/s}} \approx 0.05\text{ ms}$$
* A single layer forward pass takes $\sim 25\text{ ms}$.
* The copy overhead is **$< 0.2\%$ of one token step**, occurring at most 2–3 times over the entire lifetime of a long sequence (amortized $O(1)$).

---

## 5. Pipeline Parallelism Contract (FrontierSplit Invariant)

In FrontierSplit, the model is split across nodes by layer slices:
* **Node 1**: Layers $0 \dots 15$.
* **Node 2**: Layers $16 \dots 31$.

```
[ Ingress / Scheduler ]
        │  Sends [batch, 1] new tokens (or prefill prompt) + request_id
        ▼
 ┌──────────────┐
 │    NODE 1    │  <-- Manages local Session KV Cache for Layers 0-15
 │ Layers 0-15  │
 └──────┬───────┘
        │  Over VPC Network: ONLY transient activation tensor [batch, 4096] (fp16)
        ▼
 ┌──────────────┐
 │    NODE 2    │  <-- Manages local Session KV Cache for Layers 16-31
 │ Layers 16-31 │
 └──────┬───────┘
        │  Emits generated next token IDs
        ▼
 [ Generation Feedback ]
```

### Invariants:
1. **Zero KV Network Traffic**: KV caches are strictly local to each node. Neither Keys nor Values ever cross the network between Node 1 and Node 2.
2. **Symmetric Session Tracking**: Both Node 1 and Node 2 track the same `request_id` in their local session dictionaries.
3. **Cleanup Synchronization**: When a sequence completes (emits `EOS` or hits `max_tokens`), the scheduler sends a lightweight `ReleaseSession(request_id)` message to all worker nodes to immediately reclaim VRAM.

---

## 6. Implementation Architecture

### 6.1 Worker State Structure (`frontiersplit/worker.py`)
```python
class SessionKVCache:
    def __init__(self, capacity: int, num_layers: int, num_kv_heads: int, head_dim: int, device: torch.device):
        self.capacity = capacity
        self.current_length = 0
        self.hard_cap = capacity
        # Structure of Arrays (SoA): separate contiguous buffers for K and V
        # Shape: [num_layers, capacity, num_kv_heads, head_dim]
        self.k_cache = torch.empty((num_layers, capacity, num_kv_heads, head_dim), dtype=torch.float16, device=device)
        self.v_cache = torch.empty((num_layers, capacity, num_kv_heads, head_dim), dtype=torch.float16, device=device)

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        # Writes strictly to current_length index; doubles buffer if current_length + new_tokens > capacity
        ...
```

### 6.2 Scheduler Session Protocol (`frontiersplit/protocol.py`)
* Update `BatchedActivationPacket` to carry `request_ids: List[str]` and a `is_prefill: bool` flag.
* Add an explicit `ReleaseSessionPacket` sent when requests conclude.
