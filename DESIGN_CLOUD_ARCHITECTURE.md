# FrontierSplit: Cloud Architecture & Horizontal Scaling Specification

## 1. Executive Summary & Vision

FrontierSplit is designed to scale horizontally across commodity cloud infrastructure (GCP, AWS, Azure, on-premises Kubernetes) to serve multi-turn compound AI agent workloads (e.g., DSPy, OpenHands, Devin-style SWE swarms).

Traditional monolithic LLM serving frameworks (vLLM, TGI, TensorRT-LLM) couple prefill compute, autoregressive decode compute, and Key-Value (KV) cache memory onto the same GPU devices. In high-concurrency multi-agent environments, this tight coupling causes severe architectural bottlenecks:
1. **Prefill-Decode Interference**: Heavy, bursty prompt prefill runs (e.g., 32k-token repository codebases) preempt or stall low-latency token-by-token decoding streams.
2. **VRAM Memory Stranding**: GPUs must allocate huge VRAM buffers for KV caches, leaving compute cores (Tensor Cores) idle while waiting for memory bandwidth.
3. **Prefix Cache Fragmentation**: Multi-turn agent threads hitting different worker replicas cannot share or reuse system prompt and repository KV activations.

FrontierSplit solves this by disaggregating inference into **three independently scalable tiers**:
- **Tier 1: Ingress Gateway & Prefix-Aware Intelligent Routers**
- **Tier 2: Compute Worker Pools (Dedicated Prefill vs. Decode Nodes)**
- **Tier 3: Distributed Disaggregated Context Memory Fabric (Sharded Context Servers)**

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Multi-Agent Swarms"]
        C1["Agent Session 1 (SWE-bench)"]
        C2["Agent Session 2 (IFEval)"]
        C3["Agent Session N (Web Search)"]
    end

    subgraph Tier1 ["Tier 1: Ingress & Intelligent Routing"]
        LB["Cloud Load Balancer (NLB / ALB)"]
        R1["FrontierSplit Router 1"]
        R2["FrontierSplit Router 2"]
        CH["Consistent Prefix Hash Ring"]
    end

    subgraph Tier2 ["Tier 2: Compute Worker Pools"]
        subgraph PrefillPool ["Prefill Compute Pool (Compute-Heavy)"]
            P1["TPU v5e / H100 Node 1"]
            P2["TPU v5e / H100 Node 2"]
        end
        subgraph DecodePool ["Decode Compute Pool (Memory-Bandwidth)"]
            D1["T4 / L4 Node 1"]
            D2["T4 / L4 Node 2"]
            D3["T4 / L4 Node 3"]
        end
    end

    subgraph Tier3 ["Tier 3: Disaggregated Context Fabric"]
        CS1["Context Server Shard 0 (Radix Tree + DRAM)"]
        CS2["Context Server Shard 1 (Radix Tree + DRAM)"]
        CS3["Context Server Shard 2 (Radix Tree + DRAM)"]
        SS["Spillover Tier (Local NVMe SSD / Ceph / S3)"]
    end

    C1 & C2 & C3 --> LB
    LB --> R1 & R2
    R1 & R2 --- CH
    
    R1 -->|"Route Prompt"| P1
    R2 -->|"Route Prompt"| P2
    
    P1 & P2 -->|"Direct Persistent TCP (Write KV)"| CS1 & CS2 & CS3
    
    R1 -->|"Dispatch Decode Step"| D1 & D2 & D3
    D1 & D2 & D3 <-->|"Fetch/Update Session KV (Persistent TCP)"| CS1 & CS2 & CS3
    
    CS1 & CS2 & CS3 -.->|"LRU Eviction / Cold Spill"| SS
```

---

## 2. Generalized Zero-Threshold KV Memory Architecture

### 2.1 The Case Against Arbitrary Thresholds
In earlier architectural discussions, a candidate approach was maintaining an arbitrary sequence-length threshold (e.g., $N = 512$ tokens), keeping KV caches local to the worker for short queries and offloading to remote Context Servers only when $N > 512$.

FrontierSplit strictly adopts a **generalized zero-threshold design**:
1. **Elimination of Thrashing & Branching**: Dual-path logic (local vs. remote) introduces state synchronization races, cache coherence bugs, and memory fragmentation when requests cross the threshold mid-generation.
2. **Deterministic Memory Footprint on Workers**: Because workers do not permanently retain large session KV pools in GPU VRAM, decode workers can run with minimal static memory, allowing higher micro-batch concurrency.
3. **Universal Prefix Deduplication**: Even small 128-token system prompts (e.g., agent system instructions, tool definitions, Few-Shot exemplars) are shared across thousands of agent iterations. Retaining all KV state in the centralized Context Fabric maximizes global prefix cache reuse.

### 2.2 Generalized Request Lifecycle
1. **Ingress & Prefix Lookup**:
   - The ingress router tokenizes the incoming message list $[t_1, t_2, \dots, t_M]$.
   - The router queries the designated Context Server shard for the longest matching prefix $[t_1, \dots, t_K]$ ($K \le M$).
2. **Prefill Phase (If $K < M$)**:
   - If $K < M$, the suffix tokens $[t_{K+1}, \dots, t_M]$ are dispatched to an available **Prefill Worker**.
   - The prefill worker computes the forward pass over the suffix, generating new Key-Value activations.
   - The prefill worker streams the computed KV tensors directly to the target Context Server shard over persistent TCP (`MSG_STORE_KV`), updating the Radix Tree.
3. **Autoregressive Decode Phase**:
   - The first token is dispatched to a **Decode Worker**.
   - For each generation step, the decode worker updates the session KV state via the Context Fabric or maintains a lightweight sliding-window local buffer with periodic chunk commits.
4. **Session Reclamation**:
   - Once the generation completes or the client closes the SSE connection, the gateway issues `ReleaseSessionPacket` to free transient decode buffers, while the shared prefix branch remains pinned in the Radix Tree.

---

## 3. Tier 1: Ingress Gateway & Prefix-Aware Intelligent Routing

### 3.1 Consistent Hash Ring on Prefix Fingerprints
To scale the Context Server fabric horizontally without a central metadata bottleneck, routers employ **Consistent Hashing with Virtual Nodes** across the cluster of Context Server shards.

```
                  Hash Ring (SHA-256 Mod 2^32)
                     ┌──────────────────┐
          Shard 0 ──►│ 0x00000000       │◄── Prefix: "System: Agent..."
                     │                  │
          Shard 1 ──►│ 0x55555555       │
                     │                  │
          Shard 2 ──►│ 0xAAAAAAAA       │◄── Session ID: "sess-8f3a"
                     │                  │
          Shard 0'──►│ 0xFFFFFFFF       │ (Virtual Node)
                     └──────────────────┘
```

1. **Deterministic Shard Assignment**:
   - A fingerprint hash is calculated from the leading system prefix:
     $$\text{Key} = \operatorname{Murmur3\_32}(\text{tokens}[0 \dots \min(M, 64)])$$
   - The router maps this key to the corresponding Context Server shard on the hash ring.
   - Requests sharing identical system prompts always route their prefix cache lookups to the exact same Context Server shard, guaranteeing cache affinity.
2. **Session Affinity for Multi-Turn Agent Chains**:
   - For subsequent turns in an active agent session, the router uses the `session_id` header or conversational thread UUID to guarantee routing to the same context node.
3. **Virtual Nodes & Rebalancing**:
   - Each physical Context Server instance hosts $V = 128$ virtual nodes on the ring to guarantee uniform load distribution across memory nodes.

### 3.2 Router Responsibilities
- **HTTP / SSE Termination**: Manages client connections, TLS termination, and standard OpenAI `/v1/chat/completions` protocol semantics.
- **Micro-Batch Assembly**: Groups concurrent single-token decode requests across different sessions into dense continuous micro-batches for the Decode Workers.
- **Failover & Circuit Breaking**: If a Context Server shard fails its health probe, the router temporarily maps the virtual node interval to the next replica on the ring.

---

## 4. Tier 2: Compute Worker Pools (Disaggregated Prefill vs. Decode)

LLM inference has two diametrically opposed computational regimes:

| Characteristic | Prompt Prefill Phase | Autoregressive Decode Phase |
| :--- | :--- | :--- |
| **Compute Metric** | High FLOP/s ($O(N)$ GEMM over sequence) | Memory-bandwidth bound ($O(1)$ GEMV per step) |
| **Arithmetic Intensity**| High (Hundreds of FLOPs per byte loaded) | Extremely Low (~1–2 FLOPs per byte loaded) |
| **Optimal Hardware** | Compute-dense: Google Cloud TPU v4/v5e, NVIDIA H100, L40S | Cost-effective / Memory-bandwidth: NVIDIA L4, T4, Apple Silicon |
| **Execution Model** | Large chunked GEMM ($C = 512, 1024, 4096$) | Batched GEMV ($B = 16..64$, sequence step $s = 1$) |
| **Scaling Trigger** | Spikes in input prompt volume / repository ingest | Growth in concurrent active agent sessions |

### 4.1 Prefill Worker Pool
- **Architecture**: Contiguous pipeline stages or unified data-parallel nodes running on compute-heavy accelerators.
- **Batching Strategy**: Static chunking with Online Softmax normalization ([`frontiersplit/online_softmax.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/online_softmax.py)).
- **Interconnect**: Connects directly to the Context Fabric over high-speed datacenter VPC (10–100 Gbps). When prefill completes, activations are offloaded without occupying local VRAM.

### 4.2 Decode Worker Pool
- **Architecture**: Commodity GPU nodes running 1F1B continuous micro-batch pipelining ([`frontiersplit/scheduler.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/scheduler.py)).
- **Tile-Aligned Buffering**: Uses `WorkerKVCacheStore` with 16-token tile alignment ($\operatorname{ceil}_{16}$).
- **Elastic Auto-Scaling**: Scaled up and down based on active stream concurrency metrics emitted via `/v1/telemetry`.

---

## 5. Tier 3: Distributed Context Memory Fabric

### 5.1 Architecture of a Context Server Shard
Each Context Server shard runs as a dedicated high-memory daemon (e.g., `n2-standard-32` with 128 GB RAM or `r6i.4xlarge`):
1. **Radix Prefix Tree Index**:
   - Manages token hierarchies in memory.
   - Pointers reference precomputed FP16/FP8 Key and Value tensor slabs.
2. **Slab Memory Allocator**:
   - Pre-allocates pinned memory slabs in host DRAM (or GPU VRAM if equipped with a dedicated memory-tier GPU).
   - Prevents memory fragmentation and OS memory page faults during high-throughput tensor read/writes.
3. **Multi-Tiered Hierarchical Storage**:
   - **L1 (DRAM)**: Sub-millisecond retrieval for hot system prompts and active conversation sessions.
   - **L2 (Local NVMe SSD)**: Memory-mapped files (mmap) for warm sessions inactive for > 5 minutes.
   - **L3 (Object Storage / Cloud Storage)**: Cold persistent agent checkpoints stored asynchronously as compressed zstandard shards.

```
                    Context Server Memory Hierarchy
┌──────────────────────────────────────────────────────────────────┐
│ L1: Host DRAM (Pinned Tensors)     - Latency: < 0.2 ms           │
│     • Active agent prefixes & hot Radix tree nodes               │
├──────────────────────────────────────────────────────────────────┤
│ L2: Local NVMe SSD (mmap Store)    - Latency: < 2.0 ms           │
│     • Warm sessions, intermediate agent reasoning traces         │
├──────────────────────────────────────────────────────────────────┤
│ L3: Cloud Object Storage (S3/GCS)  - Latency: < 50 ms            │
│     • Cold checkpointed agent conversations, long-term memory     │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Wire Protocol & Tensor Serialization
Data transfer between workers and Context Servers uses FrontierSplit's native persistent binary protocol:
- **Header**: 16-byte fixed framing header (`MAGIC=0x4653`, `MSG_TYPE`, `PAYLOAD_LEN`, `FLAGS`).
- **Zero-Copy Serialization**: High-speed numpy/torch buffer extraction directly into socket buffers using `memoryview` and `asyncio.StreamWriter`.
- **Activation Quantization (Optional FP8 / INT8)**: Uses dynamic vector-scale quantization to reduce network bandwidth consumption by 50% with zero degradation in generation accuracy.

---

## 6. Fault Tolerance, High Availability & Disaster Recovery

### 6.1 Worker Node Failure Recovery
- **Decoupled State Advantage**: In traditional systems (e.g., vLLM), if a decode worker crashes, all in-flight session KV caches stored in its VRAM are lost, forcing the client to recompute the entire prompt.
- **FrontierSplit Recovery**: Because KV state is externalized in Tier 3, if a Decode Worker node crashes:
  1. The Ingress Router detects connection loss via TCP timeout or heartbeat probe.
  2. The router re-dispatches the session to an alternative healthy Decode Worker.
  3. The new worker fetches the session KV state from the Context Server shard in one round-trip and resumes autoregressive token emission without recomputing prompt prefill.

### 6.2 Context Server Replication & Failover
- **Primary-Replica Sharding**: Each shard on the consistent hash ring maintains an active asynchronous replica on an adjacent ring position:
  $$\text{Replica}(S_i) = S_{(i+1) \bmod N}$$
- **Health Checks & Dynamic Re-mapping**: Routers perform health probes (`/health`) every 1,000 ms. If Shard $i$ fails, traffic automatically fails over to Replica $S_{i+1}$.

---

## 7. Cloud Deployment Blueprint: Kubernetes Architecture

### 7.1 Component Breakdown

| Kubernetes Workload | Role | Resource Specs | Scaling Metric |
| :--- | :--- | :--- | :--- |
| `frontiersplit-router` | Ingress Gateway, SSE Streamer, Prefix Hasher | CPU: 4–8 cores, RAM: 8 GB (Deployment) | HTTP Request Rate & In-Flight Connections |
| `frontiersplit-context-shard` | KV Cache Storage, Radix Tree Store | High Memory: 32 cores, 128–256 GB RAM + Local NVMe SSD (StatefulSet) | Total Cached Tokens & DRAM Utilization |
| `frontiersplit-prefill-pool` | Batch Prompt Prefill Engine | Compute Accelerator: 1x TPU v5e or 1x H100 / L40S (Deployment) | Prefill Token Queue Depth |
| `frontiersplit-decode-pool` | Continuous Micro-Batch Decode Pipelining | Commodity GPU: 4x L4 / T4 (StatefulSet / Deployment) | Concurrency / Bubble Factor / Tokens per Sec |

### 7.2 Sample Kubernetes Architecture Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontiersplit-router
  namespace: frontiersplit
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontiersplit-router
  template:
    metadata:
      labels:
        app: frontiersplit-router
    spec:
      containers:
        - name: gateway
          image: ghcr.io/frontiersplit/frontiersplit:latest
          command: ["python", "-m", "frontiersplit.gateway"]
          args:
            - "--model-name=meta-llama/Llama-3-8B-Instruct"
            - "--context-cluster=frontiersplit-context-headless:50060"
            - "--port=8000"
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: frontiersplit-context-shard
  namespace: frontiersplit
spec:
  serviceName: "frontiersplit-context-headless"
  replicas: 4
  selector:
    matchLabels:
      app: frontiersplit-context-shard
  template:
    metadata:
      labels:
        app: frontiersplit-context-shard
    spec:
      containers:
        - name: context-server
          image: ghcr.io/frontiersplit/frontiersplit:latest
          command: ["python", "-m", "frontiersplit.context_server"]
          args:
            - "--host=0.0.0.0"
            - "--port=50060"
            - "--max-cached-tokens=5000000"
          resources:
            requests:
              memory: "64Gi"
              cpu: "16"
            limits:
              memory: "128Gi"
              cpu: "32"
          volumeMounts:
            - name: nvme-spillover
              mountPath: /mnt/spillover
  volumeClaimTemplates:
    - metadata:
        name: nvme-spillover
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: "fast-nvme-ssd"
        resources:
          requests:
            storage: 500Gi
```

---

## 8. Summary & Architectural Invariants

1. **Strictly Decoupled**: Prefill compute, decode compute, and KV cache memory scale independently with zero stranded hardware.
2. **Zero Arbitrary Thresholds**: The KV cache lifecycle is 100% unified, eliminating branch thrashing and edge cases.
3. **Prefix Affinity Without Bottlenecks**: Consistent hash rings on prefix fingerprints guarantee high cache hit rates across distributed Context Servers without requiring a central database.
4. **Resilient to Worker Failures**: Because state is externalized, decode worker crashes are non-destructive and can be transparently retried by the router.
