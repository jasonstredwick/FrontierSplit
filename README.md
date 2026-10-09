# FrontierSplit

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

**FrontierSplit** is a lightweight, transparent reference architecture for **disaggregated long-context LLM inference**.

It decouples the heavy, static prompt key-value cache from the dynamic token generation loop, using **Online Softmax rescaling** to compute mathematically exact attention across machines with tiny **8 KB network payloads**.

---

## The Problem: The Long-Context VRAM Wall

In modern Large Language Models (LLMs), long-context prompts (e.g., 50K–100K tokens of documents or code repositories) create massive memory pressure:

* A 100,000-token prompt requires **hundreds of megabytes to gigabytes of KV cache per user** on every GPU.
* To serve long-context models, teams are forced to buy or rent expensive multi-GPU servers (such as 80GB H100 clusters) simply to prevent Out-Of-Memory (OOM) crashes from the KV cache.
* During autoregressive token generation, the GPU streams that massive KV cache through memory bandwidth, choking decode throughput and limiting concurrent user capacity.

---

## The Solution: Disaggregated Context + Online Softmax

FrontierSplit recognizes a fundamental asymmetry in transformer generation:

1. **The Prompt is Static and Immutable:** Once prefill is finished, the prompt KV cache never changes, never grows, and is 100% read-only.
2. **The Output is Dynamic and Tiny:** The generated output grows token-by-token (the "ragged edge"), but only reaches a few hundred to a couple thousand tokens.

Instead of keeping both on an expensive GPU:

```
┌────────────────────────────────────────────────────────┐
│             Remote Context Server (CPU / RAM / GPU)     │
│   • Holds 100,000+ token static prompt KV cache        │
│   • Read-only, zero dynamic memory churn               │
└───────────────────────────┬────────────────────────────┘
                            │  ▲ Query vector q (8 KB)
                            │  │
                            │  ▼ Partial Attention (A, m, l) (8.25 KB)
┌───────────────────────────┴────────────────────────────┐
│             Lightweight Decode Worker (Small GPU)       │
│   • Holds ONLY the local output tokens (e.g. 32 tokens)│
│   • Merges prompt + output via Online Softmax          │
│   • Produces exact, bit-for-bit identical attention    │
└────────────────────────────────────────────────────────┘
```

---

## The Math: Exact Online Softmax Merging

Rather than sending the giant 200 MB KV cache across the network, the Decode Worker only sends the current token's Query vector $q$ (**8 KB**) to the Context Server.

The Context Server computes partial attention against the prompt and returns three compact objects per head:
* The partial accumulator vector: $A_p = \sum e^{s_p - m_p} V_p$
* The local maximum logit: $m_p = \max(s_p)$
* The sum of exponentials: $\ell_p = \sum e^{s_p - m_p}$

The Decode Worker computes the same on its tiny local output cache ($A_o, m_o, \ell_o$), and merges them with **zero approximation or accuracy loss**:

$$m_{\text{global}} = \max(m_p, m_o)$$

$$\ell_{\text{global}} = \ell_p \cdot e^{m_p - m_{\text{global}}} + \ell_o \cdot e^{m_o - m_{\text{global}}}$$

$$A_{\text{global}} = A_p \cdot e^{m_p - m_{\text{global}}} + A_o \cdot e^{m_o - m_{\text{global}}}$$

$$\mathbf{O_{\text{final}} = \frac{A_{\text{global}}}{\ell_{\text{global}}}}$$

This merge formula is mathematically identical to computing attention across the entire combined sequence in a single monolithic pass.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/jasonstredwick/FrontierSplit.git
cd FrontierSplit
pip install -e .
```

### 2. Run the Disaggregated Demo

Experience the architecture in action: a standalone Context Server holding 10,000 prompt tokens while a Decode Worker generates tokens with 8 KB socket round-trips:

```bash
python examples/disaggregated_demo.py
```

Sample output:
```text
======================================================================
 FrontierSplit: Disaggregated Long-Context Inference Demo
======================================================================

[Architecture Specs]
  • Prompt Length:       10,000 tokens
  • Prompt KV Footprint: 39.06 MB
  • Network Query Size:  8.00 KB (Send)
  • Network Return Size: 8.25 KB (Receive)
  • Merge Algorithm:     Exact Online Softmax Rescaling

[1/4] Context Server started on 127.0.0.1:49299
[2/4] Offloading 10,000 prompt tokens to Context Server...
      Prompt registered in 152.04 ms

[3/4] Initializing Decode Worker...
[4/4] Generating 10 tokens with disaggregated attention...
----------------------------------------------------------------------
 Step | Local Output Tokens | Query RTT | Verification vs. Native
----------------------------------------------------------------------
   1  |       1 tokens      | 35.70 ms  | ✓ Bit-for-bit Exact
   2  |       2 tokens      | 32.63 ms  | ✓ Bit-for-bit Exact
   3  |       3 tokens      | 30.49 ms  | ✓ Bit-for-bit Exact
  ...
----------------------------------------------------------------------
✓ Demo completed successfully.
  The Decode Worker held only 10 output tokens in local memory,
  yet computed exact attention over the entire 10,010-token context!
======================================================================
```

---

## Running Unit Tests

We enforce strict open-source standards with 100% type annotations and Ruff linting:

```bash
# Run the test suite
pytest

# Audit code formatting and linting
ruff check .
ruff format --check .
```

---

## Project Structure

```text
FrontierSplit/
├── pyproject.toml                # Open-source packaging and Ruff configuration
├── README.md                     # Project overview and quickstart
├── frontiersplit/
│   ├── online_softmax.py         # Pure PyTorch Online Softmax merging kernel
│   ├── context_server.py         # Static Prompt Server and persistent TCP client
│   └── decode_worker.py          # Decode Worker with DisaggregatedAttention module
├── examples/
│   └── disaggregated_demo.py     # Runnable disaggregated inference demonstration
├── tests/
│   ├── test_online_softmax.py    # Numerical equivalence tests against native attention
│   ├── test_context_server.py    # TCP socket lifecycle & partial query tests
│   └── test_decode_worker.py     # Multi-step autoregressive decode simulation
└── experiments/                  # Historical empirical benchmarks (8-Node Mixtral FP16)
```

---

## License

This project is licensed under the Apache 2.0 License.
