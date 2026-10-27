# Project instructions

These project-specific instructions override defaults that conflict with them.

## Project status and collaboration

- This is an active personal systems library and research engineering project, not a corporate production service. Do
  not add production infrastructure such as CI/CD unless the user explicitly asks for it.
- Preserve the user's in-progress work. The user also writes and refines code, so treat the current workspace as an
  active collaboration rather than assuming every unfinished change is yours to clean up.
- The user manages Git state. Git inspection commands are allowed, but do not stage, unstage, commit, amend, rebase,
  merge, switch or create branches, push, pull, or otherwise modify the repository through Git unless the user
  explicitly requests that action.
- Act as an engineering and research collaborator, not only as a coding agent. Discuss architectural trade-offs, review
  measurements, propose benchmarks, and design scenarios that test or falsify performance hypotheses.
- Code represents the current architectural design, and benchmarks test where that design wins or loses. Keep claims
  grounded in explicit assumptions and measured behavior.
- Use `docs/paper/frontiersplit_technical_paper_draft2.md` as the current architectural and analytical reference for
  intra-layer partitioning, sequence chunking, transport math, and hardware Roofline modeling.

## Workspace boundary and execution

- Treat the current task's repository root as the boundary for project work.
- Run all commands from that repository root.
- Paths in historical notes identify their original checkout; they do not override the current workspace.
- Keep caches, scratch data, generated files, and other project artifacts inside the repository. Do not use paths
  outside the repository without the user's explicit permission.
- Use `scratch/` for temporary project work and `.__pycache__/` for Python bytecode and tool caches.
- Always use the virtual-environment Python: `.venv/bin/python`.
- Always invoke Python with `-X pycache_prefix=.__pycache__` so bytecode does not create scattered `__pycache__`
  directories.
- Set `PYTHONPATH=.` when running scripts directly.

The standard direct-script form is:

```sh
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ path/to/script.py
```

## Regression and runtime signals

- Tests must define model/stage configurations, input tensors or requests, and observable outputs, exercising components
  through their public entry points. Do not monkeypatch runtime classes (such as `torch.cuda.Stream`), inject internal
  bookkeeping state, or bypass public interfaces to manufacture test results.
- Judge correctness from numerical outputs, wire round-trips, session lifecycle behavior, and published telemetry rather
  than pinning private attributes or internal cache layout. Review legacy tests before treating failures as defects, and
  preserve useful scenarios when migrating them.
- Report test count, worker count (when running in parallel), wall time, and pass/fail together, keeping configuration
  differences visible.
- For transport, scheduling, or kernel changes, include a fixed, reproducible scenario timing with minimal logging.
  State the model/shape configuration, sequence length, batch/chunk settings, device, and timed interval. Keep profiled
  runs separate and avoid concurrent benchmark runs.
- Treat regression and benchmark timings as signals of unexpected overhead or poor design, not standalone proof of
  cross-hardware scaling. Compare like configurations and investigate unexpected increases.

## Architecture direction and invariants

FrontierSplit is a modular distributed LLM inference library and fleet-sizing toolkit focused on unquantized 16-bit
(`BF16` / `FP16`) execution across commodity, older, or heterogeneous hardware, using `Mistral-7B` (Dense) and
`Mixtral-8x7B` (MoE) as the primary baseline models.

Core architectural invariants are:

1. **Single coherent transport and protocol:** Use one explicit, zero-pickle binary TCP framing protocol across stages.
   Do not duplicate framing stacks across modules, depend on `numpy`, or wrap core dependencies (`torch`) in optional
   `try/except` import fallbacks.
2. **Explicit separation of `Attn+KV` state from `FFN` execution:**
   - `Attn+KV` stages own attention projections ($W_q, W_k, W_v, W_o$) and per-session KV caches. Historical KV blocks
     align to fixed chunk boundaries, while newly generated tokens append to an active **ragged-edge** tail chunk that
     migrates into the aligned store once full.
   - `FFN` / expert stages hold zero KV state, coalesce ready activations over an explicit arrival window
     ($\Delta t_{\text{wait}}$) to reuse resident weights across batch size $B$, and fan out single-token outputs to
     each session's next-stage host.
3. **Sequence chunking vs. request batching:** Distinguish cross-request weight batching ($B$ on `FFN` stages, bounded
   deterministically by Dense layers and reduced stochastically on MoE layers) from per-session KV sequence chunking
   along $S$ (supporting intra-request **self-striping** and inter-request **concurrency-striping** across a tunable
   **interruption-interval** chunk quantum).
4. **Explicit accounting:** Keep global in-flight pipeline concurrency ($M$) distinct from stage batch size ($B$),
   resident model weights distinct from per-step active bytes read, physical memory stack sizes distinct from
   software-usable VRAM, and binary byte units (`KiB`, `MiB`, `GiB`) distinct from decimal link rates (`GB/s`, `Gbps`).

## Code style

### Formatting

- The line-length soft limit is 120 characters. Longer lines are acceptable when wrapping hurts readability.
- Do not insert empty lines containing spaces or indentation. Empty lines must contain only a newline.

### Naming

- Functions and methods use PascalCase, such as `PackFrame`, `ComputePartialAttention`, and `MergeChunks`.
- Overrides of Python builtins, dunder methods, and protocol methods retain their required names, such as `__init__`,
  `__str__`, `__post_init__`, `__iter__`, `keys`, and `items`.
- Classes use PascalCase.
- Module-level constants use UPPER_SNAKE_CASE.
- Short variables such as `a`, `s`, and `n` are acceptable when used within roughly 10-20 lines of their definition.
  Longer scopes warrant meaningful names. Mathematical and systems variables such as `x`, `q`, `k`, `v`, `b`, `s`, and
  `dt` are always acceptable.

### Imports

Imports use four groups, separated by blank lines, in this order:

1. Global `import X` imports from the standard library and third-party packages.
2. Global `from X import Y` imports.
3. `frontiersplit` `import X` imports.
4. `frontiersplit` `from X import Y as fs_Y` imports.

Sort imports alphabetically within each group. For example:

```python
import asyncio
import dataclasses
import torch

from collections.abc import Iterator

import frontiersplit

from frontiersplit.attention import online_softmax as fs_online_softmax
from frontiersplit.transport import frame as fs_frame
```

- Prefer module imports over direct-name imports and access symbols through the module namespace at the call site.
- Project-internal `from frontiersplit...` imports must be aliased as `fs_<module>`. Never leave them bare. Consistent
  `fs_*` prefixes at call sites make project-internal symbols immediately distinguishable from third-party ones.
- Do not introduce `numpy` as a project dependency; use `import torch` and `torch.Tensor` directly without single-letter
  third-party aliases.
- All imports must be at the top of the file. Resolve circular dependencies by refactoring rather than using lazy
  imports inside functions.
- Avoid `if TYPE_CHECKING:` guards and string-quoted type annotations. Use real imports and
  `from __future__ import annotations` when forward references are needed (`from __future__ import annotations` is a
  compiler directive placed at the very top before the four import groups).

### Types

- Target Python 3.14 and use modern type syntax: `list[int]`, `dict[str, int]`, `X | None`, and `tuple[A, B]`.
- Do not import `List`, `Dict`, `Optional`, or `Tuple` from `typing`.

### Function calls

- Use named keyword arguments for project function and method calls, whether or not the signature enforces them.
- All new functions and methods with explicit parameters must use `*` before those parameters (after `self` or `cls`
  for methods) to enforce keyword-only arguments. Use `kw_only=True` for new dataclass constructors. Required Python
  protocol signatures retain their prescribed calling convention; parameterless functions do not need `*`.
- Existing signatures without `*` still require named keyword arguments at call sites.

### Validation and invariants

- Make assumptions explicit and validate them once at the boundary responsible for accepting or constructing the
  data. After that validation, treat those assumptions as invariants and trust them throughout downstream code.
- Do not add defensive checks for required state guaranteed by construction, an established invariant, a selector, or
  earlier validation. Rechecking required state, tensor properties, or component existence throughout normal execution
  obscures incorrect code rather than making it safer.
- Never silently accept malformed required state by skipping work, substituting a default, returning an empty result,
  or continuing along a fallback path. Required state is expected to be correct. Let an invariant violation fail
  directly, or raise an exception at its responsible boundary when the exception adds meaningful context.
- Intentionally optional data is different from malformed required state. When an API explicitly permits absence,
  such as a return type of `T | None`, check for `None` and execute the behavior defined for that case. Defaults and
  empty results are appropriate only when absence is part of the explicit contract. This is intentional handling of
  modeled possibilities, not defensive programming against mistakes or violated invariants.

### Documentation

- Treat code as the primary documentation and use Google-style documentation conventions.
- Add docstrings or comments only when they explain a non-obvious reason, a subtle invariant, genuinely complex logic,
  or an assumption a reader could not derive from the implementation.
- Do not add docstrings that merely restate a signature or name.

### Ordering

Prefer alphabetical order when it does not conflict with semantic grouping, including:

- Imports within a group.
- Dictionary keys without a natural order.
- Enum and class member listings.
- Method definitions without a natural grouping such as constructors, public methods, and private methods.
