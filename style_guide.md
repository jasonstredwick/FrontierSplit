# Project conventions

Project-specific conventions for FrontierSplit. Overrides any defaults you would otherwise apply.

## Workspace and repository ownership

- Treat the current task's repository root as the boundary for project work. Run commands from that root and keep
  caches, scratch data, generated files, and other project artifacts inside it. Do not use paths outside the repository
  without the user's explicit permission.
- Paths in historical notes identify their original checkout; they do not override the current workspace.
- Use `scratch/` for temporary project work and `.__pycache__/` for Python bytecode and tool caches.
- Always use the virtual-environment Python (`.venv/bin/python`) with `-X pycache_prefix=.__pycache__` and
  `PYTHONPATH=.` when running scripts directly:
  ```sh
  PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ path/to/script.py
  ```
- The user manages Git state. Git inspection commands are allowed, but do not stage, unstage, commit, amend, rebase,
  merge, switch or create branches, push, pull, or otherwise modify the repository through Git unless the user
  explicitly requests that action.
- This is an active personal systems library and research engineering project. Do not add production infrastructure
  such as CI/CD unless the user explicitly asks for it.

## Code style

### Formatting

- The line-length soft limit is 120 characters. Longer lines are acceptable when wrapping hurts readability.
- Do not insert empty lines containing spaces or indentation. Empty lines must contain only a newline.

### Naming

- Functions and methods use PascalCase, such as `PackFrame`, `ComputePartialAttention`, and `MergeChunks`.
  - Overrides of Python builtins, dunder methods, and protocol methods retain their required names, such as `__init__`,
    `__str__`, `__post_init__`, `__iter__`, `keys`, and `items`.
  - Note: this is intentionally non-standard for Python (PEP 8 uses `snake_case`). Follow it consistently without
    commentary.
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
