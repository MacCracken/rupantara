# 0003 — Validate at the model entries; leaf kernels trust their caller

**Status**: Accepted
**Date**: 2026-11-09

## Context

Through 0.5.1 nothing checked anything. A config with `nh = 0` divided by
zero. One with `nh % nkv != 0` sent query heads past the last K/V head, reading
off the end of the K/V rows. `attn_bias = 2` sized blocks without biases and
then read biases. A token id outside `[0, V)` read past the embedding table.
Worst, a decode step at a position `>= T` wrote its K and V rows past the end
of the caller's KV cache, a heap out-of-bounds write reached by an ordinary
mistake ("generate 100 tokens" with `T = 64`).

The kernels are also the hot loops, and rosnet's kernels deliberately do not
check (rosnet's no-bounds-checks contract). A check inside
`ru_attn_core_fwd`'s inner loops would cost on every element, and some
consumers (attn11) call the leaf kernels directly with shapes they have already
established.

## Decision

**The entries that interpret a config or index with caller data validate it;
the leaf kernels do not.**

- `ru_cfg_check` defines a valid config. It covers dims, whole heads, whole
  GQA groups, a 0/1 bias flag, and every derived buffer size within 2^56
  elements using overflow-checked arithmetic.
- The model entries (`ru_model_fwd`, `ru_model_fwd_s`, `ru_model_fwd_row`,
  `ru_model_fwd_row_s`), `ru_attn_fwd`, `ru_argmax` and the sizing functions
  run their checks first. On failure they return a negative code (`RU_ERR_*`)
  and leave every output untouched, the KV cache included.
- The leaf kernels (`ru_ln_fwd`, `ru_attn_core_fwd`, `ru_head_fwd_row`, …)
  check nothing. Their preconditions are documented in `docs/api.md`.
- The error codes are functions, not `const`, so the bundle stays compilable
  by consumers on cyrius releases older than the 6.7 `const` keyword.

## Consequences

- **Positive**: every consumer-facing entry fails closed and loudly. The decode
  overflow cannot happen through them. The checks are O(1), plus O(T) for
  `ru_model_fwd`'s token scan, and invisible next to the forward.
- **Negative**: a consumer composing its own forward from leaf kernels gets no
  protection unless it runs `ru_cfg_check` itself. Existing callers that ignore
  return values (attn11, anukūlana) keep ignoring them. They pass valid
  configs, so nothing changes for them, but they would not notice a rejection.
- **Neutral**: the sizing functions now return -1 for an invalid config, which
  `t_alloc` refuses loudly ("invalid size"). Before, they divided by zero or
  returned a wrapped count.

## Alternatives considered

- **Check inside every kernel**: rejected. It costs the hot loops, and it
  duplicates checks the callers already made.
- **Abort the process on a bad argument** (rosnet `t_alloc`'s style): rejected
  for the forward entries. A model loader that hands rupantara an untrusted
  config needs a code it can report, not an exit.
