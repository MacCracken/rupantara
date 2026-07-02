# 0001 — rupantara scope: the transformer forward, extracted from attn11

**Status**: Accepted
**Date**: 2026-07-01

## Context

The Type-3 pretrained-import path (`anukūlana`) needs to run a real model's
transformer **forward** — but the forward currently lives inside **attn11**, a
reference *binary*, not a consumable library. Reimplementing it in anukūlana
would duplicate attn11's proven, FD-gated forward. The AGNOS ecosystem's answer
is the kashi/sandhi **extract-and-re-fold** pattern: carve the reusable core into
a lib, and re-point the parent to consume it.

## Decision

`rupantara` is the transformer **forward** library, extracted from attn11:
attention (softmax MHA, later GQA/MQA/RoPE), MLP (GELU/SwiGLU), LayerNorm /
RMSNorm, the pre-norm block, token + positional embeddings, the forward over a
block stack, weight-tied LM head + softmax, and KV-cache decode.

In scope: the **forward** (inference). Out of scope (and their homes):
- **Training / backprop / optimizer** → stays in **attn11** (rupantara is
  forward-only; adapter backward for LoRA is rosnet `linear_bwd` in anukūlana).
- **Tensors / matmul** → **rosnet** (rupantara consumes it).
- **Pretrained import + LoRA/QLoRA** → **anukūlana** (the Type-3 reference).
- **Weight format** → **tula**.
- **The full sampler zoo** (top-k/top-p/beam) → the Autoregressive Type-2
  reference (`generative-paradigms.md`); rupantara provides KV-cache forward +
  greedy/argmax.

## Consequences

- **Positive**: anukūlana + the murti load-seam run a transformer via a small,
  stable lib; attn11 becomes a thin consumer of its own extracted forward
  (proven bit-identical), matching the rosnet/tyche/akshara precedent.
- **Negative**: the extraction is a **cross-repo** change — attn11 must re-point
  to `dist/rupantara.cyr`, which edits the attn11 repo (its own explicit go).
- **Neutral**: architecture breadth (RoPE/GQA/SwiGLU/RMSNorm) lands per consumer
  demand, not all up front.

## Alternatives considered

- **Reimplement the forward in anukūlana** — rejected: duplicates attn11's proven
  forward; two copies drift.
- **Have anukūlana depend on attn11-the-binary** — rejected: you can't link a
  reference *binary* as a library; the whole point is a consumable forward.
