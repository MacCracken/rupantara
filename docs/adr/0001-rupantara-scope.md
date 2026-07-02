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
  stable lib; attn11 consumes rupantara's **leaf** forward ops (`ru_ln_fwd`,
  `ru_gelu_fwd`, `ru_attn_core_fwd`), **proven bit-identical** by attn11's full
  grad-check suite green in one binary (2026-07-02) — the rosnet/tyche/akshara
  precedent. Scope is honest: only the three leaf CPU ops are single-sourced; the
  composition/embed/head ops and the bidir/row/MLA/SSM/MoE/MTP paths stay
  attn11-local and are not de-duplicated.
- **Negative**: the extraction is a **cross-repo** change — attn11 re-points to
  `dist/rupantara.cyr` (done for the leaf ops); rupantara's own composition ops
  (`ru_embed_fwd`, `ru_model_fwd`, …) are ported + internally tested but not yet
  cross-validated against attn11 (attn11 doesn't exercise them).
- **Neutral**: architecture breadth (RoPE/GQA/SwiGLU/RMSNorm) lands per consumer
  demand, not all up front.

## Alternatives considered

- **Reimplement the forward in anukūlana** — rejected: duplicates attn11's proven
  forward; two copies drift.
- **Have anukūlana depend on attn11-the-binary** — rejected: you can't link a
  reference *binary* as a library; the whole point is a consumable forward.
