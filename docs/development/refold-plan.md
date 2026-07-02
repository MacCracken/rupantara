# rupantara → attn11 re-fold: status

**Status:** ✅ **DONE for the leaf ops (2026-07-02).** attn11 now consumes
rupantara's `ru_*` leaf forward ops; the parity gate (attn11's full grad-check
suite green in one binary) is GREEN. Cross-repo with attn11.

## The problem (divergence)

rupantara's transformer **forward** was *ported verbatim* from attn11, and attn11
still had its own forward. Two CPU forwards → they will **drift**. An extraction
only pays off if the parent *consumes* it (one source). Half-done = worse than not
done. (The **GPU** forward is NOT duplicated — it already lives, shared, in
`rosnet-gpu`; only the **CPU** math was doubled.)

## Fix — rupantara side: namespace ALL public + private ops (`ru_*` / `_ru_*`)

Cyrius has **no module-private scoping** — every `fn` is a global symbol, and
duplicate definitions do **not** hard-error: cyrius emits
`warning: duplicate fn '…' (last definition wins)` and silently shadows. So a
naïve link would build green while running the wrong body. The rename therefore
had to cover **every** colliding symbol, not just the public ops:

- **12 public forward ops** → `ru_` prefix: `ru_ln_fwd`, `ru_gelu_fwd`,
  `ru_attn_core_fwd`, `ru_attn_fwd`, `ru_mlp_fwd`, `ru_embed_fwd`, `ru_block_fwd`,
  `ru_model_fwd`, `ru_head_fwd`, `ru_head_fwd_row`, `ru_softmax_fwd`,
  `ru_attn_arena_size` (plus the pre-existing `ru_cfg_init` / `ru_params_count` /
  `rupantara_version`).
- **28 private helpers** → `_ru_` prefix: `_ru_at_{Qc,Kc,Vc,Pc,concat}`,
  `_ru_eps_ln`, `_ru_gelu_{a,c}`, the `_ru_o_*` block-offset family, and
  `_ru_{blk,blk_base,blk_base_size,emb_end}`.
- **`ru_attn_arena_size` MUST stay renamed** — rupantara's arena is **forward-only**
  (`2·T·C + 2·T·Ckv + nh·T·T`) and **smaller** than attn11's (which includes the
  backward temporaries + gated-linear scratch). If attn11 ever adopted rupantara's
  smaller arena, its backward would write past the allocation → heap corruption.
  attn11 keeps its own `attn_arena_size`.

Behavior is **bit-identical** — this was a rename only. `make test` = **43/43**
green after the rename; `dist/rupantara.cyr` regenerated exposing the `ru_*`
surface; `comm -12` of the attn11 vs rupantara `fn` sets is now **empty**.

## attn11 side — DONE (see attn11 `docs/development/rupantara-refold.md`)

attn11 added `[deps.rupantara]` (local `path` override during the re-fold) +
`include "lib/rupantara.cyr"`, and routed the **CPU bodies of three leaf ops** to
rupantara's `ru_*`, keeping the GPU (`--gpu` / `--gpu-tc`) and bidirectional
(`g_bidir`) branches attn11-local:

| attn11 op | delegates to | condition |
|---|---|---|
| `ln_fwd` | `ru_ln_fwd` | CPU path (below the `--gpu` guard) |
| `gelu_fwd` | `ru_gelu_fwd` | CPU path (below the `--gpu-tc` guard) |
| `attn_core_fwd` | `ru_attn_core_fwd` | **causal only** (`g_bidir == 0`); the diffusion body stays attn11-local |

## Scope — what IS and ISN'T single-sourced (honest)

- **Single-sourced now (parity PROVEN):** `ru_ln_fwd`, `ru_gelu_fwd`,
  `ru_attn_core_fwd` (causal). attn11 runs these under its 1049 grad-check suite,
  so their bit-identity is proven live, in one binary.
- **NOT delegated (stays attn11-owned, by design):** the **composition/orchestration**
  ops — `embed_fwd`, `head_fwd`, `head_fwd_row` (attn11's forms read globals with a
  0/2-arg signature; rupantara's take explicit pointers — incompatible without a
  shim, and these carry training-time concerns) — plus `mlp_fwd`/`block_fwd`/
  `model_fwd` orchestration and the **row/decode, MLA, SSM, linear-attention, MoE,
  MTP, and diffusion (`g_bidir`)** paths, which rupantara does not implement. These
  remain a second copy in attn11 and are **not** de-duplicated by this re-fold.
- **Consequence for rupantara's own composition ops** (`ru_embed_fwd`,
  `ru_head_fwd`, `ru_model_fwd`, …): they are ported verbatim + pass rupantara's
  internal known-answer / plumbing tests, but attn11 does **not** exercise them, so
  they are **not** cross-validated against attn11. anukūlana is their first real
  consumer; a rupantara↔attn11 whole-forward parity fixture is still worthwhile
  before anukūlana's fidelity gate leans on `ru_model_fwd`.

## Gate (the live parity proof) — GREEN

`make check` on attn11 (fmt + lint + full grad-check suite) = **1049 passed, 0
failed** in one binary, with the three leaf ops delegated. No offline dump/compare
hack. This is what earns the word "bit-identical" for the delegated leaf ops.
