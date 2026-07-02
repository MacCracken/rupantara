# Changelog

All notable changes to `rupantara` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); SemVer (pre-1.0 — the public
surface is still moving, no API freeze until v1.0).

## [Unreleased]

## [0.3.0] — 2026-07-02

**Re-fold — `ru_*` namespacing + attn11 consumes rupantara's leaf ops.** The full
op surface is namespaced so attn11 can link rupantara, and attn11 now delegates its
CPU leaf forward (LayerNorm / GELU / causal attention core) to rupantara — **proven
bit-identical** by attn11's 1049 grad-checks green in one binary (the real parity
gate, no offline compare).

### Breaking
- **Public forward ops renamed to `ru_*`.** Consumers calling the 0.2.0 bare names
  (`ln_fwd`, `gelu_fwd`, `attn_fwd`, `model_fwd`, `embed_fwd`, `head_fwd`,
  `head_fwd_row`, `mlp_fwd`, `block_fwd`, `softmax_fwd`, `attn_core_fwd`,
  `attn_arena_size`) must switch to the `ru_`-prefixed forms (`ru_ln_fwd`, …).
  **Migration:** prefix the call site with `ru_`. Numerics are unchanged — pure rename.

### Changed
- **`ru_*` / `_ru_*` namespacing of the full op surface** (the re-fold
  prerequisite). Renamed all 12 public forward ops to `ru_` (`ru_ln_fwd`,
  `ru_gelu_fwd`, `ru_attn_core_fwd`, `ru_attn_fwd`, `ru_mlp_fwd`, `ru_embed_fwd`,
  `ru_block_fwd`, `ru_model_fwd`, `ru_head_fwd`, `ru_head_fwd_row`,
  `ru_softmax_fwd`, `ru_attn_arena_size`) and all 28 colliding private helpers to
  `_ru_` (`_ru_at_*`, `_ru_o_*`, `_ru_eps_ln`, `_ru_gelu_*`, `_ru_blk*`,
  `_ru_emb_end`). Rename only — **bit-identical**, `make test` = 43/43, `dist`
  regenerated. Motivation: Cyrius has no module-private scoping and *silently
  shadows* duplicate `fn`s (last-def-wins, warn-only), so `comm -12` vs attn11 had
  to reach **empty** before attn11 could link rupantara. **API note:** consumers
  now call `ru_*` (breaking vs 0.2.0's un-prefixed names) — pre-1.0, surface still
  moving.

### Added
- **attn11 re-fold (leaf ops) landed** (cross-repo; 2026-07-02). attn11 added
  `[deps.rupantara]` + `include "lib/rupantara.cyr"` and now delegates the CPU
  bodies of `ln_fwd` → `ru_ln_fwd`, `gelu_fwd` → `ru_gelu_fwd`, and
  `attn_core_fwd` (causal, `g_bidir==0`) → `ru_attn_core_fwd`, keeping its GPU +
  diffusion branches. attn11's full grad-check suite is **1049 green in one
  binary** with the delegation → those three leaf ops are **proven bit-identical**
  (the real parity gate, no offline compare). See `docs/development/refold-plan.md`.

### Known limitations (scope — honest)
- The **composition** ops (`ru_embed_fwd` / `ru_head_fwd` / `ru_model_fwd` / …) are
  *not* cross-validated against attn11 (attn11 keeps its own — incompatible
  global-reading signatures + training concerns). A rupantara↔attn11
  whole-`ru_model_fwd` parity fixture is the true M1-acceptance gap; do it before
  anukūlana's fidelity gate relies on `ru_model_fwd`.

## [0.2.0] — 2026-07-02

**M1 (rupantara side complete) — the transformer forward, extracted from attn11.**
First tagged release: the standalone GPT-2-shaped forward runs green, bit-identical
by construction to attn11 (the cross-repo parity run is the remaining step above).

### Added
- **Leaf ops** ported bit-identical from attn11 `ops.cyr`: **LayerNorm**
  (`src/norm.cyr:ln_fwd`, ε=1e-5 mean/var, caches mean+rstd) and **GELU**
  (`src/act.cyr:gelu_fwd`, tanh-approx c=√(2/π) a=0.044715). Names kept identical
  to attn11 so the eventual re-point is a clean drop-in.
- **`rosnet` 0.2.0 dependency** (`[deps.rosnet]`, `dist/rosnet.cyr`) — supplies
  `linear_fwd` (the Q/K/V/O + MLP projections) and `t_alloc`/`t_zero`/`tget`/`tset`.
  Same tag attn11 pins, so the matmul feeding the forward is bit-identical on both
  sides (the parity contract).
- **Attention forward** (`src/attn.cyr`) ported bit-identical from attn11
  `src/attn.cyr`: `attn_core_fwd` (causal scaled-dot-product, log-sum-exp softmax,
  GQA head-sharing, 4-lane `f64v_fmadd` dot + PV axpy) and `attn_fwd` (Q/K/V via
  `linear_fwd` → core → output proj). Forward-only arena (`attn_arena_size`) with
  the `_at_Qc`..`_at_concat` sub-pointers identical to attn11 (its full arena drops
  in unchanged). GPU / bidirectional-diffusion / RoPE paths dropped (learned-abs,
  CPU-only; RoPE is a later increment carried as the `pos_kind` param).
- **MLP forward** (`src/mlp.cyr:mlp_fwd`): `linear_fwd` → `gelu_fwd` → `linear_fwd`.
- **Embedding forward** (`src/embed.cyr:embed_fwd`): token lookup + learned
  ABSOLUTE positional embedding (from attn11 `embed_fwd_n`).
- **Weight-tied LM head + softmax** (`src/head.cyr`): `head_fwd_row`/`head_fwd`
  (logits[v] = f · tok_emb[v], the same 4-lane SIMD dot as attn11) and
  `softmax_fwd` (row-wise max-subtracted softmax — the probability half of
  attn11's `softmax_xent_fwd`).
- **Pre-norm block + block-stack model forward** (`src/blocks.cyr`): a caller-owned
  packed-parameter layout (byte-identical to attn11's MHA/dense block) + config
  descriptor (`ru_cfg_init`/`ru_params_count`); `block_fwd`
  (x' = x + Attn(LN(x)); x'' = x' + MLP(LN(x')), in-place-safe) and `model_fwd`
  (embed → NL blocks → final LN → weight-tied head → softmax). No dropout — this
  is the inference forward, bit-identical to attn11's `model_forward` with dropout
  off. GPT-2 pre-norm decoder (Vaswani 2017 / Xiong 2020 / Radford 2019).
- Tests: `tests/tcyr/attn.tcyr` (**12** — attention known-answer + identity-proj)
  and `tests/tcyr/forward.tcyr` (**21** — embed/mlp/head/softmax leaf checks +
  `model_fwd` plumbing parity [zeroed-weight pass-through == embed→LN→head,
  byte-identical] + attention-contributes + softmax-normalization). Suite total **43**.

## [0.1.0] — Unreleased (folded into the 0.2.0 first release)

**M0 — buildable scaffold.** Repo skeleton for the transformer-forward library
extracted from attn11. Never tagged on its own — 0.2.0 is the first release.

### Added
- Scaffold: `cyrius.cyml` (lib; pin 6.3.27), `src/lib.cyr` include chain,
  `src/blocks.cyr` (version probe), `programs/smoke.cyr` link-check,
  `tests/tcyr/smoke.tcyr` (1 assertion), CI + release workflows, docs, Makefile.
