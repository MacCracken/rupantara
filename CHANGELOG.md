# Changelog

All notable changes to `rupantara` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); SemVer (pre-1.0 — the public
surface is still moving, no API freeze until v1.0).

## [Unreleased]

### Next (M1 remaining — cross-repo, maintainer's go)
- The **attn11-side re-point**: make attn11 consume `dist/rupantara.cyr` (attn11
  becomes a thin consumer of its own extracted forward) and run the full
  **bit-identical-vs-attn11 parity gate**. This edits the **attn11** repo — its own
  explicit go. The rupantara side shipped green + standalone in 0.2.0.

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
