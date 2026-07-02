# rupantara — Current State

> **Last refresh**: 2026-07-02 | **Cadence**: every release.
> `CLAUDE.md` is preferences/process; this file is volatile state.

## Version

**0.2.0 (+ [Unreleased]) — M1 the transformer forward, ported from attn11.** The
GPT-2-shaped forward ported verbatim: LayerNorm (`ru_ln_fwd`, ε=1e-5), GELU
(`ru_gelu_fwd`), the attention core + `ru_attn_fwd` (causal softmax + GQA),
`ru_mlp_fwd`, `ru_embed_fwd` (token + learned-abs pos), `ru_head_fwd`/
`ru_softmax_fwd` (weight-tied LM head), and the pre-norm `ru_block_fwd` +
block-stack `ru_model_fwd` over a packed-param layout byte-identical to attn11's
MHA/dense block. Suite: `smoke` 1 + `ops` 9 + `attn` 12 + `forward` 21 = **43
assertions** green; builds warning-clean.

**Re-fold landed 2026-07-02 (`[Unreleased]`):** the full op surface is
`ru_*`/`_ru_*`-namespaced (12 public + 28 private renamed; `comm -12` vs attn11 =
empty), and **attn11 consumes rupantara's three leaf ops** — `ru_ln_fwd`,
`ru_gelu_fwd`, `ru_attn_core_fwd` (causal). attn11's full grad-check suite is
**1049 green in one binary** with them delegated → those three are **proven
bit-identical** (the real parity gate; no offline compare).

**Parity status (honest):** *proven* only for the three delegated leaf ops.
rupantara's **composition** ops (`ru_embed_fwd`, `ru_head_fwd`, `ru_head_fwd_row`,
`ru_mlp_fwd`, `ru_block_fwd`, `ru_model_fwd`, `ru_softmax_fwd`) are ported +
internally tested but attn11 keeps its own (incompatible global-reading
signatures / training concerns) → **not yet cross-validated** against attn11. A
rupantara↔attn11 whole-`ru_model_fwd` parity fixture is still worthwhile before
anukūlana's fidelity gate leans on it. attn11's GPU/bidir/row/MLA/SSM/MoE/MTP
paths are not de-duplicated (rupantara has no peer, by design).

## Toolchain

Cyrius pin **6.3.27** (`cyrius.cyml`). Deps: `math` (f64_exp/F64_PI) + `ganita`
(f64_tanh); **`rosnet` 0.2.0** (`linear_fwd` matmul + tensor helpers) via
`[deps.rosnet]`, git+tag (no path override — resolves from the release tag).

## Build artifacts

- `programs/smoke.cyr` → `build/rupantara_smoke` (link-check).
- `dist/rupantara.cyr` — distlib bundle for consumers (`[deps.rupantara]`).
- CI + release workflows in `.github/workflows/`. `cyrius.lock` gitignored.

## Next

**M1 DONE** (forward ported + `ru_*` namespaced + attn11 leaf-op re-fold green).
Open follow-ons: (a) a **rupantara↔attn11 whole-`ru_model_fwd` parity fixture**
(fixed tiny model + input, logits bit-for-bit) to cross-validate the composition
ops attn11 does not exercise — do this *before* anukūlana's fidelity gate leans on
`ru_model_fwd`; (b) **M2 — KV-cache decode** (incremental forward bit-identical to
the uncached path). See `roadmap.md`.
