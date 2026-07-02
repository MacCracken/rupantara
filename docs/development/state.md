# rupantara — Current State

> **Last refresh**: 2026-07-02 | **Cadence**: every release.
> `CLAUDE.md` is preferences/process; this file is volatile state.

## Version

**0.3.0 (released 2026-07-02) — M1 the transformer forward, ported from attn11 +
`ru_*`-namespaced + attn11 leaf-op re-fold.** The GPT-2-shaped forward ported
verbatim: LayerNorm (`ru_ln_fwd`, ε=1e-5), GELU
(`ru_gelu_fwd`), the attention core + `ru_attn_fwd` (causal softmax + GQA),
`ru_mlp_fwd`, `ru_embed_fwd` (token + learned-abs pos), `ru_head_fwd`/
`ru_softmax_fwd` (weight-tied LM head), and the pre-norm `ru_block_fwd` +
block-stack `ru_model_fwd` over a packed-param layout byte-identical to attn11's
MHA/dense block. Suite: `smoke` 1 + `ops` 9 + `attn` 12 + `forward` 21 = **43
assertions** green; builds warning-clean.

**Re-fold landed 2026-07-02 (shipped in 0.3.0):** the full op surface is
`ru_*`/`_ru_*`-namespaced (12 public + 28 private renamed; `comm -12` vs attn11 =
empty), and **attn11 consumes rupantara's three leaf ops** — `ru_ln_fwd`,
`ru_gelu_fwd`, `ru_attn_core_fwd` (causal). attn11's full grad-check suite is
**1049 green in one binary** with them delegated → those three are **proven
bit-identical** (the real parity gate; no offline compare).

**Parity status — FULLY PROVEN (2026-07-02):** both the delegated leaf ops **and
the whole composition forward** are bit-identical to attn11. Leaf ops via attn11's
1049-in-one-binary gate; the composition (`ru_embed_fwd`, `ru_head_fwd`,
`ru_head_fwd_row`, `ru_mlp_fwd`, `ru_block_fwd`, `ru_model_fwd`, `ru_softmax_fwd`)
via **`test_rupantara_parity`** in attn11's suite — `ru_model_fwd` feeds attn11's
`g_params` directly and matches `model_forward` **bit-for-bit** (`diffs==0`,
`maxrel=0.000000000`) across 4 configs (MHA ±bias / GQA `nkv<nh` / MQA `nkv=1` /
1–3 blocks); attn11 suite **1057 green**. The M1-acceptance parity gap is CLOSED —
anukūlana's fidelity gate can lean on `ru_model_fwd`. attn11's GPU/bidir/row/MLA/
SSM/MoE/MTP paths are not de-duplicated (rupantara has no peer, by design).

## Toolchain

Cyrius pin **6.3.27** (`cyrius.cyml`). Deps: `math` (f64_exp/F64_PI) + `ganita`
(f64_tanh); **`rosnet` 0.2.0** (`linear_fwd` matmul + tensor helpers) via
`[deps.rosnet]`, git+tag (no path override — resolves from the release tag).

## Build artifacts

- `programs/smoke.cyr` → `build/rupantara_smoke` (link-check).
- `dist/rupantara.cyr` — distlib bundle for consumers (`[deps.rupantara]`).
- CI + release workflows in `.github/workflows/`. `cyrius.lock` gitignored.

## Next

**M1 DONE** (forward ported + `ru_*` namespaced + attn11 leaf-op re-fold green +
whole-forward parity PROVEN vs attn11). **M2 — KV-cache decode DONE** (2026-07-02):
`src/decode.cyr` — `ru_model_fwd_row` + per-layer K/V caches + `ru_argmax`;
`tests/tcyr/decode.tcyr` proves cached decode == uncached `ru_model_fwd` **bit-for-bit**
across 4 configs (MHA ±bias / GQA / MQA); suite **48** green. Next: **M3**
(architecture breadth — RoPE/RMSNorm/SwiGLU/GQA-extras, demand-gated) or hand off
to **anukūlana M1** (the importer, which only needs the batch forward). See `roadmap.md`.
