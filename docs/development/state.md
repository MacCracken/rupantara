# rupantara — Current State

> **Last refresh**: 2026-07-02 | **Cadence**: every release.
> `CLAUDE.md` is preferences/process; this file is volatile state.

## Version

**0.2.0 — M1 (rupantara side complete): the transformer forward extracted from
attn11.** First tagged release. Ported bit-identical: LayerNorm (`ln_fwd`, ε=1e-5),
GELU (`gelu_fwd`), the attention core + `attn_fwd` (causal softmax + GQA), `mlp_fwd`,
`embed_fwd` (token + learned-abs pos), `head_fwd`/`softmax_fwd` (weight-tied LM
head), and the pre-norm `block_fwd` + block-stack `model_fwd` over a packed-param
layout byte-identical to attn11's MHA/dense block. Suite: `smoke` 1 + `ops` 9 +
`attn` 12 + `forward` 21 = **43 assertions** green; builds warning-clean.

M1 remaining — **cross-repo, maintainer's go:** the attn11 re-point (make attn11
consume `dist/rupantara.cyr`) + the full bit-identical-vs-attn11 parity run.

## Toolchain

Cyrius pin **6.3.27** (`cyrius.cyml`). Deps: `math` (f64_exp/F64_PI) + `ganita`
(f64_tanh); **`rosnet` 0.2.0** (`linear_fwd` matmul + tensor helpers) via
`[deps.rosnet]`, git+tag (no path override — resolves from the release tag).

## Build artifacts

- `programs/smoke.cyr` → `build/rupantara_smoke` (link-check).
- `dist/rupantara.cyr` — distlib bundle for consumers (`[deps.rupantara]`).
- CI + release workflows in `.github/workflows/`. `cyrius.lock` gitignored.

## Next

**M1** — extract the minimum GPT-2 forward from attn11 (embed / LayerNorm /
softmax MHA / GELU MLP / block / LM head), add `math` + `rosnet`, parity-test
bit-identical vs attn11. **Cross-repo:** the attn11 re-point needs its own go.
See `roadmap.md`.
