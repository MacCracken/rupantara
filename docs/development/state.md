# rupantara — Current State

> **Last refresh**: 2026-07-01 | **Cadence**: every release.
> `CLAUDE.md` is preferences/process; this file is volatile state.

## Version

**0.1.0 — unreleased.** **M0: buildable scaffold** for the transformer-forward
library extracted from attn11. Version probe only (`rupantara_version()`), smoke
+ 1 test green, dist bundle generated. No transformer forward yet — that is M1.

No released tags yet.

## Toolchain

Cyrius pin **6.3.27** (`cyrius.cyml`). Deps: **stdlib-only** at M0. M1 adds
`math` + `rosnet` (see the commented block in `cyrius.cyml`).

## Build artifacts

- `programs/smoke.cyr` → `build/rupantara_smoke` (link-check).
- `dist/rupantara.cyr` — distlib bundle for consumers (`[deps.rupantara]`).
- CI + release workflows in `.github/workflows/`. `cyrius.lock` gitignored.

## Next

**M1** — extract the minimum GPT-2 forward from attn11 (embed / LayerNorm /
softmax MHA / GELU MLP / block / LM head), add `math` + `rosnet`, parity-test
bit-identical vs attn11. **Cross-repo:** the attn11 re-point needs its own go.
See `roadmap.md`.
