# rupantara — Claude Code Instructions

## Identity

`rupantara` (रूपान्तर — *transformation*) — AGNOS's sovereign **transformer
forward** library (attention / MLP / norm / block / embeddings / LM head /
KV-cache decode), **extracted from attn11** so consumers run a transformer
without depending on attn11-the-binary. Pure Cyrius. GPL-3.0-only.

**Scope — IS / IS NOT:**
- IS: the transformer **forward** (inference).
- IS NOT: training/backprop (that stays in **attn11**); tensors (**rosnet**);
  the importer/adapt reference (**anukūlana**); the weight format (**tula**).

## Structure & conventions

- `src/lib.cyr` — the include chain. **Stdlib includes only here**; `src/*.cyr`
  domain modules are flat (no includes) so `cyrius distlib` produces a clean
  `dist/rupantara.cyr`. New module → add to `src/lib.cyr` AND `cyrius.cyml
  [lib].modules` (dependency order).
- `programs/smoke.cyr` — link-check (no CLI binary). `tests/tcyr/*.tcyr` —
  standalone suites (own `main` + `assert_summary`). `tests/common.cyr` (shared
  helpers) and `tests/reference.cyr` (oracles: the 0.5.1 attention cores verbatim
  plus a naive forward) are included by suites, not run. `tests/rupantara.fcyr` is
  the fuzz harness, `tests/rupantara.bcyr` the benchmarks.
- **Never `cat | cycc`** — always `cyrius build`. `lib/` is a `cyrius deps`
  artifact (gitignored), never a symlink to a cyrius checkout. `cyrius.lock`
  gitignored.
- The Cyrius pin (`cyrius = "…"` in `cyrius.cyml`) is the single source of truth.
- **Optimizations must keep every output bit-identical** (ADR 0002): the attn11
  parity contract. Gate a kernel change with the oracles in `tests/reference.cyr`
  and the committed golden gate (`tests/tcyr/golden.tcyr` against
  `tests/golden/<target>.txt`). `make golden-bless` only for a reviewed
  numerical change, named in the CHANGELOG.
- **Model entries validate, leaf kernels trust** (ADR 0003). The error codes are
  functions (`RU_ERR_*()`), not `const`, so the bundle compiles on pre-6.7 cyrius.
- anukūlana calls the underscore layout helpers (`_cfg_*`, `_ru_o_*`, `_p_*`,
  `_b?`, `_ru_blk_base`). Do not rename them (`docs/api.md`).
- **Do not bump VERSION; do not run git** — the maintainer cuts releases. New
  work accretes under CHANGELOG `[Unreleased]`.

## Build / test

```sh
make build   # cyrius build programs/smoke.cyr build/rupantara_smoke
make test    # cyrius test tests/tcyr/*.tcyr (the golden gate among them)
make test-aarch64  # every suite cross-built for aarch64, run under qemu-aarch64
make fuzz    # seeded property fuzzing (tests/rupantara.fcyr)
make bench   # benchmarks; scripts/bench-history.sh records medians
make dist    # cyrius distlib -> dist/rupantara.cyr
```

Definition of done each bite: `make test` and `make fuzz` green · `make fmt-check`
clean · `make lint` no `warn ` · `make doc` clean · `make dist` regenerated. See `docs/development/roadmap.md` for
the 0.6 → 1.0 plan and the post-1.0 backlog. Cross-repo items (attn11, anukūlana)
each need the maintainer's explicit go.
