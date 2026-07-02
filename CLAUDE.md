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
  standalone suites (own `main` + `assert_summary`).
- **Never `cat | cycc`** — always `cyrius build`. `lib/` is a `cyrius deps`
  artifact (gitignored), never a symlink to a cyrius checkout. `cyrius.lock`
  gitignored.
- Cyrius pin `cyrius = "6.3.27"` in `cyrius.cyml` (single source of truth).
- **Do not bump VERSION; do not run git** — the maintainer cuts releases. New
  work accretes under CHANGELOG `[Unreleased]`.

## Build / test

```sh
make build   # cyrius build programs/smoke.cyr build/rupantara_smoke
make test    # cyrius test tests/tcyr/*.tcyr
make dist    # cyrius distlib -> dist/rupantara.cyr
```

Definition of done each bite: `make test` green · `cyrius fmt` clean · `cyrius
lint` no `warn ` · `make dist` regenerated. See `docs/development/roadmap.md` for
the march to v1.0 (M1 = the attn11 forward extraction — a cross-repo change that
needs its own explicit go for the attn11 re-point).
