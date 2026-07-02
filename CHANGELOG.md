# Changelog

All notable changes to `rupantara` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); SemVer (pre-1.0 — the public
surface is still moving, no API freeze until v1.0).

## [0.1.0] — Unreleased

**M0 — buildable scaffold.** Repo skeleton for the transformer-forward library
extracted from attn11.

### Added
- Scaffold: `cyrius.cyml` (lib; pin 6.3.27, stdlib-only), `src/lib.cyr` include
  chain, `src/blocks.cyr` (version probe stub), `programs/smoke.cyr` link-check,
  `tests/tcyr/smoke.tcyr` (1 assertion), CI + release workflows, docs (ADR 0001
  scope, roadmap, state), README/CHANGELOG/CLAUDE/LICENSE/Makefile.

### Notes
- The transformer forward is extracted from attn11 at **M1** (see
  `docs/development/roadmap.md`). M1 adds `math` + `rosnet` deps and is a
  **cross-repo** change (the attn11-side re-point needs its own go).
