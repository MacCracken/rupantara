# Changelog

All notable changes to `rupantara` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); SemVer (pre-1.0 — the public
surface is still moving, no API freeze until v1.0).

## [Unreleased]

### Changed

- **`docs/development/roadmap.md` reorganized.** Completed milestones are
  removed: M0–M2, and M4 and M5, which the 0.5.2 sweep covered. The remaining
  work is planned as 0.6.x (consumers on real models), 0.7.x (the API shape the
  freeze will lock), 0.8.x (the freeze candidate) and the 1.0.0 cut.
  Architecture breadth (RoPE, RMSNorm, SwiGLU), the GPU and
  bit-contract-limited performance items, and ecosystem items move to a
  post-1.0, demand-gated backlog.
- New items found while planning:
  - a committed golden-dump gate;
  - a memory-lean batch forward. `ru_model_fwd`'s scratch reaches 6.84 GiB at
    GPT-2-small widths and T = 8192, past the 2 GiB allocation cap, and 6 GiB
    of it is attention probabilities the forward never re-reads;
  - pre-freeze decisions on how the config descriptor grows and where the GPU
    path lives.
- `state.md`, `CLAUDE.md` and the `src/gpu.cyr` comment point at the new
  sections.

## [0.5.2] - 2026-11-09

The P(-1) hardening, optimization and repair sweep, recorded in
[`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md). No public
signature changed. For valid input every output is bit-identical to 0.5.1: a
6,742-value raw-bits golden dump is byte-identical on x86_64, on aarch64
(qemu), and when built with cyrius 6.6.2.

### Fixed

- **Decoding past the context window wrote past the KV cache** (HIGH).
  `ru_model_fwd_row` at `pos >= T` returned 0 and appended each layer's K and V
  at `pos * Ckv`: into the next layer's region, and past the end of the buffer
  for the last layer's V. It also read past the learned position table. A
  position outside `[0, T)` now returns `RU_ERR_POS` before anything is
  written.
- **Configs were not validated.** `nh = 0` or `nkv > nh` divided by zero.
  `nh % nkv != 0` sent heads past the last K/V head, reading off the K/V rows.
  `C % nh != 0` left concat columns unwritten. `attn_bias = 2` read biases past
  the block. Sizes could wrap an i64. The new public `ru_cfg_check` rejects all
  of these, and every buffer size is now overflow-checked and capped at 2^56
  elements.
- **Token ids were not checked.** An id outside `[0, V)` read past the token
  embedding. Both forward entries now return `RU_ERR_TOKEN`.
- **Decode loops grew the heap without bound.** `ru_model_fwd_row` made 17
  allocations per token and `ru_model_fwd` 15 per call, all from a bump
  allocator that never frees: 106.5 MB per 1,000 steps at GPT-2 widths. The new
  `ru_model_fwd_s` and `ru_model_fwd_row_s` run on caller-owned scratch and
  allocate nothing (0 bytes over the same 1,000 steps). The allocating entries
  keep their behaviour but make one allocation per call.
- `ru_attn_fwd` silently ran learned-absolute attention for `pos_kind = 1`
  (RoPE). It now returns `RU_ERR_ARG`. `ru_argmax(…, 0)` read an empty row and
  now returns `RU_ERR_ARG`. `ru_attn_arena_size` divided by zero at `nh = 0`
  and now returns -1.
- Rejected calls write nothing: logits, probs and the KV cache keep their
  contents. Every error code is listed in `docs/api.md`.
- `tests/tcyr/gpu.tcyr` failed one case under aarch64, and 0.5.1 fails it too.
  `linear_fwd_gpu`'s padded-tile CPU fallback sends a ragged last tile's
  final `N % 4` columns through `linear_fwd`'s packed body, which is fused on
  aarch64. x86_64, the GPU target, is unaffected. The caveat is documented, and
  the test holds those columns to 1e-12 on aarch64 and every other element
  bit-exact. Every suite now passes under qemu-aarch64.

### Added

- `src/guard.cyr`: the error codes (`RU_OK`, `RU_ERR_CFG`, `RU_ERR_TOKEN`,
  `RU_ERR_POS`, `RU_ERR_ARG`, written as functions so the bundle still compiles
  on pre-6.7 cyrius) and overflow-checked size arithmetic.
- `ru_cfg_check`, `ru_model_fwd_s`, `ru_model_fwd_row_s`,
  `ru_model_scratch_count`, `ru_decode_scratch_count`.
- `docs/api.md`, `docs/benchmarks.md`, `SECURITY.md`, ADR 0002 (optimizations
  preserve bits) and ADR 0003 (validate at the model entries).
- **Tests: 59 → 164 assertions.**
  - `tests/tcyr/guard.tcyr` (78): validation and the no-write guarantee.
  - `tests/tcyr/reference.tcyr` (27): both attention cores held bit for bit to
    the 0.5.1 loops, kept verbatim in `tests/reference.cyr`, and the forward
    held within 1e-9 to an independent plain-loop implementation.
  - `tests/rupantara.fcyr`: 20,000 seeded fuzz rounds × 5 properties.
  - `tests/rupantara.bcyr`: 12 benchmarks.
  - `scripts/bench-history.sh` with `bench-history.csv`.
  - Eight planted bugs were all caught.

### Changed — faster, same bits

- **Attention 1.5×.** The PV accumulation is one `f64v_axpy` per key, and both
  cores run one shared per-head routine, so batch and decode cannot drift.
  - `ru_attn_core_fwd` (T=128, C=768, nh=12): 18.1 → 12.2 ms.
  - The cached row: 1.17 → 0.78 ms.
- **Embedding 9.9×** (201 → 20 µs), and the residual adds run on the same
  helper, `f64v_add`.
- The whole forward is 1.06× faster and a decode step 1.09×. See
  `docs/benchmarks.md`.

### Changed — CI, build, docs

- **Workflows.**
  - Actions are pinned by commit SHA.
  - The token is read-only by default; only the publish job can write.
  - `persist-credentials: false`.
  - The dist check now covers the `.deps` sidecar.
  - New gates: doc check, fuzz, benchmarks, the gpulayer build, and running
    the smoke binary.
  - The release fails if the bundle differs from the commit.
- **Lint and fmt** cover every Cyrius source, test helpers included. The
  Makefile gains `fuzz`, `bench` and `doc`.
- **Every public function is documented** (`cyrius doc --check` clean), and
  strict lint reports 0 warnings and 0 untracked deferrals.
- **The README, CLAUDE.md, `cyrius.cyml` description and ADR index** had
  drifted; they are corrected. The ADR index was another project's. The README
  and description claimed RMSNorm, which is not implemented.
- **Suites and programs** exit through `SYS_EXIT` and write through
  `sys_write`, not literal syscall numbers.

### Corrected

- **0.4.4 said "No change to the forward". It did change.** cyrius 6.6.13
  (ganita 1.2.11) replaced `f64_tanh` with fdlibm's `expm1` form, an accuracy
  fix. GELU differs by up to 2 ulp between builds on cyrius ≤ 6.6.12 and
  ≥ 6.6.13, and model outputs by up to 2.7e-13 relative. LayerNorm, softmax,
  `f64_exp` and `linear_fwd` did not move.
- attn11's parity test is unaffected: it compiles both sides with one
  toolchain. attn11's own GELU outputs will shift by these amounts when it
  moves off 6.6.2.

## [0.5.1] - 2026-11-09

A toolchain and dependency realign onto the latest releases. No change to the
forward's results or the API.

### Changed

- **Toolchain `6.7.9` → `6.7.18`** (through 6.7.10–6.7.17), the latest cyrius
  release. `src/` needed no code change. Suite **59/59** across 6 files,
  identical to 0.5.0. fmt and vet are clean, and lint stays at 0 warnings
  under 6.7.11's and 6.7.12's new lint rules. The DCE smoke binary shrinks from
  38,072 to 37,584 B. The 6.7.10–6.7.18 behaviour changes are almost all new
  compile-time refusals (generics, struct and pointer shapes, enums), plus
  runtime changes in net, TLS, thread-local slots and pam. rupantara uses none
  of these, and the build, the tests and the golden dump below confirm it.
- **`[deps.rosnet]` `1.2.0` → `1.2.2`**, the latest rosnet, built against
  6.7.18. 1.2.2 rewrites `linear_fwd`, the projection matmul behind every
  rupantara layer: one packed `f64v_axpy` per row step instead of
  `f64v_fmadd` per four elements. It is about 2.7× faster at 64×768×768.
  rosnet's ADR 0002 keeps each output's rounding sequence (on x86_64 the axpy
  is unfused `mulpd` + `addpd`), and rosnet's own golden dump is bit-identical.
  **Checked independently here.** A scratch golden dump recorded 6,742 raw f64
  bit patterns, built at 0.5.0 (6.7.9 + rosnet 1.2.0) and again at 0.5.1
  (6.7.18 + rosnet 1.2.2), and the two are byte-identical. The dump covers
  `ru_model_fwd` logits and probs, the cached `ru_model_fwd_row` decode and
  `ru_argmax`, over 6 configs (MHA ±bias, GQA, MQA, 3 blocks, odd V = 37 / C = 24,
  C = 64). It also covers `ru_ln_fwd`, `ru_gelu_fwd`, `ru_mlp_fwd`,
  `ru_softmax_fwd`, `linear_fwd` and `linear_fwd_gpu` at shapes with 1–3
  element vector tails. So the attn11 parity contract is untouched.
  rosnet's other 1.2.2 fixes are in its GPU profile and in `t_alloc(0)`, which
  rupantara does not call. rosnet now pulls tyche `1.2.2` in transitively.
- **Vendored `lib/` re-synced** (`cyrius lib sync --full`). It byte-matches the
  6.7.18 snapshot. `cyrius.lock` was regenerated from scratch: one line per git
  dep, trailer `cyrius 6.7.18`. `cyrius deps --verify` and `cyrius deps --locked`
  are clean. rupantara gitignores both `lib/` and the lock, so CI's fresh
  `cyrius deps` is not affected by rosnet 1.2.1's note that a moved pin, `lib/`
  and the lock must land together.
- **`src/gpu.cyr`'s bit-identity note** names `f64v_axpy` as `linear_fwd`'s body
  (it named `f64v_fmadd`). The argument it makes is unchanged.
- `dist/rupantara.cyr` changes only in its version line, the version constant
  and that comment. `dist/rupantara.deps` is unchanged.

## [0.5.0] - 2026-11-09

Moves rupantara onto the cyrius 6.7 series. No change to the forward or the API.

### Changed

- **Toolchain `6.6.20` → `6.7.9`** (through 6.7.0–6.7.8). `src/` needed no
  change. Suite **59/59** across 6 files, identical to 0.4.4. fmt, lint and vet
  are clean. The DCE smoke binary stays at 38,072 B.

  Upstream changes checked against rupantara's source (`src/`, `programs/`,
  `tests/`):
  - **6.7.0–6.7.8** reserve `trait`, `const`, `true`, `false`, `do` and `dyn`.
    rupantara uses none of them outside comments.
  - **6.7.3** makes `!` a real operator. Before that a lone `!x` compiled as
    `x`. rupantara has no `!` outside `!=`.
  - **6.7.9** makes unary `-` and `~` on a constant bind tighter, and
    **6.7.8–6.7.9** change how typed `*T` pointers step and dereference.
    rupantara has neither: its buffers are untyped addresses read with
    `load64` / `store64`.
  - **6.7.6** rounds every write into an `f32`. rupantara has no `f32`.
  - The stdlib functions the forward computes with did not change. In `math`,
    only `f64_parse` changed (a length-bounded `f64_parse_n` was added), and
    rupantara does not call it. `ganita` changed only its version line
    (`1.2.14` → `1.2.15`). So `f64_exp` and `f64_tanh` give the same results
    as before. That keeps the attn11 parity contract intact while attn11 is
    still on an older pin.
- **`[deps.rosnet]` `1.1.3` → `1.2.0`**, the rosnet release built against
  6.7.9. rosnet's `src/` is unchanged between the two tags. rosnet reports its
  CPU ops give byte-identical f64 results at 1.1.3 @ 6.6.20 and 1.2.0 @ 6.7.9,
  `linear_fwd` included. rosnet now pulls tyche `1.2.0` in transitively.
- **Vendored `lib/` re-synced** (`cyrius lib sync --full`). It byte-matches the
  6.7.9 snapshot. `cyrius.lock` was regenerated from scratch: one line per git
  dep, trailer `cyrius 6.7.9`. `cyrius deps --verify` is clean.
- `dist/rupantara.cyr` changes only in its version line and the version
  constant. `dist/rupantara.deps` is unchanged.

## [0.4.4] - 2026-11-09

A toolchain and dependency realign. No change to the forward.

### Changed

- **Toolchain `6.6.2` → `6.6.20`.** Suite **59/59** across 6 files, identical to
  the 6.6.2 baseline. fmt, lint and vet are clean. The DCE smoke binary grows
  from 37,752 to 38,072 B.
- **`[deps.rosnet]` `1.1.1` → `1.1.3`**, the rosnet release built against
  6.6.20. rosnet's CPU bundle changed only in whitespace between the two tags.
  The code changes are in its GPU backend, which rupantara does not use. So
  `linear_fwd` is unchanged and the attn11 parity contract is untouched. rosnet
  pulls tyche `1.1.1` in transitively. rupantara never calls `t_randn`, the one
  function that needs it.
- **Vendored `lib/` re-synced** (`cyrius lib sync --full`). It byte-matches the
  6.6.20 snapshot. This clears the warning that `./lib/` shadowed 11 pinned
  bundled libs (bayan, sakshi, niyama, sigil, sandhi, yukti, patra, vani, mabda,
  sankoch, yantra). `cyrius.lock` was regenerated from scratch, and
  `cyrius deps --verify` is clean.
- **`dist/` regenerated.** From 6.6.18, `cyrius distlib` lists only the stdlib
  leaves the bundle provably needs (compile-verified), and also writes them as
  `include` lines at the top of the bundle. `dist/rupantara.deps` goes from 11
  leaves to 2 (`math` and `ganita`). `linear_fwd` and the tensor helpers stay
  unresolved in the bundle, as before, so consumers still declare
  `[deps.rosnet]`.

### Fixed

- **`rupantara_version()` drifted again.** Through 0.4.3 it still returned `402`.
  It is now `404`, and the `smoke.tcyr` assertion moves with it. CI's docs job
  now also checks the packed constant in `src/blocks.cyr` against `VERSION`, so
  a third drift fails the build.

## [0.4.3] - 2026-09-11

### Changed

- **Toolchain `6.5.27` → `6.6.2`.** No source change; the value form needed none.
  Build, tests, and any bench/fuzz/distlib target the repo ships re-verified at the new pin.


## [0.4.2] — 2026-08-17

### Changed

- **Cyrius pin `6.4.72` -> `6.5.27`** (2026-08-17, ecosystem-wide ML/AI-arc realign ahead of
  the arc reopening). `cyrius lib sync --full` re-vendored the whole version-matched stdlib
  snapshot, clearing the toolchain-drift and `./lib/ shadows version-pinned` warnings.
  Suite **43/43** across 5 files, identical to the pre-bump baseline; the attn11 parity contract is untouched.

### Fixed

- **`rupantara_version()` had been stale since 0.4.1** — it still returned `400` (0.4.0) after
  that cut moved `VERSION` and the CHANGELOG but not the constant. `tests/tcyr/smoke.tcyr`
  asserted the stale value with a message naming 0.4.0, so the gate *confirmed* the drift
  rather than catching it. Now `402`, with the assertion and its message moved in step and a
  note in `src/blocks.cyr` requiring both to move on every cut.

### Changed

- **`[deps.rosnet]` `0.2.0` -> `1.1.1`** — the substrate-freeze propagation reaches rupantara.
  rosnet froze its public surface at **1.0.0** (2026-07-04) and added conv2d/conv1d at 1.1.0, and
  1.1.1 carries the toolchain realign plus its own tyche bump. rupantara consumes `linear_fwd` and
  the tensor helpers, all inside the frozen CPU bundle, so this is a tag realign with no behavior
  change and the attn11 parity contract is untouched. Verified the bump took, not merely built —
  vendored `lib/rosnet.cyr` moved to `1.1.1`. Suite **48/48**, unchanged.

## [0.4.1] — 2026-07-23

**AGNOS GPU offload (the 1.54.x GPU crown / C6): rupantara's f64 projection matmul can run on the AMD gfx90c
shader cores.** Additive and OFF by default — `linear_fwd_gpu` is a new peer of `linear_fwd`; `ru_mlp_fwd` /
`ru_block_fwd` are unchanged — so the CPU forward and the attn11 parity contract are untouched.

### Added
- **`src/gpu.cyr` — `linear_fwd_gpu(x, W, b, y, M, K, N)`**, a bit-identical peer of rosnet `linear_fwd` that
  tiles the matmul into 8×8×8 blocks and dispatches each on the AMD Cezanne gfx90c shader cores via the AGNOS
  ring-3 syscall `#83` (`gpu_dispatch_f64`), one tile per call, gathering strided sub-blocks CPU-side.
  **Bit-identical** (not tolerance) for the **bias-free, K≤8** regime: `#83` computes `C=A*B` unfused
  (`v_mul_f64` then `v_add_f64`, k-ascending) and `linear_fwd`'s f64 body is cyrius `EMIT_F64V_FMADD` = SSE2
  `mulpd+addpd` (also unfused, two roundings, same order), and K≤8 is a single k-tile (no cross-tile f64
  reassociation). On a non-AGNOS target, or when `#83` returns <0 (no GPU / boot self-test unproven), each
  tile falls back to `linear_fwd` — identical result, so the tiling is exercisable off-hardware. Returns the
  count of tiles that ran on the GPU. ⚠ K>8 (cross-tile accumulation) is a ~1 ULP tolerance match, not
  bit-exact — a follow-on; bias must be 0 (the association would otherwise change).
- **`programs/gpulayer.cyr` — the crown proof.** Runs a real bias-free MLP up-projection
  `linear_fwd(x, Wfc, 0, fc, 8, 8, 32)` (the first op of `ru_mlp_fwd`) twice — CPU `linear_fwd` vs
  `linear_fwd_gpu` — on deterministic **fractional** f64 data (products not exactly representable, so a match
  proves the GPU rounds exactly like the CPU) and byte-compares the 2048-byte outputs. Exit code is the
  oracle: **95** = byte-identical AND all 4 tiles on the GPU (crown); **96** = byte-identical, 0 GPU tiles
  (host/QEMU — plumbing proven, no GPU); **90** = byte mismatch (the guard against a silently-wrong gather).

### Verified
- **Tiling bit-identical on host** (2026-07-23): `build/gpulayer` exits **96** — `linear_fwd_gpu` (CPU-tiled)
  is byte-for-byte equal to the production `linear_fwd` over the full 8×32 output, proving the gather/scatter
  arithmetic. Both `programs/gpulayer.cyr` and `programs/smoke.cyr` build clean `--agnos`.
- **CROWN PROVEN ON IRON** (2026-07-23): `run /bin/gpulayer` → **`exit 95`** on archaemenid (AMD Cezanne) —
  the outputs were byte-identical AND all 4 tiles dispatched on the GPU, so a real rupantara MLP up-projection
  matmul executed on the gfx90c shader cores, bit-identical to the CPU `linear_fwd`. This is the AGNOS 1.54.x
  GPU crown (C6) — the first ML layer's matmul on the shader cores. Tracked in agnosticos
  `iron-nuc-zen-log.md#tracker-156x-ml-crown`.

## [0.4.0] — 2026-07-02

**KV-cache decode (M2) + whole-forward parity proof.** rupantara now runs an
incremental cached decode **bit-identical** to the uncached forward, and its whole
`ru_model_fwd` composition is **proven bit-identical** to attn11's `model_forward`.
Additive over 0.3.0 (new `ru_*_row` / `ru_decode_*` / `ru_argmax` surface; no breaking change).

### Added
- **M2 — KV-cache decode** (`src/decode.cyr`). Incremental single-token forward:
  `ru_attn_core_fwd_row` / `ru_attn_fwd_row` (single-row causal attention core +
  Q/K/V/O projections, appending to per-layer K/V caches — ported from attn11
  `attn_*_fwd_row`), `ru_model_fwd_row` (one decode step: embed@pos → pre-norm
  blocks over the caches → final LN → weight-tied head), `ru_decode_kv_count` +
  `_ru_kv_k`/`_ru_kv_v` (caller-owned cache layout `[K:NL·T·Ckv][V:NL·T·Ckv]`), and
  `ru_argmax` (greedy next-token). Reuses `ru_ln_fwd`/`ru_mlp_fwd`/`ru_head_fwd_row`
  on one row; learned-absolute positions, causal MHA/GQA (M1 scope). **Bit-identical
  to the uncached path:** `tests/tcyr/decode.tcyr` asserts cached decode ==
  `ru_model_fwd` bit-for-bit (`diffs==0`) across 4 configs (MHA ±bias / GQA `nkv<nh`
  / MQA `nkv=1` / 1–3 blocks) + a greedy-generation smoke. Suite 43 → **48**.

### Verified
- **Whole-forward parity vs attn11 — PROVEN bit-identical** (2026-07-02). A new
  `test_rupantara_parity` in **attn11's** grad-check suite feeds attn11's `g_params`
  directly into `ru_model_fwd` and asserts the logits match attn11's own
  `model_forward` **bit-for-bit** (`diffs==0`, `maxrel=0.000000000`) across 4 configs
  (MHA ±bias / GQA `nkv<nh` / MQA `nkv=1` / 1–3 blocks), in one binary — no offline
  compare. This closes the 0.3.0 "Known limitations": rupantara's **composition** ops
  (`ru_embed_fwd`/`ru_head_fwd`/`ru_model_fwd`/…), not just the delegated leaf ops,
  are now cross-validated against attn11. No rupantara code change — the proof lives
  where both forwards coexist (attn11 links `lib/rupantara.cyr`). attn11 suite 1057 green.

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
