# rupantara — Roadmap (0.6 → 1.0, then the backlog)

> **For a secondary agent picking this up cold.** Self-contained plan. Read this
> + `CLAUDE.md` (conventions) + `docs/adr/0001-rupantara-scope.md` (scope) +
> `docs/api.md` (the surface) + `docs/development/state.md` (where it stands).
> rupantara is the transformer **forward** library extracted from attn11 — not
> training, not tensors, not the importer. Pre-1.0: surface may move, no freeze
> until v1.0.

## Where it stands (0.5.2)

The GPT-2-shaped forward and KV-cache decode are done. That covers MHA/GQA/MQA
attention, the GELU MLP, LayerNorm, learned positions and the weight-tied head.
Both are bit-identical to attn11's forward: attn11's `test_rupantara_parity`
checks it, and attn11's leaf ops run through rupantara.

The model entries validate their input, and allocation-free caller-scratch
entries exist. A fuzz harness, benchmarks and a security audit are in place.
Completed work is recorded in `CHANGELOG.md` and `docs/development/state.md`,
not here.

---

## Working agreement

- **Build/test:** `make build` · `make test` · `make fuzz` · `make bench` ·
  `make dist`. The pin is in `cyrius.cyml`.
- **Flat modules:** stdlib includes only in `src/lib.cyr`; `src/*.cyr` flat. New
  module → `src/lib.cyr` + `cyrius.cyml [lib].modules` (dep order). The LSP flags
  cross-module symbols as "undefined" in standalone files — flat-module pattern;
  the build is authoritative.
- **Definition of done each bite:** `make test` and `make fuzz` green ·
  `make fmt-check` clean · `make lint` no `warn ` · `make doc` clean ·
  `make dist` regenerated · CHANGELOG `[Unreleased]`.
- **Do NOT bump VERSION or git** — the maintainer cuts releases; work accretes
  under `[Unreleased]`.
- **Cross-repo work** (attn11, anukūlana) is listed here so the plan is
  complete, but each item needs the maintainer's explicit go. Do not edit
  another repo without it.

## Discipline

- Cyrius-native, no BLAS/libc/autodiff. The correctness gate is **bit-identical
  parity with attn11's forward**. Any op ported from attn11 must reproduce
  attn11's numbers.
- **Optimizations keep every output bit-identical** (ADR 0002). Gate a kernel
  change with the oracles in `tests/reference.cyr` and a before/after golden
  dump on x86_64 and aarch64.
- **Model entries validate, leaf kernels trust** (ADR 0003). New entries return
  `RU_ERR_*` and write nothing on a rejection.

---

## 0.6.x — consumers on real models

**Goal:** anukūlana runs a real GPT-2 through rupantara end to end, at any run
length, and both consumers are on a current rupantara.

- **A committed golden-dump gate (first, because it guards the rest).** Today's
  golden dumps are ad hoc: 6,742 raw f64 bit patterns over every public forward
  op. Commit one file per target (x86_64, aarch64) and a suite that compares
  against it. A change that moves a bit then fails until it is reviewed and
  re-blessed with a CHANGELOG note. This gate would have caught cyrius 6.6.13's
  `f64_tanh` change at the 0.4.4 pin bump.
- **Variable-length forward: `ru_model_fwd_n` / `ru_model_fwd_n_s`.** Run
  `n <= T` tokens without repacking. `T` stays the layout's context (the
  `pos_emb` size), and `n` is the run length.
  - Today `ru_model_fwd` always runs `T` tokens. anukūlana's
    `anuk_gpt2_import` overrides `T` to the run length and repacks the whole
    parameter buffer (about 1 GB for GPT-2 small) for every run length.
  - **Acceptance:** rows `0..n-1` are bit-identical to `ru_model_fwd` on a
    padded window. Fuzz covers it.
- **A memory-lean batch forward.** `ru_model_fwd(_s)` sizes its scratch for the
  attention probabilities of the whole window (`nh·T·T` f64). The model forward
  never reads them again. At GPT-2-small widths the scratch is 0.20 GiB at
  T = 1024, 1.92 GiB at T = 4096, and 6.84 GiB at T = 8192. The last is past
  the allocator's 2 GiB per-allocation cap; 6.0 GiB of it is probabilities.
  - **The change:** run the shared `_ru_attn_qrow` with one T-element row
    instead. The scratch becomes 0.84 GiB at T = 8192, and the outputs stay
    bit-identical (same routine, same order).
  - `ru_attn_core_fwd` keeps writing the full `Pc`, because attn11's backward
    reads it.
- **GPT-2-scale throughput.** Add tokens/s for the batch forward and a decode
  step at GPT-2-small dims (C = 768, 12 layers, V = 50257, T = 1024) to
  `tests/rupantara.bcyr` and `docs/benchmarks.md`. Synthetic weights suffice:
  values do not change the time.
- **⚠ Cross-repo, anukūlana:**
  - move its decode loop (`src/main.cyr`) onto `ru_model_fwd_row_s`, since it
    allocates about 107 KB per token today;
  - drop the run-length repack once `ru_model_fwd_n` lands;
  - check the `RU_ERR_*` returns;
  - move `[deps.rupantara]` off 0.4.2.
- **⚠ Cross-repo, attn11:**
  - move it off cyrius 6.6.2, rosnet 1.1.2 and rupantara 0.4.2;
  - re-run `test_rupantara_parity` and the 1,049 gradient checks at the new pin.

  attn11's own GELU outputs will move by up to 2 ulp from the 6.6.13 `tanh`
  change. The parity diff should stay 0, because both sides are built with one
  toolchain.
- **Acceptance:** anukūlana's GPT-2 fidelity gate is green through rupantara
  (the v1.0 consumer criterion), and attn11 is green on a current rupantara.

## 0.7.x — the API shape the freeze will lock

**Goal:** make every decision that would be expensive to change after 1.0.

- **An extensible config.** The descriptor is a fixed `i64[8]`, and anukūlana
  allocates exactly 64 bytes for it. RoPE, RMSNorm and SwiGLU each need fields:
  the position kind and RoPE base, the norm kind and eps, the MLP kind.
  - Choose the growth path now. One option is a larger descriptor whose
    reserved slots mean "the GPT-2 default" when 0, sized by a function such as
    `ru_cfg_slots()`. Then post-1.0 architecture work is additive.
  - Migrate the consumers in the same release.
- **Public names for the layout helpers.** anukūlana uses 22 underscore helpers
  as the layout API (`_cfg_*`, `_ru_o_*`, `_p_*`, `_b?`, `_ru_blk_base`).
  Either freeze them as they are, or give them public names and keep the
  underscore names as aliases through 1.x.
- **The allocating entries.** `ru_model_fwd` and `ru_model_fwd_row` allocate
  on every call from an allocator that never frees. Keep them as a documented
  convenience, or deprecate them in favour of the `_s` entries.
- **Where the GPU path lives.** `linear_fwd_gpu` is AGNOS-only, not reentrant,
  bit-identical only on x86_64, and ships in the main bundle. Either move it to
  a `[lib.gpu]` profile (`dist/rupantara-gpu.cyr`, rosnet's split) or mark it
  outside the frozen surface.
- **The reproducibility promise.** Decide what 1.x guarantees:
  - across cyrius pins (the 6.6.13 `tanh` change moved GELU by up to 2 ulp);
  - across targets (aarch64's fused FMLA);
  - who re-blesses the golden gate, and when.
- **Acceptance:** each decision is recorded (an ADR or `docs/api.md`) and the
  consumers are migrated.

## 0.8.x — freeze candidate

- **`STABILITY.md`**, the 1.x promise. It covers signatures, error codes, the
  packed layout, the config descriptor, and the reproducibility policy from
  0.7.x. Mark `docs/api.md` frozen.
- **Re-audit what 0.6–0.7 added** on the six dimensions: memory safety, integer
  overflow, shape and bounds, resource caps, fail-loud, and the no-write
  guarantee. That covers the variable-length forward, the lean scratch and the
  config growth. Update `SECURITY.md`.
- **A complete CHANGELOG** for 0.x. The version-consistency gate already exists.
- **Release candidates:** cut `1.0.0-rc.N` tags (`release.yml` accepts them),
  and run attn11 and anukūlana against each one.

## 1.0.0 — the cut

- [ ] API frozen and documented (`docs/api.md` + `STABILITY.md`)
- [ ] ≥1 downstream consumer green on a real model (anukūlana, GPT-2)
- [ ] attn11 parity re-proven on the release candidate
- [ ] CHANGELOG complete
- [ ] the maintainer bumps VERSION → 1.0.0 and tags

These criteria were already met in 0.5.2 and are not repeated above: attn11
parity, decode equal to batch, fuzz and bench, and the security audit with
`SECURITY.md`.

---

## Backlog (post-1.0, demand-gated)

Nothing here is scheduled. Each item lands when a consumer needs it, additively
on the 1.0 surface. Each gets its own ADR if it touches the bit contract.

### Architecture breadth

These build on the extensible config from 0.7.x. Each is ported from attn11
where attn11 has it, and parity-tested, when an import needs it (for example a
Llama-class model).

- **RoPE positions.** `pos_kind = 1` is already reserved in `ru_attn_fwd`, which
  rejects it today.
- **RMSNorm.**
- **The SwiGLU MLP**, with its gated parameter layout.
- **An untied LM head.**

### GPU

- **`linear_fwd_gpu` at K > 8 or with a bias.** Those shapes need cross-tile
  accumulation in `linear_fwd`'s order to stay bit-exact (ADR 0002). Today they
  hand the whole problem to the CPU.
- **Fallback tiles that are exact on aarch64.** The CPU fallback would use
  exact-shape tiles, which first needs the crown's padded gather/scatter check
  to move off the fallback (see `src/gpu.cyr`).

### Performance under the bit contract

- **The scores and LM-head dots.** Both are 4-lane `f64v_fmadd` chains. No
  ordering-preserving way to pack them was found in 0.5.2.
- **AVX2.** It needs a runtime CPU check and a new stdlib leaf in the bundle.
  Rounding would be unchanged.
- **A faster GELU `tanh`.** It would change bits, so it needs an explicit
  parity-policy decision with attn11 first.

### Serving

- **Batched multi-sequence decode**, only if a consumer serves several streams.

### Ecosystem

- **Pin CI's cyrius installer to the release** instead of `main`. This is a
  cross-repo decision: rosnet and the sibling repos do the same today.

## Out of scope (ADR 0001) — keep the boundary

- **Training and backprop:** attn11.
- **Tensors:** rosnet.
- **Pretrained import and LoRA:** anukūlana.
- **The weight format:** tula.
- **The sampler zoo** (temperature, top-k, top-p, beam): the Autoregressive
  Type-2 reference. rupantara provides greedy `ru_argmax`.

---

## Gates / relations

- **Deps:** `rosnet` (`linear_fwd`, `t_alloc`, `t_zero`) and the stdlib's `math`
  and `ganita`. CPU f64, plus the AGNOS GPU tile path.
- **Cross-repo:** changes to attn11 or anukūlana each need the maintainer's go.
- **Consumers:**
  - anukūlana (Type-3 import);
  - attn11 (its leaf ops, and the parity test);
  - the murti load-seam (planned).

  Keep the forward API stable for them.
- **Ecosystem:** the AGNOS planning docs `type3-weight-import.md` (gap #1, where
  rupantara is the transformer-forward prerequisite) and
  `software-port-path.md`.
