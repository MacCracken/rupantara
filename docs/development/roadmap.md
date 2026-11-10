# rupantara — Roadmap (0.6 → 1.0, then the backlog)

> **For a secondary agent picking this up cold.** Self-contained plan. Read this
> + `CLAUDE.md` (conventions) + `docs/adr/0001-rupantara-scope.md` (scope) +
> `docs/api.md` (the surface) + `docs/development/state.md` (where it stands).
> rupantara is the transformer **forward** library extracted from attn11 — not
> training, not tensors, not the importer. Pre-1.0: surface may move, no freeze
> until v1.0.

## Where it stands (0.6.0)

The GPT-2-shaped forward and KV-cache decode are done. That covers MHA/GQA/MQA
attention, the GELU MLP, LayerNorm, learned positions and the weight-tied head.
Both are bit-identical to attn11's forward: attn11's `test_rupantara_parity`
checks it, and attn11's leaf ops run through rupantara.

The model entries validate their input, and allocation-free caller-scratch
entries exist. A fuzz harness, benchmarks and a security audit are in place.

0.6.0 shipped the rupantara side of 0.6.x:

- the committed golden gate, with an aarch64 CI job;
- the variable-length forward;
- the lean model scratch;
- GPT-2-small throughput.

Both consumers are on cyrius 6.7.18, rosnet 1.4.1 and rupantara 0.5.3; moving
them to 0.6.0 is the remaining 0.6.x work.
Completed work is recorded in `CHANGELOG.md` and `docs/development/state.md`,
not here.

---

## Working agreement

- **Build/test:** `make build` · `make test` · `make test-aarch64` (qemu) ·
  `make fuzz` · `make bench` · `make dist`. The pin is in `cyrius.cyml`.
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
  change with the oracles in `tests/reference.cyr` and the committed golden
  gate (`tests/tcyr/golden.tcyr`; `make test` and `make test-aarch64`).
  Re-bless it (`make golden-bless`) only for a reviewed numerical change that
  the CHANGELOG names.
- **Model entries validate, leaf kernels trust** (ADR 0003). New entries return
  `RU_ERR_*` and write nothing on a rejection.

---

## 0.6.x — consumers on real models

**Goal:** anukūlana runs a real GPT-2 through rupantara end to end, at any run
length, on a current rupantara.

The rupantara side shipped in 0.6.0. What remains is cross-repo:

- **⚠ Cross-repo, anukūlana:**
  - drop the run-length repack. `anuk_gpt2_import` overrides `T` to the run
    length and packs a fresh ~1 GB parameter buffer for each length. It can
    import once at the file's `T` (1024) and call `ru_model_fwd_n(_s)` with
    any `n`;
  - move `[deps.rupantara]` to 0.6.0.
- **⚠ Cross-repo, attn11:** move `[deps.rupantara]` to 0.6.0 and re-run its
  suite. It passed 1,089/1,089, `test_rupantara_parity` included, built against
  the 0.6.0 bundle on 2026-11-09.
- **Acceptance:** anukūlana's GPT-2 fidelity gate (`make fidelity`,
  `gpt2-oracle`, its 1.3.x item) is green through rupantara on the
  maintainer's machine. That is the v1.0 consumer criterion.

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

- **Scratch alignment.** In the 0.6 A/B, moving the model forward's scratch
  block between 0 and 24 bytes off a 64-byte boundary moved its time by up to
  ±5%, more than the lean change itself. The allocating entries could align
  their own block, and the `_s` docs could ask callers to. Alignment changes no
  arithmetic, so no bits move.
- **The scores and LM-head dots.** Both are 4-lane `f64v_fmadd` chains. No
  ordering-preserving way to pack them was found in 0.5.2.
- **AVX2 in rupantara's own kernels.** rosnet 1.4.1's `linear_fwd` already has
  an unfused AVX2 path, through a runtime CPU check and the stdlib `simd` leaf.
  The attention and head dots could follow if their 4-lane order is kept.
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
