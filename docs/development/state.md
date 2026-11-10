# rupantara — Current State

> **Last refresh**: 2026-11-09 (0.6.1) | **Cadence**: every release.
> `CLAUDE.md` is preferences/process; this file is volatile state.

## Version

**0.6.1 (2026-11-09):** `ru_gelu_fwd_t`, GELU that also hands out its tanh so a
consumer's backward reuses it. This was attn11's filed proposal
([`issues/2026-11-09-gelu-fwd-tanh-output.md`](issues/2026-11-09-gelu-fwd-tanh-output.md)).
It is additive, and every existing output keeps its bits.

**0.6.0 (2026-11-09):** the rupantara side of the 0.6.x roadmap:

- a committed golden gate (`tests/tcyr/golden.tcyr` against
  `tests/golden/{x86_64,aarch64}.txt`) and an aarch64 CI job;
- the variable-length forward `ru_model_fwd_n` / `ru_model_fwd_n_s`;
- a model forward whose scratch holds no `nh·T·T` probabilities (0.84 GiB at
  GPT-2 widths and T = 8192, from 6.84);
- GPT-2-small throughput benchmarks.

Every output is bit-identical to 0.5.3 on both targets. It is built on
**0.5.3** (rosnet 1.2.2 → 1.4.1, bits unchanged) and **0.5.2**, the P(-1)
hardening, optimization and repair sweep
([`docs/audit/2026-11-09-audit.md`](../audit/2026-11-09-audit.md)).

**The forward** is the GPT-2-shaped forward ported verbatim from attn11:

- LayerNorm (`ru_ln_fwd`, ε=1e-5) and GELU (`ru_gelu_fwd`, tanh form);
- causal MHA/GQA/MQA attention (`ru_attn_core_fwd`, `ru_attn_fwd`);
- `ru_mlp_fwd`, `ru_embed_fwd` (token + learned-absolute position), and the
  weight-tied head (`ru_head_fwd`, `ru_softmax_fwd`);
- the pre-norm `ru_block_fwd` and the block-stack `ru_model_fwd`, over a packed
  layout byte-identical to attn11's;
- KV-cache decode (`ru_model_fwd_row`, `ru_argmax`).

**New in 0.5.2:**

- the model entries validate their input (`ru_cfg_check`, the `RU_ERR_*`
  codes) and write nothing on a rejection;
- allocation-free caller-scratch entries `ru_model_fwd_s` and
  `ru_model_fwd_row_s`;
- attention 1.5× faster, with outputs bit-identical to 0.5.1.

The surface is documented in [`docs/api.md`](../api.md).

**Suite: 195 assertions in 9 files, green on x86_64 and on aarch64 (qemu).**

| Suite | Assertions |
|---|---:|
| `smoke` | 1 |
| `ops` | 12 |
| `attn` | 12 |
| `forward` | 30 |
| `decode` | 5 |
| `gpu` | 11 |
| `guard` | 94 |
| `reference` | 27 |
| `golden` | 3 (11,550 raw bit patterns per target) |

Fuzz runs 20,000 rounds × 7 properties with 0 failures, and there are 15
benchmarks. fmt, lint, vet and doc are clean.

- Execution coverage was 22/22 public functions at 0.5.2.
- The two new entries are exercised by `forward`, `guard`, `golden`'s callers
  and fuzz property P6.
- 11 mutants of the new code were all killed.

**Parity with attn11 is proven, and unchanged since 2026-07-02.**

- **Leaf ops.** attn11 delegates `ru_ln_fwd`, `ru_gelu_fwd` and
  `ru_attn_core_fwd`, and its 1,049 gradient checks run green in one binary.
- **Composition.** `test_rupantara_parity` in attn11's suite feeds attn11's
  `g_params` to `ru_model_fwd` and matches `model_forward` bit for bit
  (`diffs==0`) across MHA ±bias, GQA, MQA and 1–3 blocks.
- **0.5.2 through 0.6.0.** They change no output, so these proofs carry
  over. Re-checked 2026-11-09: attn11's suite (1,089 assertions,
  `test_rupantara_parity` included) and anukūlana's 9 suites (219) pass when
  built against the 0.6.0 bundle. attn11's GPU,
  bidirectional, row, MLA, SSM, MoE and MTP paths are not de-duplicated:
  rupantara has no peer for them, by design.

## Toolchain

Cyrius pin **6.7.18** (`cyrius.cyml`). Dependencies:

- `math` (`f64_exp`, `F64_PI`) and `ganita` (`f64_tanh`), both stdlib;
- **`rosnet` 1.4.1** (`linear_fwd` and the tensor helpers) via
  `[deps.rosnet]`, git + tag. rosnet pulls in `tyche` 1.2.2 transitively, and
  the stdlib `simd` leaf through its own sidecar.
- On an AVX2 CPU, `linear_fwd` takes rosnet 1.4.1's 256-bit path (`vmulpd` +
  `vaddpd`, unfused). rupantara's golden dump is byte-identical on it.

**Bits across pins** (measured 2026-11-09):

- **6.6.20 → 6.7.18:** identical.
- **6.6.2 → 6.6.20:** GELU moves. cyrius 6.6.13 replaced `f64_tanh` with
  fdlibm's `expm1` form, so GELU differs by up to 2 ulp and model outputs by up
  to 2.7e-13 relative. `f64_exp`, LayerNorm, softmax and `linear_fwd` are
  identical from 6.6.2 through 6.7.18.
- **attn11 and anukūlana** moved to 6.7.18, rosnet 1.4.1 and rupantara
  0.5.3 (attn11 1.15.1, anukūlana 1.2.2). attn11's parity test compiles both
  sides with one toolchain, so it holds at any pin.
- **The golden gate pins the toolchain's numerics too.** It fails at cyrius
  6.6.12 and 6.6.2, on every GELU-dependent case, and passes at 6.6.13. A pin
  bump that moves bits must re-bless it.

## Build artifacts

- `programs/smoke.cyr` → `build/rupantara_smoke` (link-check; 38,784 B with DCE: rosnet 1.4.1's `simd` leaf and AVX2 path).
- `dist/rupantara.cyr`: the distlib bundle for consumers (`[deps.rupantara]`).
  The sidecar is `math` and `ganita`. It compiles under cyrius 6.6.2.
- `tests/rupantara.fcyr` (fuzz) and `tests/rupantara.bcyr` (bench).
  `scripts/bench-history.sh` records into `bench-history.csv`.
- `tests/golden.cyr` (the gate's generator) and `tests/golden/*.txt` (the
  committed dumps, about 170 KB each). `make golden-bless` rewrites them, and
  `make test-aarch64` runs every suite under qemu.
- CI and release workflows in `.github/workflows/`, SHA-pinned and read-only by
  default. `cyrius.lock` is gitignored.

## Downstream follow-ups (cross-repo, the maintainer's call)

Done in the consumers since 0.5.2:

- anukūlana 1.2.2 decodes on `ru_model_fwd_row_s` with one scratch block, and
  checks the `RU_ERR_*` returns;
- both consumers moved to rupantara 0.5.3.

Open:

- **anukūlana: drop the run-length repack.** `anuk_gpt2_import` overrides `T`
  to the run length and packs a fresh ~1 GB parameter buffer for each length.
  With 0.6.0's `ru_model_fwd_n` it can import once at the file's `T` (1024)
  and run any `n`.
- **anukūlana's real-checkpoint gates** (`make fidelity`, `gpt2-oracle`, …)
  have not run since its 1.1.3. They need the maintainer's machine.

## Next

[`roadmap.md`](roadmap.md) plans 0.6 → 1.0. The rupantara side of **0.6.x**
shipped in 0.6.0. What remains of 0.6.x is cross-repo: both consumers moving to
0.6.0, anukūlana's repack removal, and its real-checkpoint fidelity gate run
through rupantara.

0.7.x fixes the API shape before the freeze, 0.8.x is the freeze candidate,
and architecture breadth (RoPE, RMSNorm, SwiGLU) is post-1.0 backlog,
demand-gated.
