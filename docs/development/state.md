# rupantara — Current State

> **Last refresh**: 2026-11-09 (0.5.2) | **Cadence**: every release.
> `CLAUDE.md` is preferences/process; this file is volatile state.

## Version

**0.5.2 (2026-11-09): the P(-1) hardening, optimization and repair sweep**
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

**Suite: 164 assertions in 8 files, green with and without `--bounds`.**

| Suite | Assertions |
|---|---:|
| `smoke` | 1 |
| `ops` | 9 |
| `attn` | 12 |
| `forward` | 21 |
| `decode` | 5 |
| `gpu` | 11 |
| `guard` | 78 |
| `reference` | 27 |

Fuzz runs 20,000 rounds × 5 properties with 0 failures, and there are 12
benchmarks. fmt, lint (strict, deferrals included), vet, deny and doc are
clean, and execution coverage is 22/22 public functions.

**Parity with attn11 is proven, and unchanged since 2026-07-02.**

- **Leaf ops.** attn11 delegates `ru_ln_fwd`, `ru_gelu_fwd` and
  `ru_attn_core_fwd`, and its 1,049 gradient checks run green in one binary.
- **Composition.** `test_rupantara_parity` in attn11's suite feeds attn11's
  `g_params` to `ru_model_fwd` and matches `model_forward` bit for bit
  (`diffs==0`) across MHA ±bias, GQA, MQA and 1–3 blocks.
- **0.5.2.** It changes no output, so these proofs carry over. attn11's GPU,
  bidirectional, row, MLA, SSM, MoE and MTP paths are not de-duplicated:
  rupantara has no peer for them, by design.

## Toolchain

Cyrius pin **6.7.18** (`cyrius.cyml`). Dependencies:

- `math` (`f64_exp`, `F64_PI`) and `ganita` (`f64_tanh`), both stdlib;
- **`rosnet` 1.2.2** (`linear_fwd` and the tensor helpers) via
  `[deps.rosnet]`, git + tag. rosnet pulls in `tyche` 1.2.2 transitively.

**Bits across pins** (measured 2026-11-09):

- **6.6.20 → 6.7.18:** identical.
- **6.6.2 → 6.6.20:** GELU moves. cyrius 6.6.13 replaced `f64_tanh` with
  fdlibm's `expm1` form, so GELU differs by up to 2 ulp and model outputs by up
  to 2.7e-13 relative. `f64_exp`, LayerNorm, softmax and `linear_fwd` are
  identical from 6.6.2 through 6.7.18.
- **attn11** still pins 6.6.2 and rosnet 1.1.2 and will be moved up
  separately. Its parity test compiles both sides with one toolchain, so it
  holds at any pin. attn11's own GELU outputs will shift by the amounts above
  when it moves.

## Build artifacts

- `programs/smoke.cyr` → `build/rupantara_smoke` (link-check; 33,488 B with DCE).
- `dist/rupantara.cyr`: the distlib bundle for consumers (`[deps.rupantara]`).
  The sidecar is `math` and `ganita`. It compiles under cyrius 6.6.2.
- `tests/rupantara.fcyr` (fuzz) and `tests/rupantara.bcyr` (bench).
  `scripts/bench-history.sh` records into `bench-history.csv`.
- CI and release workflows in `.github/workflows/`, SHA-pinned and read-only by
  default. `cyrius.lock` is gitignored.

## Downstream follow-ups (cross-repo, the maintainer's call)

- **anukūlana** decodes with `ru_model_fwd_row`, which allocates per token and
  never frees (`src/main.cyr`). Move it to `ru_model_fwd_row_s` with one
  scratch block (`ru_decode_scratch_count`).
- **attn11 and anukūlana** ignore return values. They pass valid configs, so
  nothing changes for them; a check would surface a future misuse.

## Next

- **M1** (the forward, the attn11 re-fold, parity) and **M2** (KV-cache decode)
  are done.
- **M4** is mostly done in 0.5.2: config validation, fuzz and benchmarks. Still
  open: tokens/s numbers at GPT-2 scale.
- **M5** (the security audit) is covered for memory safety, integer overflow,
  shape bounds and fail-loud by the 0.5.2 audit, and `SECURITY.md` exists.
- Next up: **M3**, architecture breadth (RoPE, RMSNorm, SwiGLU), demand-gated,
  or support for **anukūlana**'s importer. See `roadmap.md`.
