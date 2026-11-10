# rupantara benchmarks

`tests/rupantara.bcyr` (run with `make bench`), recorded by
`scripts/bench-history.sh` into `bench-history.csv`. Each number is the median
of 3 runs, and every row is labelled with the date and `VERSION`. Compare rows
only when they were recorded in the same session on the same host. A
laptop-class host moves the same binary by a few percent between runs.

## 0.5.1 → 0.5.2 (one session, 2026-11-09, x86_64)

| Benchmark | 0.5.1 | 0.5.2 | Speed-up |
|---|---:|---:|---:|
| `ru_attn_core_fwd` T=128 C=768 nh=12 | 18.13 ms | 12.24 ms | 1.48× |
| `ru_attn_core_fwd` T=128 C=768 nh=12 nkv=4 | 19.03 ms | 12.54 ms | 1.52× |
| `ru_attn_core_fwd_row` pos=511 C=768 nh=12 | 1.166 ms | 0.780 ms | 1.50× |
| `ru_head_fwd_row` C=768 V=12564 | 8.95 ms | 8.91 ms | — |
| `ru_ln_fwd` 64×768 | 586 µs | 559 µs | — |
| `ru_gelu_fwd` n=64×3072 | 19.30 ms | 18.88 ms | — |
| `ru_softmax_fwd` 1×50257 | 1.231 ms | 1.200 ms | — |
| `ru_embed_fwd` 64×768 | 201.5 µs | 20.4 µs | 9.9× |
| `ru_model_fwd` V=1024 C=256 T=64 nh=4 NL=2 F=1024 | 67.4 ms | 63.9 ms | 1.06× |
| `ru_model_fwd_row` pos=63, same model | 1.133 ms | 1.037 ms | 1.09× |
| `ru_model_fwd_s`, same model, caller scratch | — | 61.5 ms | new |
| `ru_model_fwd_row_s` pos=63, caller scratch | — | 1.081 ms | new |

A dash in the speed-up column means the op was not changed, and its 1–5% move
is run-to-run spread. Every 0.5.2 output is bit-identical to 0.5.1 (x86_64 and
aarch64; see the audit).

- **Attention.** The PV accumulation is one `f64v_axpy` per key over the
  4-aligned head dim. Before, it filled a 4-lane broadcast buffer and made one
  `f64v_fmadd` call per 4 elements. The scores dot is unchanged and is now most
  of the attention cost.
- **Embedding.** One `f64v_add` per row. Before, it was a scalar loop with
  64-bit index arithmetic per element. The residual adds in the block and the
  decode step use the same helper.
- **`_s` entries.** They are no faster: a bump allocation is a pointer bump
  plus a zero-fill. Their value is memory. Over 1,000 decode steps at GPT-2
  widths the allocating step grows the heap by 106.5 MB, and the `_s` step by
  0 bytes.

## Where the time goes

- **`ru_gelu_fwd`**: about 96 ns per element, nearly all of it `f64_tanh`
  (stdlib ganita). A faster tanh would change bits, and GELU's bits are the
  attn11 parity contract, so it stays.
- **`ru_head_fwd_row`**: V dot products, each a 4-lane `f64v_fmadd` chain. That
  reduction order is the contract. Packing several vocabulary rows into one
  call would need a gather copy per call, and no ordering-preserving packing
  was found.
- **The whole forward**: dominated by rosnet `linear_fwd` (rosnet 1.2.2 made it
  2.2–2.7× faster, and 1.4.1 adds an AVX2 path), then GELU and attention.

## 0.5.2 → 0.5.3 (rosnet 1.2.2 → 1.4.1, 2026-11-09, Ryzen 7 5800H)

rosnet 1.4.1 runs `linear_fwd` 4 lanes wide on an AVX2 CPU. Measured in one
process, alternating the AVX2 path with rosnet's forced-SSE2 switch
(`_rosnet_force_sse2`, test-only), median of 5:

| Shape | AVX2 / SSE2 |
|---|---:|
| `linear_fwd` at the bench model's projections (M = 64 or 1; C = 256, F = 1024) | 0.858 – 0.997 |
| `linear_fwd` at GPT-2 widths (M = 64 or 1; C = 768, F = 3072) | 0.893 – 0.960 |
| `ru_model_fwd_s`, bench model (V=1024 C=256 T=64 NL=2) | 0.955 |
| `ru_model_fwd_row_s`, bench model | 0.976 |
| `ru_model_fwd_s`, GPT-2 widths (V=1024 C=768 T=64 NL=2) | 1.000 |
| `ru_model_fwd_row_s`, GPT-2 widths | 1.008 |

- **Every matmul is faster**, by 0.3–14%.
- **End to end, the bench model gains 2.4–4.5%, and GPT-2 widths are even**
  (within ±1%). rosnet 1.4.1 reports the same effect: on this power-limited
  CPU, 256-bit work slows the code that runs after it, here GELU and
  attention.
- **Use the in-process A/B, not the history rows.** The `bench-history.csv`
  rows for this bump, two `scripts/bench-history.sh` sessions taken minutes
  apart, read the model rows 4–7% *slower*. The A/B above shows that is drift
  between sessions, not the bump. Every output is bit-identical on both paths.

## 0.6.0: GPT-2 throughput and the lean scratch (2026-11-09, Ryzen 7 5800H)

**GPT-2 small at its real dims** (V = 50257, C = 768, T = 1024, 12 heads, 12
layers, F = 3072, attention biases: 124M parameters, 1 GB of f64), synthetic
weights. These are the `bench-history.csv` rows, medians of 3 runs. An earlier
set in the same session measured 9.04 s and 98.3 ms.

| Benchmark | Time | Throughput |
|---|---:|---:|
| Prefill, 128 tokens through the T = 1024 layout (`ru_model_fwd_n_s`, logits only) | 9.28 s | 13.8 tokens/s |
| One decode step at pos = 1023 (`ru_model_fwd_row_s`, full-context attention) | 99.1 ms | 10.1 tokens/s |

- **A decode step streams the whole model.** Each token reads all 1 GB of f64
  weights once, a third of it the weight-tied head (38.6M of the 124M).
- **Prefill is compute-bound**, at about 72 ms per token.
- **The decode step's KV cache is filled synthetically.** Decoding 1,023 steps
  to fill it would take minutes, and cache values do not change the time.

**The model forward's scratch** at GPT-2-small widths, now that it holds one
T-element probability row instead of `nh·T·T`:

| T | Through 0.5.3 | Now | Smaller by |
|---:|---:|---:|---:|
| 1024 | 0.20 GiB | 0.106 GiB | 1.9× |
| 4096 | 1.92 GiB | 0.42 GiB | 4.6× |
| 8192 | 6.84 GiB, past the 2 GiB allocation cap | 0.84 GiB | 8.1× |

**The lean path is speed-neutral.** Measured in one process against the 0.5.3
path rebuilt from public pieces (`ru_block_fwd` over a full arena, which
zero-fills and writes `Pc`), alternating which runs first, with both scratch
blocks at the same 64-byte alignment:

| Model | Lean / old |
|---|---:|
| Bench model (V=1024 C=256 T=64 NL=2) | 1.009 |
| GPT-2 widths (V=1024 C=768 T=64 NL=2) | 0.996 |

Every output is bit-identical.

- **Scratch alignment matters more than the change itself.** Moving the lean
  block's offset mod 64 through 0, 8, 16 and 24 bytes moved lean/old between
  0.951 and 1.064. A bit-safe lead, in the roadmap backlog.
- **The 0.6.0 history rows came from a slow session.** Every benchmark, changed
  or not, reads 13–22% slower than the 0.5.3 rows. Three examples:
  - `ru_ln_fwd`: 542 → 619 µs;
  - GELU: 17.8 → 20.7 ms;
  - the decode step, whose code did not change: 1.03 → 1.22 ms.

  The model forward's +17–18% sits inside that band. Compare 0.6.0 with 0.5.3
  through the A/B above, not the CSV.

