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
  2.2–2.7× faster), then GELU and attention.
