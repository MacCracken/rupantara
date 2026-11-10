# rupantara API (0.6.0)

The transformer **forward**: f64, CPU, pure Cyrius. Consumers import
`dist/rupantara.cyr` through `[deps.rupantara]` and must also declare
`[deps.rosnet]` (rupantara calls rosnet's `linear_fwd`, `t_alloc` and `t_zero`).
Pre-1.0: the surface may still move. Nothing here is frozen until v1.0.

## Conventions

- **Buffers** are untyped addresses of f64 (8 bytes each) or i64 (token ids,
  the config). **Sizes** are element counts, not bytes.
- **Matrices** are row-major. `linear_fwd(x, W, b, y, M, K, N)` computes
  `y(M,N) = x(M,K) @ W(K,N) + b`.
- **Return values** are `0` on success. The validated entries (below) return a
  negative code instead, before writing anything.

## Errors and validation

| Code | Function | Meaning |
|---|---|---|
| `0` | `RU_OK()` | success |
| `-1` | `RU_ERR_CFG()` | a malformed config (see `ru_cfg_check`) or a shape the kernels cannot index |
| `-2` | `RU_ERR_TOKEN()` | a token id outside `[0, V)` |
| `-3` | `RU_ERR_POS()` | a decode position outside `[0, T)`, or a run length `n > T` |
| `-4` | `RU_ERR_ARG()` | an unsupported option, a run length `n < 1`, or a 0 pointer |

**Validated entries** check their arguments and return a code with every
output untouched (the KV cache included) on failure. They are `ru_model_fwd`,
`ru_model_fwd_s`, `ru_model_fwd_n`, `ru_model_fwd_n_s`, `ru_model_fwd_row`,
`ru_model_fwd_row_s`, `ru_attn_fwd`, `ru_argmax`, and the sizing functions.

**Leaf kernels** take dims and buffers on trust, as rosnet's kernels do. They
run no checks, so the inner loops pay nothing. They are `ru_ln_fwd`,
`ru_gelu_fwd`, `ru_attn_core_fwd`, `ru_attn_core_fwd_row`, `ru_attn_fwd_row`,
`ru_mlp_fwd`, `ru_embed_fwd`, `ru_head_fwd_row`, `ru_head_fwd`,
`ru_softmax_fwd`, `ru_block_fwd` and `linear_fwd_gpu`. A caller that builds its
own forward from these should run `ru_cfg_check` once first.

## The config

`ru_cfg_init(cfg, V, C, T, nh, nkv, NL, F, attn_bias)` fills a caller-owned
`i64[8]`: vocabulary, width, context (window length), query heads, K/V heads,
layers, MLP hidden width, and 0/1 for attention biases. It stores the values as
given.

`ru_cfg_check(cfg)` returns 0 or `RU_ERR_CFG`. It requires:

- `V, C, T, nh, nkv, F >= 1` and `NL >= 0`
- `C % nh == 0` (whole heads; `hd = C / nh`)
- `nh % nkv == 0` (whole GQA groups; `nkv == nh` is MHA, `nkv == 1` is MQA)
- `attn_bias` is 0 or 1
- every buffer the config sizes stays within 2^56 elements, so no count or
  byte offset can wrap an i64

Query head `h` reads K/V head `h / (nh / nkv)`.

## Sizing

Each function returns an element count, or `RU_ERR_CFG` (-1) for a config
`ru_cfg_check` rejects. Passing -1 to `t_alloc` fails loudly ("invalid size").

| Function | Buffer |
|---|---|
| `ru_params_count(cfg)` | the packed parameters |
| `ru_model_scratch_count(cfg)` | scratch for `ru_model_fwd_s`, and for `ru_model_fwd_n_s` at any `n <= T` |
| `ru_decode_kv_count(cfg)` | the KV cache: `2 * NL * T * Ckv`, with `Ckv = nkv * hd` |
| `ru_decode_scratch_count(cfg)` | scratch for `ru_model_fwd_row_s` |
| `ru_attn_arena_size(T, C, nh, nkv)` | the arena for `ru_attn_fwd` |

The caller also owns the outputs: `logits` and `probs` hold `(T, V)` for the
batch forward, `(n, V)` for a run of `n` tokens, and `(V)` for one decode step.

**The model scratch holds no attention probabilities** (0.6.0). The model
forward runs each (head, query row) pair through one reused T-element
probability row instead of an `nh·T·T` buffer, which it never read again. At
GPT-2-small widths (C = 768, 12 heads, F = 3072) the scratch is 0.106 GiB at
T = 1024, 0.42 GiB at T = 4096 and 0.84 GiB at T = 8192, against 0.20, 1.92 and
6.84 GiB through 0.5.3. The outputs are bit-identical. `ru_attn_fwd`,
`ru_attn_core_fwd` and `ru_block_fwd` still fill their full `Pc`, because
attn11's backward reads it.

## The forward

| Function | What it does |
|---|---|
| `ru_model_fwd(cfg, params, tokens, logits, probs)` | `tokens(T)` → `logits(T,V)`, and `probs(T,V)` unless `probs` is 0. **Allocates** its scratch on every call (see below). |
| `ru_model_fwd_s(cfg, params, tokens, logits, probs, scratch)` | The same on caller scratch, with no allocation. Bit-identical. |
| `ru_model_fwd_n(cfg, params, tokens, n, logits, probs)` | The first `n` positions only, `1 <= n <= T`: `tokens(n)` → `logits(n,V)`, and `probs(n,V)` unless `probs` is 0. Rows bit-identical to `ru_model_fwd`'s. **Allocates** scratch for `n` rows. (0.6.0) |
| `ru_model_fwd_n_s(cfg, params, tokens, n, logits, probs, scratch)` | The same on caller scratch (`ru_model_scratch_count`, enough for any `n`). Bit-identical. (0.6.0) |
| `ru_model_fwd_row(cfg, params, tok, pos, kv, logits)` | One decode step: `tok` at position `pos` reads the KV cache for `0..pos-1`, appends its own K/V, and writes `logits(V)`. **Allocates**. |
| `ru_model_fwd_row_s(cfg, params, tok, pos, kv, logits, scratch)` | The same on caller scratch. Bit-identical. |
| `ru_argmax(logits, V)` | Greedy next token. A tie goes to the lowest index. A NaN past index 0 never wins, and a NaN at index 0 stays the answer. Returns `RU_ERR_ARG` for `V < 1`. |

**Use the `_s` entries in loops.** rosnet's allocator is a bump allocator and
never frees. `ru_model_fwd_row` leaves its scratch behind on every token:
measured, 106.5 MB of heap over 1,000 decode steps at GPT-2 widths (C = 768,
F = 3072), against 0 bytes for `ru_model_fwd_row_s`. Allocate one scratch block
and one KV cache, and reuse them for the whole sequence. The scratch needs no
clearing: every slot is written before it is read.

**Decode.** Feed positions `0, 1, …, T-1` in order. Row `pos` of
`ru_model_fwd` and the cached step at `pos` are bit-identical. A position at or
past `T` returns `RU_ERR_POS`. Through 0.5.1 it wrote past the KV cache.

**Prompts shorter than T.** Call `ru_model_fwd_n(_s)` with the prompt length
`n`. It runs positions `0..n-1` only, writes `n` rows, and reads and checks only
the first `n` ids. Its rows are bit-identical to rows `0..n-1` of
`ru_model_fwd` over the same tokens padded to `T` with any valid ids, because
attention is causal. One parameter buffer, laid out for the context `T` (the
`pos_emb` rows), serves every run length, so there is no need to repack it per
length. Through 0.5.3 the only way was to pad the window and pay for all `T`
rows.

## Packed parameter layout

The layout is byte-identical to attn11's MHA/dense layout. Offsets are in
elements:

```
[ tok_emb (V*C) ][ pos_emb (T*C) ][ block_0 ] … [ block_{NL-1} ][ lnf_g (C) ][ lnf_b (C) ]
block: ln1_g(C) ln1_b(C) Wq(C*C) Wk(C*Ckv) Wv(C*Ckv) Wo(C*C)
       ln2_g(C) ln2_b(C) Wfc(C*F) bfc(F) Wproj(F*C) bproj(C)
       [ bq(C) bk(Ckv) bv(Ckv) bo(C) ]        <- only when attn_bias == 1
```

The output head is weight-tied: `logits = LN_f(x) @ tok_emb^T`.

**Layout helpers.** An importer that fills the buffer locates each field with
the helpers below. anukūlana already uses them. They are underscore-named, but
they are part of the layout contract and will not be renamed before v1.0.

- **Config fields:** `_cfg_V`, `_cfg_C`, `_cfg_T`, `_cfg_nh`, `_cfg_nkv`,
  `_cfg_NL`, `_cfg_F`, `_cfg_bias`, `_cfg_Ckv`
- **Block base:** `_ru_blk_base(cfg, L)`, the element offset of block `L`, and
  `_ru_blk(cfg)`, the element count of one block
- **Offsets within a block:** `_ru_o_ln1g`, `_ru_o_ln1b`, `_ru_o_Wq`,
  `_ru_o_Wk`, `_ru_o_Wv`, `_ru_o_Wo`, `_ru_o_ln2g`, `_ru_o_ln2b`, `_ru_o_Wfc`,
  `_ru_o_bfc`, `_ru_o_Wproj`, `_ru_o_bproj`, `_ru_o_bq`, `_ru_o_bk`,
  `_ru_o_bv`, `_ru_o_bo`
- **Pointers:** `_p_tokemb`, `_p_posemb`, `_p_lnfg`, `_p_lnfb`, and `_bq` /
  `_bk` / `_bv` / `_bo` (0 when `attn_bias == 0`)

## Building blocks (leaf kernels)

| Function | Shape / notes |
|---|---|
| `ru_ln_fwd(x, gamma, beta, y, mean, rstd, M, C)` | LayerNorm per row, eps 1e-5. `y` may alias `x`. |
| `ru_gelu_fwd(x, y, n)` | GELU, tanh form (GPT-2). `y` may alias `x`. |
| `ru_attn_core_fwd(Qc, Kc, Vc, Pc, concat, T, C, nh, nkv)` | Causal scaled-dot-product attention after the projections. Fills `Pc(nh,T,T)` and `concat(T,C)`. |
| `ru_attn_core_fwd_row(q_row, Kc, Vc, prow, concat_row, pos, C, nh, nkv)` | One query against cache rows `0..pos`. |
| `ru_attn_fwd(Wq, bq, Wk, bk, Wv, bv, Wo, bo, x, out, ar, T, C, nh, nkv, pos_kind)` | Projections, the core, then Wo. Biases may be 0. `pos_kind` must be 0 (learned-absolute); 1 (RoPE) returns `RU_ERR_ARG`. Validated. |
| `ru_attn_fwd_row(...)` | The cached single-row version: appends K/V at `pos`. |
| `ru_mlp_fwd(x, Wfc, bfc, Wproj, bproj, fc, gg, y, M, C, F)` | `gelu(x @ Wfc + bfc) @ Wproj + bproj` |
| `ru_embed_fwd(tokens, tok_emb, pos_emb, x0, n, C)` | `x0[t] = tok_emb[tokens[t]] + pos_emb[t]` |
| `ru_head_fwd_row(f_row, tok_emb, logits_row, C, V)` / `ru_head_fwd(f, tok_emb, logits, n, C, V)` | The weight-tied head. |
| `ru_softmax_fwd(logits, probs, M, V)` | Row softmax, max-subtracted. `probs` may alias `logits`. |
| `ru_block_fwd(cfg, pblk, xin, xout, …, arena)` | One pre-norm block over `T` rows. Safe in place. `arena` holds `ru_attn_arena_size(T, C, nh, nkv)`, and its `Pc` holds the block's probabilities afterwards. |
| `linear_fwd_gpu(x, W, b, y, M, K, N)` | `linear_fwd` on the AGNOS GPU (#83), in 8×8×8 tiles when K ≤ 8 and b = 0, otherwise on the CPU. Bit-identical on x86_64. On aarch64, a ragged last tile's final `N % 4` columns can differ by the fused-FMLA rounding. Returns the tile count. **Not reentrant.** |
| `rupantara_version()` | `major*10000 + minor*100 + patch` (0.5.2 → 502) |

## Numerical reproducibility

- **Same toolchain, same target:** bit-identical. That covers batch against
  cached decode, the `_s` entries against the allocating ones, and 0.5.2
  against 0.5.1. attn11's parity test (`ru_model_fwd` against `model_forward`)
  compiles both sides in one binary with one toolchain.
- **x86_64 against aarch64:** not bit-identical. cyrius fuses the packed
  multiply-add on aarch64 (FMLA), and on x86_64 it is `mulpd` then `addpd`.
  The attention dot products and rosnet's `linear_fwd` inherit that. GELU
  matches across the two targets over its whole range. Softmax's `f64_exp`
  differs by at most 1 ulp, in 75 of 801 values of the golden sweep.
- **The golden gate** (0.6.0). `tests/tcyr/golden.tcyr` holds every
  public forward op to the committed `tests/golden/x86_64.txt` or
  `tests/golden/aarch64.txt`: 9,969 raw bit patterns per target. CI runs it on
  both targets, aarch64 under qemu. A change that moves any bit fails there,
  with the case, the value and both patterns named, until it is reviewed,
  re-blessed with `make golden-bless`, and noted in the CHANGELOG.
- **Across cyrius pins:** cyrius **6.6.13** replaced `f64_tanh` (ganita
  1.2.11) with fdlibm's `expm1` form, an accuracy fix. GELU therefore differs
  by up to 2 ulp between builds on cyrius ≤ 6.6.12 and ≥ 6.6.13, and model
  outputs by up to about 3e-13 relative. Nothing else in the forward moved:
  `f64_exp`, LayerNorm, softmax and `linear_fwd` are bit-identical from 6.6.2
  through 6.7.18. Measured 2026-11-09: 0.5.1 built at 6.6.2 against 6.7.18.
- **Precision:** the whole forward is f64. There is no f32 or mixed path.
