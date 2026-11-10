# Proposal: `ru_gelu_fwd_t` — also return GELU's tanh, so a consumer's backward reuses it (measured −17 % per training step in attn11)

**Status:** 🟢 **RESOLVED in rupantara 0.6.1** (2026-11-09). `ru_gelu_fwd_t(x, y, t, n)` landed
as asked. See *Resolution* at the end.
**Placement:** unpinned.
**Discovered:** 2026-11-09, by attn11's 1.15.2 P(-1) audit (performance review), then prototyped
and measured in attn11.
**Severity:** Performance only. There is no defect.
**Affects:** every consumer that runs a GELU backward after `ru_gelu_fwd`. In attn11 that is
the dense MLP, MoE experts and the vision/hearing CNNs. Verified against rupantara 0.5.3 (attn11's
pin); `ru_gelu_fwd` is unchanged at 0.6.0.

## Summary

`ru_gelu_fwd` (`src/act.cyr`) computes, per element:

```
x3    = (x·x)·x
inner = c·(x + a·x3)          # c = _ru_gelu_c(), a = _ru_gelu_a()
t     = tanh(inner)
y     = (½·x)·(1 + t)
```

It then throws `t` away. A GELU backward needs exactly that `t`, so attn11's `gelu_bwd` (which
stays attn11-local, as all backward does) **recomputes the same expression and calls
`f64_tanh` again** for every element. Since ganita 1.2.11, `f64_tanh` is the dominant cost of
that loop: it goes through `_gn_expm1` and measures ~91–95 ns per call in attn11's build. At
attn11's default config, one training step runs 3 layers × T 16 × F 128 = **6,144 such tanh**.
(The tanh cost itself is filed with ganita:
`ganita/docs/development/issues/2026-11-09-tanh-cost-on-gelu-consumers.md`.)

**Caching `t` is bit-exact.** The backward forms `x2 = x·x` and `x3 = x2·x`, which is the same
sequence as the forward's `(x·x)·x`, and the same `inner` with the same constants
(`_gelu_c`/`_gelu_a` equal `_ru_gelu_c`/`_ru_gelu_a`). So the stored `t` is bit-for-bit the
value the backward recomputes today.

## Measured (attn11 prototype, scratch copy — not shipped)

The prototype replaced the dense-MLP training forward with an attn11-local copy of the
`ru_gelu_fwd` expression that also stores `t` into a per-layer T·F cache (48 KB at the default
config). `gelu_bwd` then loads `t` instead of calling `f64_tanh`. Everything after `t` is
unchanged.

- **Bit-identity:** attn11's default 2,000-step training run, `--steps 300 --eval` and
  `--bpe 64 --steps 400 --eval` are **byte-identical** to the unmodified build.
- **Speed** (attn11 bench, DCE builds, run alone, 4 interleaved pairs, Cezanne):

  | row | today | with cached `t` | Δ |
  |---|---:|---:|---:|
  | `model_forward` | 1.142 ms | 1.141 ms | 0 % (storing `t` costs nothing measurable) |
  | `model_backward` | 1.970 ms | 1.461 ms | **−25.8 %** |
  | fwd+bwd step (default) | 3.208 ms | 2.651 ms | **−17.4 %** |

MoE (T·K·F tanh per layer) and the CNN lanes have the same structure and were not part of the
prototype.

## Ask

Add, alongside the unchanged `ru_gelu_fwd`:

```
# As ru_gelu_fwd, and also writes t[i] = tanh(c·(x + a·x³)) — the exact value the
# forward used — so a consumer's GELU backward can reuse it bit-identically.
fn ru_gelu_fwd_t(x, y, t, n): i64
```

Same loop, plus `store64(t + i * 8, t_i)`. The change is additive: no existing signature or
result changes, `ru_gelu_fwd` keeps its bits, and the attn11 ↔ rupantara parity contract is
untouched. Both functions should be pinned by the existing raw-bits golden dump, plus a `t` row.
attn11 would then call `ru_gelu_fwd_t` in its training forward (keeping `ru_gelu_fwd` for
eval and decode, which run no backward). It would also gain a backward that takes the cached
`t`, with a "cache valid" guard so its `--gpu-tc` path, where the GPU runs the forward, falls
back to recomputing.

Doing this attn11-locally instead would duplicate the forward and break the single source of
truth the re-fold set up, which is why it is filed here rather than shipped in attn11 1.15.2.

## Resolution (rupantara 0.6.1)

- **`ru_gelu_fwd_t(x, y, t, n)`** is in `src/act.cyr`, with the signature asked for.
  - **One loop.** It and `ru_gelu_fwd` both run `_ru_gelu_rows`, the single GELU loop. So the
    `t` it hands out is the `t` the forward used, and `ru_gelu_fwd`'s bits cannot drift.
  - **Aliasing:** `y` may alias `x`; `t` must not overlap `y`.
- **Pinned four ways:**
  - **`tests/tcyr/ops.tcyr`:** `y` equals `ru_gelu_fwd`'s, and `t` equals the tanh recomputed in
    `gelu_bwd`'s form, bit for bit. That covers a [-10, 10] sweep, ±0, a subnormal and ±1000,
    in place too.
  - **The golden gate:** a new `@gelu-t sweep` case on both targets. The re-bless only appended
    lines; the first 9,969 are unchanged.
  - **Fuzz:** property P7, on random values.
  - **Bench:** a `ru_gelu_fwd_t` row.
- **attn11's side** (the training forward and a backward that takes the cached `t`) is attn11's
  change, recorded in attn11's CHANGELOG.

