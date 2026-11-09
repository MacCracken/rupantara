# 0002 — A forward optimization must leave every output bit-identical

**Status**: Accepted
**Date**: 2026-11-09

## Context

rupantara's correctness gate is **bit-identical parity with attn11's forward**.
attn11's test suite runs `ru_model_fwd` against its own `model_forward` and
requires `diffs == 0`. attn11 also delegates `ru_ln_fwd`, `ru_gelu_fwd` and
`ru_attn_core_fwd` to rupantara, and its 1,049 finite-difference gradient
checks run through them. Cached decode is held to the same standard: bit-equal
to the batch forward.

The 0.5.2 sweep had speed-ups available, and they come in two kinds. Some keep
each output's sequence of roundings: one `f64v_axpy` per key instead of a
broadcast buffer and one `f64v_fmadd` per 4 elements, or one `f64v_add` per
residual row instead of a scalar loop. Others do not: summing the scores in a
different lane order, fusing a multiply-add the scalar code rounds twice, or a
faster `tanh`. rosnet made the same split for its kernels in 1.2.2 (rosnet ADR
0002).

## Decision

**An optimization of the forward must leave every output with the same
sequence of roundings on every target it already matched.** Concretely:

- each output's terms are added in the same order, from the same starting
  value;
- each product and sum is rounded the same number of times. A packed op may
  replace a scalar op only if it rounds identically on that target. On x86_64
  `mulpd` + `addpd` qualify; `f64v_add` and `f64v_mul` qualify everywhere; a
  packed multiply-add on aarch64 (FMLA) does not;
- the vector/scalar split of an already-vectorized kernel stays where it was
  (the attention head dim's `hd4`), so no element changes path on a target
  where the two paths round differently.

The gate is a raw-bit golden dump of every public forward op, compared before
and after on x86_64 and on aarch64 (under qemu). In-repo, the 0.5.1 attention
cores are kept verbatim in `tests/reference.cyr`, and `tests/tcyr/reference.tcyr`
and the fuzz harness hold the new cores to them bit for bit.

## Consequences

- **Positive**: attn11's parity test keeps meaning what it says, and so does
  every consumer that compares bits (decode against batch, a stored golden). An
  optimization is reviewable: "did it change anything?" has an exact answer.
- **Negative**: the fastest options stay off the table. GELU's `tanh` is about
  96 ns per element, the scores and the LM-head dot products keep their 4-lane
  chains, and there is no AVX2 path.
- **Neutral**: the rule binds rupantara's own code. A toolchain or stdlib
  change can still move bits. cyrius 6.6.13's `f64_tanh` accuracy fix did,
  measured at up to 2 ulp in GELU (`docs/api.md`, *Numerical
  reproducibility*). Such moves are recorded when a pin changes, not prevented.

## Alternatives considered

- **Tolerance-gated optimizations** (accept any change within 1e-12): rejected.
  This would silently break the attn11 parity test and every bit comparison
  downstream.
- **Bit-identity on x86_64 only**: rejected for new code. aarch64 already
  differs from x86_64 where cyrius fuses the packed multiply-add. This rule
  keeps that set from growing.
