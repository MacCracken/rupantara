# rupantara

**रूपान्तर — transformation / metamorphosis.**

`rupantara` is AGNOS's sovereign **transformer forward** library — causal
MHA/GQA/MQA attention, the GELU MLP, LayerNorm, the pre-norm block, learned
positional embeddings, the forward pass over a block stack, the weight-tied LM
head, and KV-cache decode. It is **extracted from attn11** so consumers can run a
transformer **without depending on attn11-the-binary**. Pure Cyrius, f64.

**Scope:** the transformer **forward** (inference). Training / backprop stays with
**attn11**; tensors are **rosnet**; the pretrained-import + adapt reference is
**anukūlana**; the weight format is **tula**. rupantara is just the forward blocks.

## Status

**0.5.2: the GPT-2-shaped forward, hardened.** The forward is ported verbatim
from attn11: token + learned-positional embedding, LayerNorm, causal softmax
attention (MHA/GQA/MQA), GELU MLP, the pre-norm block, the block-stack forward,
and the weight-tied LM head + softmax. The op surface is `ru_*`-namespaced.

0.5.2 adds:

- validated model entries (`ru_cfg_check` and the `RU_ERR_*` codes; a rejected
  call writes nothing);
- allocation-free decode on caller scratch (`ru_model_fwd_s`,
  `ru_model_fwd_row_s`);
- attention 1.5× faster, bit-identical to 0.5.1.

0.6.0 adds:

- a committed golden gate: every public op's raw bits, on x86_64 and aarch64;
- the variable-length forward (`ru_model_fwd_n`, `ru_model_fwd_n_s`);
- a model scratch with no attention probabilities: 0.84 GiB at GPT-2 widths and
  T = 8192, down from 6.84;
- GPT-2-small throughput: about 14 tokens/s prefill and 10 tokens/s decode on
  one core (`docs/benchmarks.md`).

**192 assertions** pass on x86_64 and aarch64, plus a 20,000-round fuzz harness
and 14 benchmarks.
Start with [`docs/api.md`](docs/api.md). See also
[`docs/benchmarks.md`](docs/benchmarks.md), [`SECURITY.md`](SECURITY.md) and
the [audit](docs/audit/2026-11-09-audit.md).

**Re-fold + parity (2026-07-02):** attn11 **consumes** rupantara's leaf ops
(`ru_ln_fwd`, `ru_gelu_fwd`, `ru_attn_core_fwd`) — its full 1049 grad-check suite is
green in one binary — and rupantara's **whole** `ru_model_fwd` is **proven
bit-identical** to attn11's `model_forward` on identical params
(`test_rupantara_parity` in attn11's suite: 4 configs, `diffs==0`, `maxrel=0`).

**KV-cache decode (M2):** `ru_model_fwd_row` runs an incremental cached forward
**bit-identical to the uncached path** (`tests/tcyr/decode.tcyr`, 4 configs) with
per-layer K/V caches + `ru_argmax` greedy next-token. See
[`docs/development/roadmap.md`](docs/development/roadmap.md) +
[`docs/development/refold-plan.md`](docs/development/refold-plan.md). The Cyrius
pin is in `cyrius.cyml` (6.7.18 as of 0.6.0).

## Build & test

```sh
make build   # link-check the library (programs/smoke.cyr)
make test    # tests/tcyr/*.tcyr (the golden gate among them)
make test-aarch64  # every suite on aarch64, under qemu-aarch64
make fuzz    # seeded property fuzzing (tests/rupantara.fcyr)
make bench   # benchmarks (tests/rupantara.bcyr)
make dist    # regenerate dist/rupantara.cyr
```

## Consumers

`anukūlana` (Type-3 pretrained import: runs a real model's forward) and attn11
(its leaf ops and its parity test) pull `dist/rupantara.cyr` via
`[deps.rupantara]`, as will the murti load-seam. See the AGNOS planning docs
`type3-weight-import.md` and `software-port-path.md`.

## License

GPL-3.0-only.
