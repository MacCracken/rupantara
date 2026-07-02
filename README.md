# rupantara

**रूपान्तर — transformation / metamorphosis.**

`rupantara` is AGNOS's sovereign **transformer forward** library — attention, MLP,
LayerNorm/RMSNorm, the pre-norm block, positional embeddings, the forward pass
over a block stack, weight-tied LM head, and KV-cache decode. It is **extracted
from attn11** so consumers can run a transformer **without depending on
attn11-the-binary**. Pure Cyrius.

**Scope:** the transformer **forward** (inference). Training / backprop stays with
**attn11**; tensors are **rosnet**; the pretrained-import + adapt reference is
**anukūlana**; the weight format is **tula**. rupantara is just the forward blocks.

## Status

**0.4.0 — transformer forward + attn11 re-fold + KV-cache decode.** The
GPT-2-shaped forward ported verbatim from attn11: token + learned-positional
embedding, LayerNorm, causal softmax multi-head attention (+ GQA/MQA), GELU MLP,
the pre-norm block, block-stack forward, weight-tied LM head + softmax. Full public
+ private op surface is `ru_*`-namespaced. Runs green standalone (**48 assertions**).

**Re-fold + parity (2026-07-02):** attn11 **consumes** rupantara's leaf ops
(`ru_ln_fwd`, `ru_gelu_fwd`, `ru_attn_core_fwd`) — its full 1049 grad-check suite is
green in one binary — and rupantara's **whole** `ru_model_fwd` is **proven
bit-identical** to attn11's `model_forward` on identical params
(`test_rupantara_parity` in attn11's suite: 4 configs, `diffs==0`, `maxrel=0`).

**KV-cache decode (M2):** `ru_model_fwd_row` runs an incremental cached forward
**bit-identical to the uncached path** (`tests/tcyr/decode.tcyr`, 4 configs) with
per-layer K/V caches + `ru_argmax` greedy next-token. See
[`docs/development/roadmap.md`](docs/development/roadmap.md) +
[`docs/development/refold-plan.md`](docs/development/refold-plan.md). Cyrius pin **6.3.27**.

## Build & test

```sh
make build   # link-check the library (programs/smoke.cyr)
make test    # tests/tcyr/*.tcyr
make dist    # regenerate dist/rupantara.cyr
```

## Consumers (planned)

`anukūlana` (Type-3 pretrained-import — runs a real model's forward), the murti
load-seam — pull `dist/rupantara.cyr` via `[deps.rupantara]`. See the AGNOS
planning docs `type3-weight-import.md` + `software-port-path.md`.

## License

GPL-3.0-only.
