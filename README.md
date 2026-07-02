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

**0.2.0 — M1 the transformer forward (rupantara side)** — the GPT-2-shaped forward
ported verbatim from attn11: token + learned-positional embedding, LayerNorm,
causal softmax multi-head attention (+ GQA), GELU MLP, the pre-norm block, the
block-stack forward, and the weight-tied LM head + softmax. Runs green standalone
(43 assertions). The full public + private op surface is `ru_*`-namespaced.

**Re-fold landed (2026-07-02):** attn11 now **consumes** rupantara's three leaf
ops — `ru_ln_fwd`, `ru_gelu_fwd`, `ru_attn_core_fwd` (causal) — and its full
grad-check suite is green in one binary, so those three are **proven bit-identical**
(the real parity gate, no offline compare). rupantara's **composition** ops
(`ru_embed_fwd`, `ru_head_fwd`, `ru_model_fwd`, …) are ported + internally tested,
but attn11 keeps its own, so they are **not yet cross-validated** against attn11 —
anukūlana is their first real consumer. See
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
