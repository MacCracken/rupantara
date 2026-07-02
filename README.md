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
extracted bit-identical from attn11: token + learned-positional embedding,
LayerNorm, causal softmax multi-head attention (+ GQA), GELU MLP, the pre-norm
block, the block-stack forward, and the weight-tied LM head + softmax. Runs green
standalone (43 assertions). The **cross-repo attn11 re-point** + full parity run
is the remaining step (its own go) — see
[`docs/development/roadmap.md`](docs/development/roadmap.md). Cyrius pin **6.3.27**.

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
