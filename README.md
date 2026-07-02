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

**0.1.0 — M0 buildable scaffold** (version probe only; smoke + CI green). The real
forward is extracted from attn11 at **M1** — see
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
