# rupantara — Roadmap (march to v1.0)

> **For a secondary agent picking this up cold.** Self-contained plan. Read this
> + `CLAUDE.md` (conventions) + `docs/adr/0001-rupantara-scope.md` (scope).
> rupantara is the transformer **forward** library extracted from attn11 — not
> training, not tensors, not the importer. Pre-1.0: surface may move, no freeze
> until v1.0.

---

## Working agreement

- **Build/test:** `make build` · `make test` · `make dist`. Pin in `cyrius.cyml`.
- **Flat modules:** stdlib includes only in `src/lib.cyr`; `src/*.cyr` flat. New
  module → `src/lib.cyr` + `cyrius.cyml [lib].modules` (dep order). The LSP flags
  cross-module symbols as "undefined" in standalone files — flat-module pattern;
  the build is authoritative.
- **Definition of done each bite:** `make test` green · `cyrius fmt` clean ·
  `cyrius lint` no `warn ` · `make dist` regenerated · CHANGELOG `[Unreleased]`.
- **Do NOT bump VERSION or git** — the maintainer cuts releases; work accretes
  under `[Unreleased]`.

## Discipline (inherited from attn11)

Cyrius-native, no BLAS/libc/autodiff. Where a gradient exists it is
**finite-difference-gated** (attn11's rule) — but rupantara is forward-focused, so
its primary correctness gate is **bit-identical parity vs attn11's forward** on a
fixed model+input. Any op ported from attn11 must reproduce attn11's numbers.

---

## Shipped

- **M0 ✅** buildable scaffold (version probe; smoke + CI green; `smoke.tcyr` 1).

---

## Remaining milestones

### M1 — extract the minimum GPT-2 forward from attn11 ⚠ cross-repo
**Goal:** run a GPT-2-small-shaped transformer forward, bit-identical to attn11.

**Progress:** ✅ **rupantara side COMPLETE + green** (suite total 43). Ported
bit-identical from attn11: `ln_fwd` (ε=1e-5) + `gelu_fwd` (tanh-approx); `rosnet`
0.2.0 wired for `linear_fwd`; `attn_core_fwd`/`attn_fwd` (causal softmax + GQA,
forward-only arena); `mlp_fwd`; `embed_fwd` (token + learned-abs pos);
`head_fwd_row`/`head_fwd` + `softmax_fwd`; the pre-norm `block_fwd` + block-stack
`model_fwd` over a caller-owned packed-parameter layout byte-identical to attn11's
MHA/dense block. Tests: `ops.tcyr` (9), `attn.tcyr` (12), `forward.tcyr` (21) —
attention known-answer, and a `model_fwd` plumbing-parity check (zeroed-weight
pass-through == embed→LN→head, byte-identical).
**Re-fold DONE (2026-07-02):** the op surface is `ru_*`/`_ru_*`-namespaced (12
public + 28 private; `comm -12` vs attn11 = empty) and **attn11 consumes
rupantara's three leaf ops** (`ru_ln_fwd`, `ru_gelu_fwd`, `ru_attn_core_fwd`
causal). attn11's full grad-check suite is **1049 green in one binary** with them
delegated → those three are **proven bit-identical** (the real parity gate; no
offline compare). **✅ Composition parity PROVEN (2026-07-02):** rupantara's whole
`ru_model_fwd` (embed → blocks → final LN → weight-tied head) is **bit-identical**
to attn11's `model_forward` on identical `g_params` — `test_rupantara_parity` in
attn11's suite, `diffs==0` / `maxrel=0.000000000` across MHA ±bias / GQA / MQA /
1–3 blocks (attn11 **1057** green). The M1-acceptance parity gap is CLOSED;
anukūlana's fidelity gate can lean on `ru_model_fwd`.
- **Scope** (new `src/*.cyr` modules, flat): token embed + **learned positional
  embed**; **LayerNorm** fwd; **softmax multi-head causal attention** fwd; **MLP
  (GELU)** fwd; the **pre-norm block**; the **forward over a block stack**;
  **weight-tied LM head + softmax**. Port from attn11 `ops.cyr` / `attn.cyr` /
  `tensor.cyr` (read them; port the converged shape, keep attn11's numerics).
- **Deps:** add `math` (exp/tanh/gelu/softmax) to `[deps].stdlib` and wire
  `[deps.rosnet]` (git+path+tag 0.2.0, `dist/rosnet.cyr`) for matmul — see the
  commented block in `cyrius.cyml`.
- **⚠ CROSS-REPO (needs its own explicit go):** re-point **attn11** to consume
  `dist/rupantara.cyr` (attn11 becomes a thin consumer of its own extracted
  forward — the kashi/sandhi extract-and-re-fold pattern). This edits the
  **attn11 repo** — do NOT touch attn11 without the maintainer's go.
- **Acceptance:** rupantara reproduces attn11's forward **logits bit-identically**
  on a fixed tiny model + input (a parity test); attn11 stays green consuming
  rupantara (its full suite unchanged).

### M2 — KV-cache decode ✅ DONE (2026-07-02)
- Incremental forward with a KV cache, **bit-identical to the uncached forward**
  (attn11's guarantee). Greedy/argmax next-token. (The full sampler zoo —
  temperature/top-k/top-p/beam — is the Autoregressive Type-2 reference's home;
  keep the boundary.)
- **Shipped** in `src/decode.cyr`: `ru_attn_core_fwd_row` / `ru_attn_fwd_row`
  (single-row causal core + projections, ported from attn11 `attn_*_fwd_row`),
  `ru_model_fwd_row` (one decode step: embed@pos → blocks over per-layer K/V caches
  → final LN → weight-tied head), `ru_decode_kv_count` + `_ru_kv_k`/`_ru_kv_v`
  (caller-owned cache layout), and `ru_argmax` (greedy next-token). Reuses
  `ru_ln_fwd`/`ru_mlp_fwd`/`ru_head_fwd_row` on one row.
- **Acceptance MET:** `tests/tcyr/decode.tcyr` — cached decode == uncached
  `ru_model_fwd` **bit-for-bit** (`diffs==0`) across 4 configs (MHA ±bias / GQA /
  MQA / 1–3 blocks) + a greedy-generation smoke; suite 43 → **48** green.

### M3 — architecture breadth (demand-gated)
- Add per consumer need: **RoPE** positions, **GQA/MQA**, **SwiGLU**, **RMSNorm**
  — e.g. when importing a Llama-class model requires them. Each ported from attn11
  + parity-tested. Don't build all up front.

### M4 — hardening + fuzz + bench
- Validate model-config shape (dims/layers/heads) against buffer/tensor sizes;
  reject malformed configs. Fuzz the config/forward entry. Bench forward
  throughput (tokens/s) → `docs/benchmarks.md`.

### M5 — security audit + SECURITY.md
- 6-dimension audit (memory safety · integer overflow · shape/bounds on untrusted
  configs · resource caps · fail-loud). Report + `SECURITY.md`.

### v1.0 — freeze & clean cut
- `docs/api.md` (frozen forward surface) + `STABILITY.md`; ≥1 downstream consumer
  green (anukūlana runs a GPT-2 forward through rupantara); maintainer bumps
  VERSION → 1.0.0 + tags.

---

## v1.0 criteria

- [ ] Public API frozen + documented (`docs/api.md` + `STABILITY.md`)
- [ ] **Bit-identical parity vs attn11 forward** (the core gate) + full coverage
- [ ] KV-cache decode == uncached
- [ ] Fuzz clean (malformed configs rejected) + bench (`docs/benchmarks.md`)
- [ ] Security audit + `SECURITY.md`
- [ ] ≥1 downstream consumer green (anukūlana)
- [ ] CHANGELOG complete; version consistency (CI docs gate)

---

## Gates / relations

- **Deps:** `rosnet` (matmul) + `math` (from M1). CPU-only.
- **Cross-repo:** M1's attn11 re-point edits the **attn11** repo → its own go.
- **Consumers:** `anukūlana` (Type-3 import), the murti load-seam. Keep the
  forward API stable for them.
- **Ecosystem:** the AGNOS planning docs `type3-weight-import.md` (gap #1, where
  rupantara is the transformer-forward prerequisite) + `software-port-path.md`.
