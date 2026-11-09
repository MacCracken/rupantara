# Architecture Decision Records

Decisions about rupantara: what we chose, the context, and the consequences we
accept. Use these when a future reader would reasonably ask *"why did we do it
this way?"*

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`, zero-padded to four digits. Never
  renumber.
- **One decision per ADR.** When a decision supersedes a prior one, add a new
  ADR and set the old one's status to `Superseded by NNNN`.
- **Status lifecycle**: `Proposed` → `Accepted` → (optionally) `Superseded` or
  `Deprecated`.
- Start from [`template.md`](template.md).

Durable rationale belongs in an ADR. The API contract lives in
[`docs/api.md`](../api.md). Volatile state (versions, counts, in-flight work)
lives only in [`docs/development/state.md`](../development/state.md), never here.

## Index

| ADR | Status | Subject |
|---|---|---|
| [0001](0001-rupantara-scope.md) | Accepted | rupantara's scope: the transformer forward, extracted from attn11 |
| [0002](0002-optimizations-preserve-bits.md) | Accepted | A forward optimization must leave every output bit-identical |
| [0003](0003-validate-at-model-entries.md) | Accepted | Validate at the model entries; leaf kernels trust their caller |
