# Security

## Supported versions

rupantara is pre-1.0. Fixes land on `main` and ship in the next release. Only
the latest release is supported.

## Reporting a vulnerability

Please report privately through GitHub's **Security → Report a vulnerability**
on the repository (a private advisory), not in a public issue. Include the
version (`rupantara_version()` or `VERSION`), the config and inputs that
trigger the problem, and what you observed.

## Trust model

rupantara runs a transformer forward over buffers its caller owns. It parses no
files and no network data, and makes no system calls of its own beyond the
allocator. On AGNOS, `linear_fwd_gpu` also makes the `#83` dispatch syscall.
The inputs that can be wrong are the config, the packed parameters, the token
ids and the decode positions.

- **Validated (0.5.2).** The model entries check the config (`ru_cfg_check`),
  every token id against `[0, V)`, the decode position against `[0, T)` and
  the run length of the variable-length forward against `[1, T]` (0.6.0;
  only the first `n` ids are read or checked).
  They also check null pointers. Every buffer size is computed with
  overflow-checked arithmetic and capped at 2^56 elements, so a hostile config
  cannot wrap a size into a small allocation. A rejected call returns a
  negative code and writes nothing. The validated entries are `ru_model_fwd`,
  `ru_model_fwd_s`, `ru_model_fwd_n`, `ru_model_fwd_n_s`, `ru_model_fwd_row`,
  `ru_model_fwd_row_s`, `ru_attn_fwd`, `ru_argmax`, and the sizing functions.
- **Caller's contract.** The leaf kernels (`ru_ln_fwd`, `ru_attn_core_fwd`,
  `ru_head_fwd_row`, …) take dims and buffers on trust and check nothing, as
  rosnet's do. Each buffer must hold what `docs/api.md` says for its dims. A
  consumer that builds its own forward from them should run `ru_cfg_check`
  once.
- **Parameter values** are not checked. NaN or Inf weights give NaN or Inf
  outputs, never an out-of-bounds access. `ru_argmax` on a NaN row returns 0.
- **Memory.** `ru_model_fwd` and `ru_model_fwd_row` allocate per call from a
  bump allocator that never frees. A long-running caller should use the `_s`
  entries, which allocate nothing.
- **Concurrency.** The forward entries keep no global state and are reentrant
  on distinct buffers. `linear_fwd_gpu` is not reentrant, because its tile
  scratch is module-global.

## Audits

- [`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md): the 0.5.2
  P(-1) sweep. It fixed a heap out-of-bounds write on decode past the context
  window, unchecked configs, and unbounded heap growth in decode loops.
