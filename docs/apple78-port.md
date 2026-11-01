# The Apple7/8 port

[Back to Splash](../README.md)

This branch runs Splash on M1 and M2 Macs, which upstream refuses to start on
(`DeviceCapabilities::kMinimumAppleGpuFamily`, 9 upstream and 7 here). What
that takes is a kernel policy for those GPUs and Metal kernels written for
them, kept alive across every upstream release. This file records the port:
why Apple7/8 need their own path, what the 1.5.0 merge decided, what is
deliberately held off, and what is left open.

Perf work for these chips has its own file: [Apple7/8 performance](apple78-perf.md).

## What Apple7/8 cannot do

Two facts drive everything else.

**No bfloat arithmetic.** Upstream's register tiles multiply bfloat16 8x8
simdgroup fragments. Apple7/8 have no bfloat operations, so the port runs the
same tiles with exact half weights and fp32 activations (the `F32` instances,
`LinearConfig::fp32Table`, `LinearInput::Table16F`): every product is exact,
only the fp32 accumulation order differs, so outputs match the shipped tiles
up to summation order and never bitwise. Numerics tests therefore compare
against fp64 references with a slack bound instead of demanding equality
(`dev/tuning/LinearNumerics.hpp`).

**MPP `matmul2d` runs far below the simdgroup MMA rate.** The staged tiles and
the MPP attention tile use the tensor-operation path, which on an M1 Max
stalled for minutes on a single prefill chunk. Hence the parallel set of
kernels suffixed `_sgf` or `_mma`: `gguf_linear_mma.metal`,
`moe_gguf_mma.metal`, `moe_mma.metal`, `vision_mma.metal`,
`paged_attention_sgf.h` and `sparse_attention_sgf.h`.

Three smaller constraints have bitten in practice:

- Apple7 under shader validation caps a threadgroup at 832 threads, so any
  kernel dispatched at 1024 needs `max_total_threads_per_threadgroup`
  (`shared/normalization.metal`, found by `make check-native-metal`).
- File-backed weights must stay in the residency set, or the host memory
  estimate swings by the whole weight size during every request (issue #3,
  reverted by `9dc653d`).
- A whole 2048-token prefill chunk is one GPU command. On these GPUs that can
  exceed what macOS will abort without starving the display, so isolated
  prefill commands are bounded to a few seconds by default
  (`--prefill-mode bounded|full`, `Scheduler.cpp`).

## The 1.5.0 merge

`apple7-m1-1.5` is upstream 1.5.0 (`a56988e7`) plus the port commits; its
predecessor `apple7-m1-1.4` was on 1.4.0 (`1482f6bb`). The gap was 185
upstream commits and 346 files, with 54 files changed on both sides and 20
that actually conflicted. The areas that mattered:

- **Attention geometry.** Upstream replaced the per-call layout dispatch with a
  compiled-geometry table (`kCompiledAttention`) carrying head dimension,
  window, learned sinks and gating, and split a tile into
  `SPLASH_ATTENTION_TILE_PARTS` threadgroups. The port's `_sgf` kernels now ask
  that table for a geometry instead of assuming 256-dimension heads, and
  `ops::registerTileFor` decides when the register tile has kernels for a
  geometry at all (below).
- **The register tile's chains.** Upstream re-derived the GGUF register tile's
  coefficient chains: `CG` counts 16-input chains alone, `CPG` and `MinChains`
  say how many chains a coefficient group and a term span, and a term of 64
  inputs closes against its own fp32 sum rather than a seed per chain. The
  tile's one-pass gate/up body is a second copy of the same loop, which the
  merge had to bring across separately or its outputs differ bitwise from two
  passes.
- **Staged decode policy.** Apple10 gained a measured wave law
  (`apple10Staged`) replacing the fitted tiers, and `stagedDecode` grew rows,
  bits-per-weight and pass-mode arguments. Apple7/8 keep their fitted tier: the
  law's residencies were measured on M5 and M6 hardware and describe neither.
- **MoE.** `moeGgufTile` now also takes the GPU's core count (Apple9 GPUs of 10
  cores or fewer stage everything), and the expert kernels' names gained a
  shared-slices suffix (`_s2`) for routings whose shared expert is two slices
  wide, which the new Flint family uses.
- **New model shapes.** Qwen3.5-122B's 64-value-head GDN, Flash-Next's
  16512-column rows and the 128-dimension attention geometry all arrive as new
  kernel instances; each `table16` GDN instance needs its `table16f` twin, or
  the register tile asks for a function that was never compiled.

## Decisions the merge took, and their cost

Each of these trades performance for correctness today. They are the first
things to revisit; see the [performance file](apple78-perf.md) for the
measurements each needs.

**The span chain is off** (`Shape::SpanChain = false`). It decoded one
coefficient per 64-input span and left the others unwritten, which the new
chains cannot tolerate: they address `cu[]` per coefficient group. It was worth
2.4-2.8% on MLX 4-bit's 27B projections at one lane, 0.7-1.6% at two to four.

**The attention register tile covers 256-dimension geometries only**
(`ops::registerTileFor`: no window, no sinks, one tile part, suffixes `""`,
`_kv2_g8` and `_kv2_g12`). Another geometry keeps the MPP tile, which Apple7/8
run correctly but slowly. A model in the 128-dimension geometry therefore works
on an M1 or M2 without running well, and nothing says so out loud yet.

**The MoE MMA tile takes routings of one shared slice** (`moeGgufTile` falls
back to staging for two). The `_s2` MMA expert kernels were never compiled, so
this is the difference between a working fallback and a missing function at
dispatch time. A two-slice MoE model on an M1/M2 stages its experts.

**The 64-row staged tile got MMA instances** (plain, residual and gate/up in
bf16, and the fused kernel) because upstream now runs prefill chunks of 33 to
64 rows through it. They are compiled from the existing template rather than
measured: whether Apple7/8 want the 64-row tile at all, or 32, is unmeasured.

## Keeping the branch rebasable

Upstream moves, and the fork has to move with it without re-deriving everything
in this file. The rule for work on this branch:

- **Add, do not amend.** A new `if` block between two of upstream's lines merges
  silently; a condition extended on one of upstream's lines conflicts with
  every future change to that line. When a choice exists between an inserted
  guard and an edited one, take the insertion, even when it reads slightly less
  locally.
- **Fork-only logic goes in fork-only files.** `moe_gguf_mma.metal`,
  `paged_attention_sgf.h`, `gguf_linear_mma.metal` and the `apple78` tools carry
  the port entirely, which is why they cost nothing at merge time. Policy that
  must live inside an upstream function should be a one-line call to a fork
  function defined below it — `apple7StagesSpanFormat` and the four-lane clause
  that uses it are two lines of upstream edited, and the rest is insertion.
- **Never rename, reformat or reorder upstream's code** for the port's
  convenience, and do not carry a cosmetic diff. It all shows up as conflict.
- **Measure the surface before a merge, not during:**

  ```sh
  dev/tools/apple78-rebase-audit.py --upstream <upstream-ref>
  dev/tools/apple78-rebase-audit.py --upstream <upstream-ref> --show runtime/ops/LinearGguf.cpp
  ```

  Against 1.5.0 (`a56988e7`) the branch edits 58 of upstream's files. Of the 424
  upstream lines its diff removes, 36 only *moved* within the file, which git
  resolves on its own; the other 388 are real edits, concentrated in a handful
  of places: 41 lines in `linear_gguf_sgmatrix.metal` (upstream's register tile
  given the port's template arguments), 31 in `hyper_connection.metal`, 28 in
  `LinearGguf.cpp`, 21 in `block_indexer.metal`, 20 in `gdn.metal`. `--show`
  prints each one with what replaced it, so a conflicted file is worked through
  as a list rather than guessed at, and names lines that merely moved. Files
  matching `apple7`, `_sgf`, `_mma` or `apple78` are marked as the branch's own,
  because they will never conflict: they are not upstream's.

  The rule works when the numbers hold. The four-lane affine rule (`d9d77ed0`)
  is 23 added lines in `LinearGguf.cpp` and no deleted ones: a new predicate
  under the other predicates, and one inserted `if` above the register tile's.
  Nothing upstream was rewritten, so the next merge will not hear about it.
## Verification

Every claim about this branch needs a machine of its own, because upstream's CI
matrix is M3 or newer and its hosted jobs build but never run the Metal tests.

On an M2 Max (30-core GPU, family 8, macOS 27) — the only Apple7/8 machine with
its GPU free to be measured: hosted CI runs a subset of it, on hardware that
turns out to be no Mac at all (see below):

- `make` builds the runtime and every Metal kernel.
- `make test-engine-cpu` passes.
- All 40 `test-engine-metal` commands pass, each under `MTL_SHADER_VALIDATION=1`
  (`dev/tests/run-metal.sh --list` prints them). The run is about 46 minutes on
  this machine, and four groups dominate it — the same groups that catch this
  branch's own regressions, so they are what a CI job has to afford:
  `q8-flash-attention` 905 s and `q8-chunked-prefill` 268 s; `gguf-moe` 460,
  `gguf-moe-e512` 429, `gguf-moe-sigmoid` 244; `gguf-projection`,
  `paged-attention-plan`, `gdn-decode` and `ane-ffn faults` about 80 each;
  everything else together under 180.
- CI on this fork is green end to end (`e43c18e2` made it run at all, by naming
  runners a personal fork can use; `b7b3d1c3`, `d9d77ed0` and `9b9a734c` fixed
  what the first runs exposed): the CPU job, Python compatibility on 3.12, 3.13
  and 3.14, and a Metal job on `macos-26` gating the port's kernel numerics and
  plan selection. Two of its early failures were this branch's own: a residency
  set that every Mac offers and no hosted runner does, and a Python test whose
  0.2 s startup deadline let the scheduler rather than the code decide how many
  engines started. Three checks disagree with the VM rather than the branch —
  qsa_pool's mean keys in all ten of its cases, paged_attention's test that a
  bf16 store leaves its source bits and neighbouring page slots untouched, and
  metal-backend's function-replacement check, which reports "replacement method
  is missing" where no compiler backs it — and run as an informational step.

Three defects the merge introduced were found this way and fixed, each by a
test rather than by reading: a missing `gguf_decode_mma_q4k_m64_a`, one-pass
gate/up outputs differing bitwise from two passes across 282 cases, and a
two-slice MoE prefill plan naming `moe_expert_gguf_mma_m32_a_s2`.

## Open work

- Re-derive the span chain against the new coefficient chains.
- Fit Apple7/8's split policy across core counts: the tiers were measured on a
  32-core M1 Max and a 30-core M2 Max, and base M1/M2 (7-10 cores) and the
  M2 Ultra (up to 76 cores) are unmeasured.
- Compile the register attention tile for the 128-dimension geometry.
- ~~Measure the Neural Engine's share on these GPUs, dense first~~ done, and it
  needs nothing (docs/apple78-perf.md, Taken): the runtime's calibration beats
  every fixed share tried, by 24.1% over the GPU alone. MoE remains open, and
  harder — the routed experts are gathered per row, which is the opposite of the
  static shape an ANE program wants.
- ~~Run the Metal suite on Apple7/8 hardware in CI~~ the hosted runners are
  **not** this hardware class, which is what the first runs had to establish:
  GitHub's `macos-26` arm64 runner is a virtual GPU with no residency set, no
  ANE, and its own answers for bf16 stores and QSA pool means. The job that
  exists now gates on what it can settle — the port's kernel names, dequant,
  rotation, projection numerics, norm tables, plan selection, and a build of
  every Metal kernel — and runs the three that disagree as a reported,
  non-gating step. `dev/tests/device-info.m` prints what the machine claims,
  so a future failure can be read rather than guessed at. A real M1/M2 remains
  the only place the full suite means anything, which is what the
  `release-hardware` dispatch matrix is for when someone's own Mac is registered.
