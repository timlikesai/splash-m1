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

## Verification

Every claim about this branch needs a machine of its own, because upstream's CI
matrix is M3 or newer and its hosted jobs build but never run the Metal tests.

On an M2 Max (30-core GPU, family 8, macOS 27):

- `make` builds the runtime and every Metal kernel.
- `make test-engine-cpu` passes.
- The `test-engine-metal` commands run under `MTL_SHADER_VALIDATION=1`.
  Recorded timings for each command live with the benchmark harness so the
  suite can be split into a job a hosted runner can finish.

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
- Measure the Neural Engine's share on these GPUs, dense first and MoE after,
  where the GPU is comparatively slower and the case for splitting is stronger
  than on the hardware upstream tuned against.
- Run the Metal suite on Apple7/8 hardware in CI: GitHub's hosted macOS runners
  are exactly this hardware class, which is why upstream's comment says they
  cannot run the Metal tests at all.
