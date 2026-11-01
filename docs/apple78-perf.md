# Apple7/8 performance

[Back to the Apple7/8 port](apple78-port.md) · [Splash performance](performance.md)

Notes on making M1 and M2 fast, and on how to tell whether a change did. Nothing
here is a claim about speed until the named measurement says so: the numbers
upstream publishes are M4 and M5 hardware, and this branch's kernels are a
different implementation with a different balance.

## Measuring

Rules that keep results comparable:

- One GPU workload at a time. A benchmark running beside the Metal suite or a
  served model measures both of them badly.
- Record the machine with the numbers: GPU family, core count, macOS, memory,
  and the build's identity. The same kernel policy means different things on a
  10-core M2 and a 64-core M2 Ultra, and `gguf-projection-benchmark`,
  `gguf-moe-benchmark` and `dev/tests/device-info.m` all print what the machine
  is: family, core count, the families Metal claims, and whether it offers
  residency sets.
- Median, not best. `gguf-projection-benchmark` reports a median over its
  weight copies; `attention-sweep` and the MoE benchmark do the same. Keep the
  rounds the tool asks for, and do not compare across different round counts.
- Compare like with like: candidate against baseline from the same commit
  except for the one change, on the same machine, in the same session.
- For an end-to-end number, build the baseline in a `git worktree` beside the
  tree under test rather than checking it out: two binaries, no stash, no
  half-built objects, and the runs interleave (before, after, before, after) so
  a warming machine drifts both ways. The four-lane decode gain in Taken was
  measured this way, and the interleaving is what makes its 0.3% spread
  believable. It also caught something an A/B in one tree would not: the
  revision first chosen as "before" carried b7b3d1c3's residency bug and
  segfaulted on a Mac, while the same revision passed hosted CI, which has no
  residency sets to break.
  See [the harness](../dev/benchmarks/apple78) for the mechanics.

The tools that already exist, and what each is for:

| Tool | What it measures |
| --- | --- |
| `make benchmark-gguf-projection GGUF_PROJECTION_ARGS=...` | one projection, both decode tiles, every lane count and K split, and with `prefill=R,...` the prefill tiles: the workhorse for split and tile policy |
| `make benchmark-gguf-moe GGUF_MOE_ARGS=...` | one MoE layer on the device's plan and the other GGUF tile, split into gate/up and down time |
| `make benchmark-attention-sweep ATTENTION_SWEEP_ARGS=...` | paged attention prefill and verify across splits, the tile choice's cost |
| `make benchmark-decode-profile MODEL=...` | a real model's decode cycle, where the per-step budget of attention, GDN, MoE and sampling shows up |
| `make tune-kernels` | the measured kernel policy itself, per device |
| `dev/tests/engine/*_metal_test` | numerics, not speed: every candidate change has to keep these passing |

## What these chips are

The port exists because of two absences, and they set the shape of the
optimization problem rather than just its constants.

**No bfloat arithmetic.** The shipped tiles multiply bfloat16 fragments; here
weights enter as exact half and activations as exact fp32, so a product spans
half×fp32 and the accumulation is fp32. That is exact, and it is work: on a
machine with no bf16 unit, most of the arithmetic that Apple9 gets from tensor
operations lands on the ordinary pipes. It also doubles the bytes of the
prepared activation table (`LinearInput::Table16F`, fp32, against `Table16`'s
bf16).

**MPP `matmul2d` runs far below the simdgroup MMA rate.** Where Apple9 and
Apple10 stream weights through the tensor path, Apple7/8 win by dequantizing
and multiplying on the SIMD pipes instead. That is why the port carries a
second kernel family rather than tuning the first.

The rest is a scale problem, not a feature problem:

- **Bandwidth spans 6x across the family.** Base M1 is 68 GB/s, M2 100 GB/s,
  the Max parts about 400 GB/s, the Ultra 800 GB/s. Decode of a 4-bit model is
  DRAM-bound, so a policy fitted on a Max is wrong at both ends: the base chips
  saturate their bandwidth at a fraction of the threadgroups, and the Ultra
  wants far more of them before it is streaming.
- **Core counts span 8x**: 7-10 on the base parts, 16-19 on Pro, 24-38 on Max,
  24-76 on Ultra. Every rule written as "threadgroups per core" is a guess
  outside the machines it was measured on, and the fork's tiers were measured on
  a 32-core M1 Max and a 30-core M2 Max only.
- **Threadgroup size.** Apple7 under shader validation caps a threadgroup at 832
  threads, so a 1024-thread kernel needs `max_total_threads_per_threadgroup`.
  Anything new upstream that dispatches wider has to be checked here, not just
  compiled.
- **The Neural Engine is there** — 16 cores on every M1 and M2 variant, with the
  same CoreML path the later chips use, and no family gate in the ANE code:
  `AneFfnStartup` calibrates the split against the machine it runs on and
  remembers the calibration per device. So the dense FFN split is live here
  (`ane-ffn kernels`, `split`, `faults` and `program` pass on an M2 Max), and it
  self-tunes rather than inheriting a table.
- **The GPU is the weak half here.** On M3 and newer the GPU gained tensor
  throughput faster than the ANE did, so upstream's split, tuned on M4/M5
  hardware, is comparing a faster GPU against the same ANE. On Apple7/8 the
  ratio runs the other way, which is the reason to expect the ANE to earn a
  larger share here than upstream's measurements imply.

## Where the time goes

The port's own measurements, restated as a list of what to attack. All of these
are one-lane decode numbers on the 27B's shapes unless stated.

| Cost | Known | Unknown |
| --- | --- | --- |
| Widen-to-fp32 arithmetic and the fp32 activation table | the register tile beats staging on every format that can take it | whether fp16 activations on the native half MMA path would win, and for which formats |
| Split count / threadgroups per core | fitted tiers: staged on Apple7/8 keeps one tier, 24 threadgroups per core with a 512-input floor (`kMmaTiers`), and the register tile takes `kRegisterTiers` (`{4,256}`, `{32,1024}`, `{3,512,16}`), fitted on a 32-core M1 Max and a 30-core M2 Max | whether a fitted occupancy law, as Apple10 now has (`apple10Staged`), beats a tier across 7 to 76 cores |
| Attention tile | the MPP tensor path holds most of a long-context prefill chunk or verify cycle, which is why the register tile exists; the end-to-end it enabled is measured (27B on an M1 Max: cold 8K prefill 135 tok/s, 1.5K append 121) | the 128-dimension, window and sinks geometries have no register kernels, so a model that uses them runs the slow path here |
| Prefill command length | one prefill command is bounded at 2000 ms (`kDefaultPrefillCommandMilliseconds`, `runtime/engine/Scheduler.hpp`; `--prefill-bound-ms`, `--prefill-mode full` to drop it) so macOS does not abort a GPU that keeps the display waiting | the right bound per machine class, and whether the attention piece-work bound (`kPrefillAttentionPieceWork`, 2^25 row-history pairs) is what actually protects the display |
| MoE experts | simdgroup MMA expert tiles for shared experts of one **or two** slices, measured on Qwen3.6-35B-A3B's shapes and Flint's; a wider routing still lands on the MPP `matmul2d` | the ANE's share of an MoE layer — the runtime refuses the dense FFN split outright (`ane_ffn_share: 0` on a real MoE), so the shared expert would need its own program; at top 8 of 256 with a 512-wide shared expert against 8 × 512 routed, that is about one ninth of the layer's FFN work |
| Coefficient chains | the span chain measured 2.4-2.8% at one lane | its form under the new per-coefficient-group chains, which is why it is off |

## Taken

- **A four-lane step of the large 64-input-group affine projections stages on
  the MMA kernels** (d9d77ed0, refined in bf1198d5). Found by the harness's
  first baseline: at one to three lanes the register tile wins by 6-40%, at four
  it loses by 6.7-12.8%, and Q4_K/IQ4_XS keep the register tile at every width.
  Three of 55 sweep cases moved, none regressed, and on the 27B-4bit end to end
  a batch of four decoded **5.7% faster** (347.5 → 327.7 ms per cycle) with one
  to three lanes unmoved -- the signature of a change that touches only
  four-lane plans.

  The first version of the rule was too wide, and the 35B MoE caught it: with no
  size condition it *cost* that model time on its small projections (-5.9% on
  the o at 2048x4096) and its four-lane cycle measured +0.4%, no gain at all,
  because the loss cancelled the vocabulary's +13.1%. Measured across thirteen
  shapes, the sign at four lanes follows **N x K** -- the weights' element count,
  not their bytes, since 2-, 4- and 8-bit versions of a shape land within a point
  of each other -- and splits cleanly at 1<<24 elements: above it staging wins
  (+1.1% at 2048x8192 up to +13.1% on the vocabulary), below it loses (-0.6% at
  4096x2048 to -15.3% at 2048x2048). With that floor the MoE model's four-lane
  cycle drops **4.5%** (85.37 → 81.51 ms) and the 27B's **5.9%** (347.32 →
  326.88 ms), reproduced against a pre-change worktree, runs interleaved, three
  samples each. The lesson is in the rule's comment: a policy fitted on one
  model's shape list is a guess about every other model's.

- **Two-slice MoE routings run their experts on tensor hardware now.** The MMA
  expert kernels existed only for a one-slice shared expert, so Flint-class
  routings — sigmoid router, shared expert two experts wide — fell to the
  staged tile and its MPP `matmul2d`. `moe_gguf_segment<Slices>` was already
  generic, so this was four instantiations (`moe_expert_gguf_mma_m{8,32}_{a,g}_s2`)
  and lifting the `sharedSlices == 1` condition. On a 30-core M2 Max at q3k,
  7 rounds, `gguf-moe-benchmark flint`: a 2048-row prefill chunk **48% faster**
  (19.2 ms against 37.2), 512 rows 41%, 64 rows 19%, and pooled decode 21-27%
  faster at 8 to 32 rows. The one case the staged tile still wins is *random*
  routing at 8 and 16 rows, by 5% and 4% — a single sequence's verify rows share
  their experts far more than that, so the policy keeps MMA throughout.
  `gguf-moe-sigmoid` builds the test with `GGUF_MOE_SHARED_SLICES=2`, so the
  numerics here are the shipped test, not a custom run.
- **The dense ANE split needs no tuning on these chips** (measured, not
  assumed). Qwen3.8-27B-4bit on the M2 Max, prefill FFN share forced with
  `--ane-ffn-share`, medians of two samples over eleven scenario and length
  combinations (`dev/benchmarks/apple78/ane.py`): a 0.25 share is 15.4% faster
  than the GPU alone, 0.5 is 17.1% faster, and the runtime's own calibration —
  which chose 0.41 with a 256-row minimum — is **24.1% faster**, beating both
  fixed shares on every row including the 480-row chunks below its stated
  minimum. Calibration costs 75.7 s once per model, device and macOS version,
  and `ane::remember` keeps it in the per-user cache, so a restart is free. The
  opportunity listed below as "give ANE a fairer share" is closed: a fitted
  share beats a hand-picked one here too.
- **Qwen3.6-35B-A3B, both packages, measured here** (a real MoE at last: 256
  experts, top 8, shared expert one slice, so the MMA expert tile applies).
  End to end the **MLX 4-bit package is the faster one**: 34.2 ms per one-lane
  decode cycle against the GGUF UD-Q3_K_XL's 40.9 (184 against 161 tok/s), and
  1699 against 1762 ms for a 2048-row prefill — while holding *more* expert
  bytes: 17.3 GB of experts resident against 13.6 GB. The per-layer tiles say
  why: `af4g64` beats `q3k` on the same MMA tile by 18% at an 8-row random
  decode (0.701 against 0.858 ms) and 20% pooled (0.335 against 0.417). So MoE
  decode on this machine is not bandwidth-bound -- it spends its time
  dequantizing, where the affine format has no super-block scale chains to
  walk. That is an argument for item 2 (fp16 activations, which halves what an
  activation table costs) and for cheaper dequantization on the MMA tile, not
  for shaving bits off the weights. At four lanes the two packages converge
  (323 against 324 tok/s), where the per-token weight traffic that does
  differentiate them is amortized across the batch. The advice is scoped to
  this machine and this context (30 cores, 4096 tokens): on the 60-core M2
  Ultra the ordering of a GGUF-against-MLX pair reverses (below), and the two
  measurements ran different quantizations, so the strict cross-check is the
  same two packages on both chips.
- **Streaming the routed experts is a last resort, and the numbers say so.**
  With every expert resident (17.3 GB of them) the 35B refuses to start past a
  4096-token context on 32 GB -- "unable to allocate a lane's state: host memory
  reserve protected" -- so streaming is how these models run on a small Mac at
  all. `decode-profile` on the MLX package, 6 cycles per width, 512-token prompt,
  with the fused column (what a user feels) and the sum of attributed dispatch
  times: at `--expert-memory 64G` everything is resident, 553.9 ms prefill and
  102.3 / 78.6 / 96.3 / 121.0 ms for B1-B4 cycles, against 477 / 34.6 / 47.3 /
  61.6 / 80.3 ms as separate dispatches. Give it `2G` -- 15 GB of experts
  streamed from the SSD cache, which is what a 16 GB M1 does with a 35B MoE --
  and the dispatch *work* rises only 1.2-1.7x (817 ms prefill, 42.2 / 79.2 /
  87.1 / 154.6 ms), while the fused commands take 9.7x on prefill and 2.9-10x
  on the cycles (5384.8 ms, 292.8 / 518.8 / 980.6 / 949.7). The GPU is not
  doing more, it is waiting: which is worth knowing before anyone recommends a
  35B MoE on an 8 GB machine, and worth re-measuring on family 7, where both
  the GPU and the SSD controller are a generation older.
- **The same model at 60 cores and 64 GB (the M2 Ultra).** Three findings,
  the same commit. First, the ceiling the 32 GB M2 Max hit at 4096 tokens is
  a memory problem, not a model problem: on 64 GB every context from 8K to
  64K starts with every expert resident (18.77 GB of them in the GGUF
  UD-Q4_K_XL package, 32.64 GB in the MLX 8-bit one), and the model's full
  256K native window starts too, via `serve-native` at 128K and 262K. Second,
  streaming is worse in proportion, not different in kind: `decode-profile` at
  `--expert-memory 64G` against `2G` takes the fused commands 7.9x (GGUF
  prefill) to 19.8x (MLX prefill) and 4.3-16x on the decode cycles, against
  1.6-3.3x of attributed dispatch work, and `moe_expert_load` is 56.9% of the
  GPU time in the starved MLX prefill (0.0-6.5% of it when the experts are
  resident) — the same stall-not-arithmetic the M2 Max found, wider, with the
  8-bit package streaming worse in proportion than the 4-bit one (double the
  bytes, double the SSD work per token). Third, the package ordering
  reverses: with experts resident, the GGUF package is the faster one at every
  width and context (16K-64K) — 0.8% at one lane, 7.1% at four (458 against
  495 ms), and 2.1% on a cold 4k prefill (7413 against 7573 ms) — against
  the M2 Max's 16% for MLX at 4096 tokens. The caveat is the quantizations:
  Q3_K_XL against 4-bit on the M2 Max, Q4_K_XL against 8-bit on the M2 Ultra,
  so the reversal is the shape, not the level; the strict cross-check is the
  same two packages on both chips, and the package advice is scoped to
  machine and context until it runs.

## Ranked opportunities

Ordered by expected value per unit of risk, not by size.

**1. Fit the split policy to core counts, not to two Max chips.** The rule is
`kMmaTiers` -- one tier of 24 threadgroups per core with a 512-input floor --
for the staged tile, and `kRegisterTiers` for the register tile, fitted on a
32-core M1 Max and a 30-core M2 Max. Apple10 replaced exactly this kind of rule
with a
measured occupancy law (`apple10Staged`: resident wave, DRAM-saturating count,
trailing-wave cost) fit to sweeps on the M5 Max, M5 Pro and M6. The same fit is available here: sweep
splits from 1 to 16 across the 27B and 35B shapes with
`benchmark-gguf-projection` on this M2 Max, and again on the M2 Ultra; the base
parts can be measured on hosted CI runners, which are this GPU class.
*Decides:* whether a law beats the tier, and by how much at each end of the
range.

**2. Try fp16 activations on the native half MMA path — smaller than it looks,
now measured.** Apple7/8 do have fast 8x8 half×half simdgroup MMAs; what they
lack is bfloat. Today's tiles pair half weights with *fp32* activations
(`gguf_mma_tile.h`: "every product is exact and only the fp32 accumulation
rounds, and activations past half's range stay finite"). The precision half of
that argument is softer than it reads: bf16 has an 8-bit mantissa, which fits
fp16's 11-bit exactly, so a bf16 activation cast to half loses nothing in
significand — the real reason for fp32 is **range**, fp16's 65504 ceiling and
its subnormal floor below 6.1e-5, plus the bitwise story `table16f` exists to
give.

What the change could win is now bounded by a measurement rather than by
guesswork: `decode-profile` on the 27B MLX package, one-lane decode cycle,
attributed by kernel --

| Part of the cycle | Share |
| --- | --- |
| `gguf_decode_sgf_af4g64_l1_{a,r,g}` (the projections themselves) | 84.3% |
| `…_a_f32` (the fp32-output projection) | 7.0% |
| everything writing fp32 tables (`norm_rms_table16f_decode` 2.6, `verify_gdn_fused_table16f` 2.8, `decode_linear_gguf_prepare_f` 0.5, `verify_attention_gate_table16f` 0.1) | **6.0%** |

Half the table bytes is the whole prize on the store side, and the tiles read
activations of `rows × K × 4` bytes against `K × N × bits` of weights -- at one
lane 164 KB against 44 MB, so the load side is noise. Realistically this is a
2-4% decode change for a per-format numerics re-derivation through the
fp64-referenced slack bounds in `dev/tests/engine/LinearNumerics.hpp`, and it
would give up the exactness argument that makes Apple7/8's outputs defensible.
*Decides:* whether 2-4% is worth re-opening the numerics story; the sizing says
pursue it only after items 1, 7 and 9.

**3. ~~Give the ANE a fair share on these GPUs.~~ Closed** — see Taken. The
calibration's own choice beats a fixed share by 7 points of prefill time, so
the remaining ANE question is not the share but the 75.7 s it costs on a new
macOS version, and whether a family-fitted starting guess would shorten the
search rather than replace it.

**4. ~~MoE on the ANE, shared expert first.~~ Closed as not worth a program.**
Measured on Qwen3.6-35B-A3B (256 experts, top 8, shared expert one slice) on
the M2 Max: an MoE step is almost entirely expert time -- 40 layers at 0.813 ms
of gate/up and down kernels is 32.5 of a 34.2 ms one-lane decode cycle -- and
the shared expert is one route in nine, so ~11% of a decode step and ~5% of a
2048-row prefill chunk (2.2 of 19.6 ms per layer, 40 layers, 1699 ms). The
runtime already refuses the dense FFN split for MoE outright (`ane_ffn_share:
0`, `ane_ffn_bytes: 0` on the real model). A new ANE program class for one
slice, with routed passes still on the GPU, could therefore win a few percent
at absolute best and would carry every risk of the split's synchronization --
where the dense split won 24% because the whole FFN moved. MoE's FFN time is
better spent on the expert tiles themselves, which is what the `_s2` work and
item 2 (fp16 activations, which halves what an activation table costs) address.

**5. The register attention tile has one head dimension and no window — but
what that costs today is 0.7%.** `runtime/metal/abi/KvExtent.h` fixes
`SPLASH_KV_HEAD_DIMENSION` at 256, and `registerTileFor` also refuses any
`variant.windowTokens` or `kernel.sinks` geometry, so anything not 256-dim,
single-part, unwindows and sinkless geometry runs the MPP path.

Measured on the models actually served here: Qwen3.8-27B and Qwen3.6-35B-A3B
both have `head_dim: 256`, `full_attention_interval: 4` and GDN linear attention
between, so their targets take the register tile (`prefill_attention_q8_split_sgf`,
`verify_attention_q8_split_sgf`) and the MPP fallback costs nothing on the
target. **The draft does not escape:** the shared DFlash2 draft is
`head_dim: 128` with `sliding_window: 2048`, so its attention is
`draft_attention_bf16_split` — the MPP one — on every decode cycle of every
model this fork serves. In the B1 profile at a 512-token context that is 0.671 ms
of a 99.77 ms cycle, **0.67%**; at 16384 tokens it is 1.566 ms of 109.13 ms,
**1.43%** -- the draft's 2048-token window limits it but does not make it
constant, and its share then *falls* with batch (1.09% at four lanes) because
everything else grows faster. For scale, the target's own attention on the
register tile is 10.3-11.3% of the cycle at 16K.

Two corrections to how I first framed this. Register pressure goes the *other*
way from what I assumed — 128 is half of 256, so a 128-dim tile is cheaper in
registers than the one that exists, not more likely to spill; and the item's
motivation was upstream's 128-dim sinks-and-window *target* class, which remains
a real risk for a model of that shape (there the fallback applies to full
attention, whose cost does grow with context) but is not what is installed here.

So: ~1.4% of a long-context one-lane cycle is the prize on this draft, not worth the
parameterization; the trigger to revisit is a *target* model with 128-dim heads
or a sliding window, and `dev/benchmarks/attention_sweep.mm` measures that in
minutes the day one appears. Window support is the more general gap of the two,
since plenty of 256-dim-capable architectures window.

**6. Re-derive the span chain.** Worth 2.4-2.8% at one lane, 0.7-1.6% at two to
four. Blocked on the chain restructure, not on measurement.

**7. The 64-row staged prefill tile is measured here, and it loses.** First
stated below as "the benchmark never offers it", which was wrong: the sweep
already does (`dev/benchmarks/gguf_projection_benchmark.mm`, "Chunks of up to
`kMaximumStagedPrefillRows` rows also on the staged 64-row tile, at every
split"), `stagedTileRows(64)` is 64, and the Apple7/8 MMA kernels carry a 64-row
instance (`GGUF_DECODE_MMA_ROWS64`, added by this port). So the harness's
`prefill-tiles-q4k-17408x5120` rows at R33 and R64 *are* that comparison: the
policy's `prefill128` takes 1.700 and 1.703 ms, the 64-row staged tile 4.71 to
5.17 ms at every split -- **2.8x slower** -- and 33 and 64 rows costing the same
is the 128-row tile padding both to one height.

Which makes the interesting half of the question the other way round. Upstream's
comment says the 64-row staged tile beats the prefill tile by 6-43% on Apple10,
and the policy here keeps Apple7/8 on the prefill tile
(`LinearGguf.cpp`: staged only to `kMaximumDecodeTileRows` unless
`family_ == Apple10`); that division now has a number behind it on this hardware
rather than being an unexamined inheritance. Nothing further to measure at 33 or
64 rows; if the Ultra's baseline sweep (`#2`) shows the gap closing at 64 cores,
that is worth reopening, and it will show up by itself because the sweep covers
the same candidates.

**8. Retune the display-facing bounds per machine.**
`kDefaultPrefillCommandMilliseconds` (2000 ms, `runtime/engine/Scheduler.hpp`)
and `kPrefillAttentionPieceWork` (2^25 row-history pairs) were picked so a
Max-class GPU would not abort a long prefill. A 76-core Ultra and an 8-core base
M1 want different numbers, and the wrong one costs throughput (too small) or a
wedged GPU (too large).


## What the other machines are for

Nothing here is a mystery that only new hardware can explain; each machine
closes a specific numbered item, and the harness records the family and core
count with every row, so results from different machines never mix silently.

**An M1 Ultra 128 GB would be the most valuable machine by far, because every
number in this file comes from family 8: nothing since the 1.5 merge has run on
family 7 itself.** Every kernel name, tile and
fallback gated at `GpuFamilyClass::Apple7` — the MMA expert tiles, the `_sgf`
register attention, the four-lane staged rule, the split tiers — has only been
executed on family 8. That is an untested half of the port, not an unoptimized
one, and it is worth doing before any tuning:

```sh
make -j8 && make test-engine-cpu && dev/tests/run-metal.sh '.'   # all 40, ~1-2 h at 64 cores
dev/benchmarks/apple78/run.py --label m1ultra-baseline --build   # the projection sweeps
dev/benchmarks/apple78/ane.py sweep --root "$MLX27B" --share 0 --share 0.25 --calibrated
```

Then, in order of what they would settle: whether the register attention tile's
numerics hold on family 7 hardware (it was written for 7 and 8 and measured on
neither); whether `_s2` MoE MMA tiles and the four-lane affine rule win at 64
cores as they do at 30; and what the ANE calibration costs and chooses when the
GPU is a generation older and the memory is 128 GB, which is the first
configuration where a 35B MoE, a 128K context and streamed experts all coexist
(`--expert-memory`, unmeasured anywhere).

**The M2 Ultra (family 8, up to 76 GPU cores, 64 GB)** closes item 1 and item
8, the two that are fits over a range rather than correctness questions:

- Item 1, the split tiers: `dev/benchmarks/apple78/run.py --label m2ultra-tiers
  --build`, then `compare.py baseline m2ultra-tiers --policy` shows how far the
  staged tier `kMmaTiers` (24 threadgroups per core, 512-input floor) and
  `kRegisterTiers` rule, fitted at 30 and 32 cores, drifts at 64 and 76. A 2%
  mean at one end is nothing; 15% is a law.
  Sweeping splits explicitly (`benchmark-gguf-projection`, splits 1 to 16 over
  the 27B and 35B shapes) gives the occupancy curve the Apple10 law was built
  from, which is what would replace the tiers.
- Item 8, the display-facing bounds: `kDefaultPrefillCommandMilliseconds`
  (2000 ms) and `kPrefillAttentionPieceWork` (2^25 row-history pairs) are one
  number for every GPU. At 64 cores a 2-second bound is twice the work per
  command it is at 30; at 8 cores it is far less. Measure with
  `backend-benchmark --scenario context` and `--prefill-bound-ms`, and watch for
  the abort that bound exists to prevent.
- The MoE spill path at 64 cores with 64 GB, where a 35B MoE's 17.3 GB of
  experts and a long context can be resident together — the counterfactual for
  the streaming finding in Taken, which was measured on 32 GB where it cannot.

**A base M1 or M2 (7 to 10 cores, 8 to 16 GB)** is the other end of item 1 and
the only place the fork's most likely user lives. It is where `--expert-memory`
spilling is not an experiment but the default configuration, where a 27B at
four lanes may prefer staging for reasons of memory rather than arithmetic, and
where the 832-thread threadgroup cap and the register tile's residency are
closest to biting. Same commands as the Ultra, plus `serve --expert-memory` at
the sizes a 16 GB machine can actually hold.
## Things deliberately not to chase

- Making the MPP tensor path fast by trying harder. The `_sgf`/`_mma` kernels
  exist because that path is structurally slow here; the remaining gains are in
  scheduling and operand width, not in rescuing it.
- Bitwise parity with upstream's outputs. The widened products differ in
  accumulation order only; tests hold fp64-bounded agreement, and greedy output
  is checked in the real model instead.
- Any policy number that has not been measured on a machine of this class,
  including the ones in this file.
