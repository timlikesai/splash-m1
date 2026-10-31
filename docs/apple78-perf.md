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
  10-core M2 and a 76-core M2 Ultra, and `tune_kernels` and
  `gguf-projection-benchmark` both print family and core count in their heading.
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
| Split count / threadgroups per core | fitted tiers of 24 threadgroups per core (staged) and 24 with a 256-input floor (register), from an M1 Max and an M2 Max | whether a fitted occupancy law, as Apple10 now has, beats a tier across 7 to 76 cores |
| Attention tile | the MPP tensor path holds most of a long-context prefill chunk or verify cycle, which is why the register tile exists; the end-to-end it enabled is measured (27B on an M1 Max: cold 8K prefill 135 tok/s, 1.5K append 121) | the 128-dimension, window and sinks geometries have no register kernels, so a model that uses them runs the slow path here |
| Prefill command length | bounded to about 5 s so macOS does not abort it | the right bound per machine class, and whether the attention piece-work bound (2^25 row-history pairs) is what actually protects the display |
| MoE experts | the routed passes' MPP `matmul2d` replaced by simdgroup MMA (`moe_expert_gguf_mma_m*`), which is what Apple7/8 run for a one-slice shared expert; a two-slice routing still lands on the MPP kernels, because the MMA ones were never compiled for it | the ANE's share of an MoE layer; the shared expert is dense-shaped and always visited, the routed ones are not |
| Coefficient chains | the span chain measured 2.4-2.8% at one lane | its form under the new per-coefficient-group chains, which is why it is off |

## Taken

- **A four-lane step of the 64-input-group affine weights stages on the MMA
  kernels** (d9d77ed0). Found by the harness's first baseline: at one to three
  lanes the register tile wins by 6-40%, at four it loses by 6.7-12.8%, and
  Q4_K/IQ4_XS keep the register tile at every width, so the rule is the
  coefficient's 64-input span rather than the row count. Three of 55 sweep
  cases moved, none regressed. The vocabulary projection — the single largest
  decode cost of a turn — gains 12.8%. End to end on the 27B-4bit itself
  (`backend-benchmark --scenario decode --ane-ffn-share 0`, three samples, both
  builds run twice interleaved from a separate worktree at e43c18e2): a batch of
  four decodes **5.7% faster** (347.5 → 327.7 ms per cycle) and its 64-token
  throughput 4.7% faster, while one, two and three lanes move by 0.1% or less
  and prefill not at all — which is the change's signature, since only
  four-lane plans were touched. Repeat runs agreed to 0.3%.

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

## Ranked opportunities

Ordered by expected value per unit of risk, not by size.

**1. Fit the split policy to core counts, not to two Max chips.** The rule is
`{24, 512}` and `{24, 256}` threadgroups-per-core tiers, fitted on a 32-core M1
Max and a 30-core M2 Max. Apple10 replaced exactly this kind of rule with a
measured occupancy law (`apple10Staged`: resident wave, DRAM-saturating count,
trailing-wave cost) fit to sweeps on the M5 Max, M5 Pro and M6. The same fit is available here: sweep
splits from 1 to 16 across the 27B and 35B shapes with
`benchmark-gguf-projection` on this M2 Max, and again on the M2 Ultra; the base
parts can be measured on hosted CI runners, which are this GPU class.
*Decides:* whether a law beats the tier, and by how much at each end of the
range.

**2. Try fp16 activations on the native half MMA path.** The single structural
idea. Apple7/8 do have fast 8x8 half×half simdgroup MMAs; what they lack is
bfloat. Today's tiles pair half weights with *fp32* activations because fp32 is
exact, which costs both ALU work and a table of twice the bytes. Pairing half
weights with half activations instead keeps every product exact only when the
activation survives half's 11-bit mantissa — so it is a numerics question, and
the repo already has the machinery to answer it per format: the fp64-referenced
slack bounds in `dev/tuning/LinearNumerics.hpp` and the per-format numerics
tests. *Decides:* per format, whether outputs stay inside the slack the tests
allow, and what the projections and end-to-end decode do about it.

**3. ~~Give the ANE a fair share on these GPUs.~~ Closed** — see Taken. The
calibration's own choice beats a fixed share by 7 points of prefill time, so
the remaining ANE question is not the share but the 75.7 s it costs on a new
macOS version, and whether a family-fitted starting guess would shorten the
search rather than replace it.

**4. MoE on the ANE, shared expert first.** The routed experts are the bulk of
an MoE layer but their work is gathered per row through selected-expert tables,
which is the opposite of the static shapes an ANE program wants. The *shared*
expert is dense-shaped, visited by every row, and already runs as its own pass.
That makes it the only MoE candidate worth a first measurement: take the MoE
benchmark's gate/up versus down split for a real layer, and ask what fraction
of a step's FFN time the shared expert is at the top-k the model uses. If it is
worth chasing, the design question is whether a fixed-row-count shared-expert
program can be handed to the ANE while the routed passes run on the GPU, which
is the shape of the dense split already in the tree. *Decides:* whether to
design further or to stop; it is explicitly a stretch goal.

**5. Compile the register attention tile for 128-dimension heads.** Upstream's
1.5 geometry table added a 128-dimension, sinks-and-window model class. The
register tile falls back to MPP there, which is correct and slow, so a new
model of that shape will look broken on old hardware. The tile is templated on
`HeadDim` now, but the register kernels here are not: `paged_attention_sgf.h`
reads `SPLASH_KV_HEAD_DIMENSION` as a build-time constant in its tile body
(line 150) and in its addressing (line 194), so this is first a
parameterization of `D` through `q8sgf::tile` and its register budget, then
instantiation, then numerics. Doubling `D` doubles what a tile holds in
registers, and a tile that spills is plausibly slower than the MPP path it
replaces, so the win is not assumed until measured at 128 on this hardware.

**6. Re-derive the span chain.** Worth 2.4-2.8% at one lane, 0.7-1.6% at two to
four. Blocked on the chain restructure, not on measurement.

**7. Check the 64-row tile's policy here.** It exists on this branch because
upstream routes 33-to-64-row prefill chunks through it; the instances compile,
but the choice was measured on M5 hardware. `benchmark-gguf-projection` with
`prefill=33,64` answers it directly.

**8. Retune the display-facing bounds per machine.** `kIsolatedPrefillMilliseconds`
(5 s) and `kPrefillAttentionPieceWork` (2^25 row-history pairs) were picked so a
Max-class GPU would not abort a long prefill. A 76-core Ultra and an 8-core base
M1 want different numbers, and the wrong one costs throughput (too small) or a
wedged GPU (too large).

**9. Compile the two-slice MoE MMA kernels.** Listed last because it was found
last, not because it ranks there: it is the cheapest item on this list.
`moeGgufTile` sends Apple7/8 to the MMA expert tile only when the shared expert
is one slice, and to `MoeGgufTile::Staged` otherwise — which on these
generations is the MPP `matmul2d` this file's first rule says to avoid. Qwen3
Next-class routings, two-slice shared experts, therefore decode on the slow
path, and `gguf-moe-sigmoid` (which exercises `_s2`) covers the routing kernels
but not expert tiles that do not exist. The work is the `_s2` instantiations of
`moe_expert_gguf_mma_m*` and their numerics in the same test, then flipping the
`sharedSlices == 1` condition away. *Decides:* whether a two-slice MoE on an M1
or M2 runs its experts on tensor hardware at all.

## Things deliberately not to chase

- Making the MPP tensor path fast by trying harder. The `_sgf`/`_mma` kernels
  exist because that path is structurally slow here; the remaining gains are in
  scheduling and operand width, not in rescuing it.
- Bitwise parity with upstream's outputs. The widened products differ in
  accumulation order only; tests hold fp64-bounded agreement, and greedy output
  is checked in the real model instead.
- Any policy number that has not been measured on a machine of this class,
  including the ones in this file.
