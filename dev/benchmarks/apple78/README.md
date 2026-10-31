# Apple7/8 benchmark harness

Run the same measurements the same way every time, so a change to a kernel
policy is argued with numbers instead of memory. The plan those numbers feed
lives in [docs/apple78-perf.md](../../../docs/apple78-perf.md).

## Use

```sh
make build/engine-tests/gguf-projection-benchmark build/splash.metallib   # what the sweeps call
dev/benchmarks/apple78/run.py --label baseline                            # record it
# ... make one change, rebuild ...
dev/benchmarks/apple78/run.py --label span-chain
dev/benchmarks/apple78/compare.py dev/benchmarks/apple78/results/<host>-baseline.jsonl \
                                  dev/benchmarks/apple78/results/<host>-span-chain.jsonl --chosen
```

Each run appends to `results/<host>-<label>.jsonl`: one JSON object per measured
row, with the shape, the case (lane count or prefill chunk), the candidate tile
and split, the median milliseconds, whether the policy chose it, the GPU family
and core count, the build's `git` revision and whether the tree was dirty.
Nothing is committed from `results/`: it belongs to whoever ran it.

Useful flags:

| Flag | |
| --- | --- |
| `--only REGEX` / `--skip REGEX` | a subset of the sweep names |
| `--resume` | skip the sweeps already in the output file, so an interrupted run finishes where it stopped |
| `--dry-run` | print the commands, run nothing |
| `--timeout SECONDS` | per sweep (default 3600) |
| `compare.py --chosen` | only the rows the policy actually picks, which is what a user feels |
| `compare.py --tol 3` | what counts as a change rather than noise (default 2%) |

## Leaving it running

The point of `--resume` is that a run can be interrupted and continued, so a
long pass goes in a detached tmux session rather than a tool call:

```sh
tmux new-session -d -s pi-bench 'dev/benchmarks/apple78/run.py --label baseline \
  > /tmp/pi-bench.log 2>&1; echo EXIT=$? >> /tmp/pi-bench.log'
tail -5 /tmp/pi-bench.log
```

Two rules make the numbers worth having. Nothing else may use the GPU while a
sweep runs — no served model, no Metal test suite, no other benchmark — and a
candidate is compared against a baseline from the same machine and the same
rounds, with only the one change between them.

## Sweeps

`sweeps.txt` holds `name<TAB>command` lines, run from the repository root. Add
one when a policy rule starts claiming something that a measurement can refute:
a new tier, a new tile, a new threshold. Keep the shape's name in the sweep name
(`dense-q4k-17408x5120`) so a reader can tell what a regression is about.

`run.py` parses `gguf_projection_benchmark`'s output. The other benchmarks
(`attention-sweep`, `gguf-moe-benchmark`, `decode-profile`) print in shapes the
parser does not know yet, so they are not in `sweeps.txt`: wiring one in means
adding a heading and a row pattern to `run.py` rather than changing the
benchmark, which upstream owns.
