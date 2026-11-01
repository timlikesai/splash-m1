#!/bin/bash
# Run some of the Metal test suite, by name, without running all of it.
#
#   dev/tests/run-metal.sh gguf-dequant            # one test
#   dev/tests/run-metal.sh 'ane-ffn|ane-program'   # several, by regex
#   dev/tests/run-metal.sh --list                  # what the suite would run
#
# The suite itself is `make test-engine-metal`, a fixed recipe in dev/native.mk
# that no variable filters. This reads the commands that recipe would run,
# builds what each one needs, and runs the ones whose name matches, so a change
# can be checked against the two tests that cover it instead of against an hour
# of GPU work. It is also what CI runs on a hosted runner, whose hardware is
# Apple7/8 (docs/apple78-port.md, Verification).
set -u -o pipefail
cd "$(dirname "$0")/../.." || exit 1

if [ "${1:-}" = "--list" ]; then
  make -n test-engine-metal 2>/dev/null | sed '/^$/d' | awk '
    { line = $0; while (line ~ /\\$/) { sub(/\\$/, "", line); if ((getline nxt) > 0) line = line " " nxt; else break } print line }' |
    grep -E "engine-tests/" | grep -v "clang\|build_config\|^mkdir\|^make"
  exit 0
fi

[ $# -ge 1 ] || { sed -n '2,14p' "$0"; exit 2; }
pattern="$1"
# SKIP_METAL drops tests the machine cannot serve: a hosted runner has no ANE,
# and ane-ffn's kernels then fail to compile at startup, not at runtime.
skip="${SKIP_METAL:-}"
ran=0
failed=0
device=""
if xcrun --sdk macosx cc -framework Metal -framework Foundation -o /tmp/splash-device-info \
     dev/tests/device-info.m 2>/tmp/splash-device-info.err; then
  device="$(/tmp/splash-device-info)"
  echo "== $device"
else
  echo "== device-info did not build: $(tail -1 /tmp/splash-device-info.err)"
fi
# The runtime's own support minimum is Apple GPU family 7 (DeviceCapabilities'
# kMinimumAppleGpuFamily: "M1 or later"), and the port's kernels follow it -- the
# simdgroup MMA and register tiles are written for that hardware and will not even
# compile below it. GitHub's macos-26 pool has handed out at least two GPUs: an
# M1-class one that runs this suite green, and an "Apple Paravirtual" one that
# claims nothing above Apple5 and fails gguf-projection with
# "unable to create Metal pipeline gguf_decode_mma_fused_m64: Compilation failed".
# A test run on a device the product does not support measures nothing, so say so
# and stop, rather than reporting the branch as broken on a runner it happens to
# have drawn. SKIP_DEVICE=0 overrides, for a machine where the probe is wrong.
# Skip only on a device the probe *read* and found below the minimum. If the probe
# did not build, or named no family, run the tests anyway: a silent green with zero
# coverage is worse than a red that can be read.
highest=0
claims=0
for claim in $device; do
  case "$claim" in
    Apple*) n=${claim#Apple}
      case "$n" in (*[!0-9]*|'') ;; (*) claims=$((claims + 1)); [ "$n" -gt "$highest" ] && highest=$n ;; esac ;;
  esac
done
if [ "$claims" -gt 0 ] && [ "$highest" -lt 7 ] && [ "${SKIP_DEVICE:-1}" = "1" ]; then
  echo "== SKIPPED: this device claims no Apple GPU family 7 or newer (highest: $highest),"
  echo "   below the runtime's minimum (M1 or later). The Metal tests are not applicable here;"
  echo "   they are not a branch regression. Run them on M1 or later, or set SKIP_DEVICE=0."
  exit 0
fi
while IFS= read -r command; do
  name=$(printf '%s' "$command" | grep -oE "engine-tests/[a-z0-9._-]+" | head -1 | sed 's|.*/||; s|\.[a-z]*$||')
  printf '%s' "$name" | grep -qE "$pattern" || continue
  [ -n "$skip" ] && printf '%s' "$name" | grep -qE "$skip" && { echo "== $name: skipped (SKIP_METAL)"; continue; }
  # Build what this command runs and links, then run it as the recipe does.
  targets=$(printf '%s' "$command" | grep -oE "build/(engine-tests/[a-zA-Z0-9._-]+|splash\.metallib)" |
    grep -v '\.config$' | sort -u | tr '\n' ' ')
  echo "== $name: make $targets"
  make -j4 $targets > /tmp/run-metal-build.log 2>&1 || {
    echo "   BUILD FAILED"; tail -5 /tmp/run-metal-build.log; failed=$((failed + 1)); continue; }
  echo "== $name"
  if eval "$command"; then echo "   ok"; else echo "   FAILED"; failed=$((failed + 1)); fi
  ran=$((ran + 1))
done < <(make -n test-engine-metal 2>/dev/null | sed '/^$/d' | awk '
  { line = $0; while (line ~ /\\$/) { sub(/\\$/, "", line); if ((getline nxt) > 0) line = line " " nxt; else break } print line }' |
  grep -E "engine-tests/" | grep -v "clang\|build_config\|^mkdir\|^make")

echo "== $ran test(s) matched, $failed failed"
[ "$ran" -gt 0 ] || echo "   nothing matched '$pattern'; try --list"
exit $((failed > 0 ? 1 : 0))
