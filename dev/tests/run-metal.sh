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
if xcrun --sdk macosx cc -framework Metal -framework Foundation -o /tmp/splash-device-info \
     dev/tests/device-info.m 2>/tmp/splash-device-info.err; then
  echo "== $(/tmp/splash-device-info)"
else
  echo "== device-info did not build: $(tail -1 /tmp/splash-device-info.err)"
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
