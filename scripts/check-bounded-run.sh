#!/usr/bin/env bash
# Regression tests for the #43 review findings on the portable pieces of the
# acceptance harness:
#
#   P1  the fallback watchdog must enforce its deadline even when the command
#       ignores SIGTERM, and must take the command's descendants with it;
#   P2  the launcher probe must not report a supported launcher as unsupported
#       on a platform without `setsid`.
#
# Both run against the *fallback* paths. The tools that normally make the
# fallback unnecessary (`setsid`, GNU `timeout`) are hidden with a PATH shim, so
# this also runs on Linux CI instead of only on macOS.
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT

fail() {
    printf 'check-bounded-run: %s\n' "$*" >&2
    exit 1
}

# A PATH with the tools these tests need but without `setsid`/`timeout`, which is
# the macOS situation.
shim=$tmp_dir/bin
mkdir -p "$shim"
for tool in sh python3 sleep kill ps awk mktemp rm cat ls grep dirname chmod; do
    path=$(command -v "$tool") || fail "missing tool for the shim: $tool"
    ln -s "$path" "$shim/$tool"
done
export PATH=$shim

source "$root_dir/scripts/bounded-run.sh"
command -v setsid >/dev/null 2>&1 && fail 'the shim still exposes setsid'
command -v timeout >/dev/null 2>&1 && fail 'the shim still exposes timeout'

# P1, first reproduction: a command that ignores SIGTERM must still be gone at
# the deadline, and the arm must report a non-zero status.
started=$SECONDS
status=0
bounded 1 python3 -c \
    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)' ||
    status=$?
elapsed=$((SECONDS - started))
((status != 0)) || fail 'a command that ignores SIGTERM reported success'
((elapsed <= 1 + bounded_grace + 2)) ||
    fail "the deadline was not enforced: ${elapsed}s for a 1s bound"
printf '== SIGTERM-ignoring command: %ss, status %s (1s bound, %ss grace)\n' \
    "$elapsed" "$status" "$bounded_grace"

# P1, second reproduction: the command exits, its descendant must not survive.
pid_file=$tmp_dir/grandchild
started=$SECONDS
status=0
bounded 1 sh -c "sleep 30 & echo \$! >$pid_file; wait" || status=$?
elapsed=$((SECONDS - started))
((status != 0)) || fail 'a command with a surviving descendant reported success'
((elapsed <= 1 + bounded_grace + 2)) ||
    fail "the deadline was not enforced: ${elapsed}s for a 1s bound"
printf '== command with a descendant: %ss, status %s (1s bound, %ss grace)\n' \
    "$elapsed" "$status" "$bounded_grace"
[[ -s $pid_file ]] || fail 'the fixture did not record its grandchild'
grandchild=$(cat "$pid_file")
sleep 1
if kill -0 "$grandchild" 2>/dev/null; then
    fail "the fallback left descendant $grandchild alive"
fi

# A command that finishes inside the deadline keeps its own status and is not
# reported as a timeout.
status=0
bounded 5 sh -c 'exit 7' || status=$?
((status == 7)) || fail "a completed command reported $status instead of its own status"

# P2: an unrecognized banner must be probed, not written off, on a platform
# without `setsid`. The wrapper below is what a site-installed launcher looks
# like: an opaque `--version` that forwards everything else to the real one.
wrapper=$tmp_dir/site-launcher
cat >"$wrapper" <<WRAPPER
#!/bin/sh
if [ "\$1" = --version ]; then
    echo "site MPI wrapper 1.0"
    exit 0
fi
exit 0
WRAPPER
chmod +x "$wrapper"

source "$root_dir/scripts/mpi-launcher.sh"
if ! mpi_launcher_supports_oversubscribe "$wrapper"; then
    fail 'an unrecognized banner was reported as unsupported without probing the launcher'
fi

printf 'bounded-run checks passed (deadline escalation, descendant cleanup, portable probe)\n'
