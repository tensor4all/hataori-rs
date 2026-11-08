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
for tool in sh python3 sleep kill ps awk mktemp rm cat ls grep dirname chmod tr; do
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


# P1, third reproduction: the parent ignores SIGTERM and its descendant does
# not. The parent dies on the TERM, the descendant is reparented to PID 1, and
# the escalation must still take it (the earlier fixtures tested the two cases
# separately and missed this combination).
pid_file=$tmp_dir/combining-parent
started=$SECONDS
status=0
bounded 1 sh -c "sh -c 'trap \"\" TERM; exec sleep 30' & echo \$! >$pid_file; wait" || status=$?
elapsed=$((SECONDS - started))
((status != 0)) || fail 'a command whose parent ignores SIGTERM reported success'
((elapsed <= 1 + bounded_grace + 2)) ||
    fail "the deadline was not enforced: ${elapsed}s for a 1s bound"
[[ -s $pid_file ]] || fail 'the fixture did not record its descendant'
descendant=$(cat "$pid_file")
sleep 1
if kill -0 "$descendant" 2>/dev/null; then
    fail "the escalation left descendant $descendant alive"
fi
printf '== ignoring parent with a descendant: %ss, status %s\n' "$elapsed" "$status"

# P1, fourth reproduction: a command that traps TERM and exits zero must still
# be reported as a timeout. An arm that writes its output and then hangs could
# otherwise satisfy both the status and the file-existence checks.
status=0
bounded 1 python3 -c 'import signal, sys, time; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(30)' ||
    status=$?
((status != 0)) || fail 'a command that exits zero on SIGTERM reported success'
printf '== zero exit after SIGTERM: status %s\n' "$status"


# P1, fifth reproduction: a process started *inside* the TERM handler must not
# outlive the arm either. It is in no snapshot taken before the signal; the
# process group is what catches it.
pid_file=$tmp_dir/escaped
status=0
bounded 1 python3 -c '
import signal, subprocess, sys, time
def stop(*_):
    child = subprocess.Popen(["sleep", "30"])
    with open(sys.argv[1], "w") as f:
        f.write(str(child.pid))
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
time.sleep(30)
' "$pid_file" || status=$?
[[ -s $pid_file ]] || fail 'the fixture did not record its escaped process'
escaped=$(cat "$pid_file")
sleep 1
if kill -0 "$escaped" 2>/dev/null; then
    kill -KILL "$escaped" 2>/dev/null || true
    fail "a process started in the TERM handler survived ($escaped)"
fi
printf '== process created during TERM handling: status %s\n' "$status"

# P1, sixth reproduction: the command exits but a background child keeps the
# function's stdout open, so the caller's command substitution blocks for that
# child's whole lifetime even though nothing is left to bound. The group is
# killed when the command returns, so the substitution returns with the bound.
leak=$tmp_dir/stdout-holder
started=$SECONDS
output=$(bounded 1 sh -c "sleep 30 & echo \$! >$leak; echo banner")
status=$?
elapsed=$((SECONDS - started))
printf '== background child holding stdout: %ss, status %s, output %s\n' \
    "$elapsed" "$status" "$output"
[[ $output == banner ]] || fail "the command output was lost: '$output'"
((status == 0)) || fail "a command that succeeded reported $status"
((elapsed <= 1 + bounded_grace)) ||
    fail "the caller waited for a background child: ${elapsed}s for a 1s bound"
[[ -s $leak ]] || fail 'the stdout-holder fixture did not record its child'
holder=$(cat "$leak")
if kill -0 "$holder" 2>/dev/null; then
    kill -KILL "$holder" 2>/dev/null || true
    fail "the background child $holder outlived the bound"
fi



# P1, eighth reproduction: the escaped descendant starts a *new process group*
# and its parent exits at once, so neither a group kill nor a tree walk from the
# (dead, and therefore reparented) child can find it. The session does: `setpgrp`
# changes the group, not the session.
started=$SECONDS
output=$(bounded 1 python3 -c '
import os, subprocess
child = subprocess.Popen(["sleep", "30"], preexec_fn=os.setpgrp)
print(child.pid)
')
status=$?
elapsed=$((SECONDS - started))
printf '== descendant in a new process group, parent exits: %ss, status %s\n' \
    "$elapsed" "$status"
((status == 0)) || fail "a command that succeeded reported $status"
((elapsed <= 1 + bounded_grace)) ||
    fail "the caller waited for a new-group descendant: ${elapsed}s for a 1s bound"
new_group=$(printf '%s' "$output" | tr -d '[:space:]')
[[ $new_group =~ ^[0-9]+$ ]] || fail "the fixture did not report its descendant: '$output'"
if kill -0 "$new_group" 2>/dev/null; then
    kill -KILL "$new_group" 2>/dev/null || true
    fail "a descendant in a new process group outlived the bound ($new_group)"
fi

# P1, ninth reproduction: the same escape from inside the TERM handler, which no
# snapshot taken before the signal can contain.
pid_file=$tmp_dir/termed-new-group
started=$SECONDS
status=0
bounded 1 python3 -c '
import os, signal, subprocess, sys, time
def stop(*_):
    child = subprocess.Popen(["sleep", "30"], preexec_fn=os.setpgrp)
    with open(sys.argv[1], "w") as handle:
        handle.write(str(child.pid))
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
time.sleep(60)
' "$pid_file" || status=$?
elapsed=$((SECONDS - started))
printf '== new process group started inside the TERM handler: %ss, status %s\n' \
    "$elapsed" "$status"
((status == 124)) || fail "an expired deadline reported $status instead of 124"
((elapsed <= 1 + bounded_grace + 2)) ||
    fail "the deadline was not enforced: ${elapsed}s for a 1s bound"
[[ -s $pid_file ]] || fail 'the fixture did not record its descendant'
escaped_group=$(cat "$pid_file")
if kill -0 "$escaped_group" 2>/dev/null; then
    kill -KILL "$escaped_group" 2>/dev/null || true
    fail "a new-group descendant started by the TERM handler survived ($escaped_group)"
fi

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

# P1, seventh reproduction: the same leak behind a launcher banner. A wrapper
# that exits after starting a background process makes the whole detection slow
# unless the group is cleaned up when the query returns.
noisy=$tmp_dir/noisy-launcher
cat >"$noisy" <<NOISY
#!/bin/sh
if [ "\$1" = --version ]; then
    sleep 15 &
    echo "site MPI wrapper 1.0"
    exit 0
fi
exit 0
NOISY
chmod +x "$noisy"
started=$SECONDS
mpi_launcher_supports_oversubscribe "$noisy" || true
elapsed=$((SECONDS - started))
printf '== launcher that leaves a background child: %ss\n' "$elapsed"
((elapsed <= 10 + bounded_grace)) ||
    fail "detection waited for the launcher's background child: ${elapsed}s"

# P2, second reproduction: a launcher whose --version never returns must not
# block detection; the banner query carries its own deadline.
hanging=$tmp_dir/hanging-launcher
cat >"$hanging" <<HANGING
#!/bin/sh
if [ "\$1" = --version ]; then exec sleep 300; fi
exit 0
HANGING
chmod +x "$hanging"
started=$SECONDS
mpi_launcher_supports_oversubscribe "$hanging" || true
elapsed=$((SECONDS - started))
((elapsed <= 10 + bounded_grace + 2)) || fail "the banner query was not bounded: ${elapsed}s"
printf '== hanging launcher banner: %ss\n' "$elapsed"
if ! mpi_launcher_supports_oversubscribe "$wrapper"; then
    fail 'an unrecognized banner was reported as unsupported without probing the launcher'
fi

printf 'bounded-run checks passed (deadline escalation, descendant cleanup, portable probe)\n'
