#!/usr/bin/env bash
# Regression tests for the #43 review findings on the portable pieces of the
# acceptance harness:
#
#   P1  the fallback watchdog must enforce its deadline even when the command
#       ignores SIGTERM, and must take the command's descendants with it;
#   P2  the launcher probe must not report a supported launcher as unsupported
#       on a platform without `setsid`;
#   P3  the replay of the spooled output must not hold `bounded` past its bound
#       when the caller stops reading — not on stdout, not on stderr, not through
#       the truncation notice, and not through the replay's own `sleep`.
#
# All of them run against the *fallback* paths. The tools that normally make the
# fallback unnecessary (`setsid`, GNU `timeout`) are hidden with a PATH shim, so
# this also runs on Linux CI instead of only on macOS.
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
undrained_pid=
trap 'stop_fixture "${undrained_pid:-}"; rm -rf -- "$tmp_dir"' EXIT

fail() {
    printf 'check-bounded-run: %s\n' "$*" >&2
    exit 1
}

stop_fixture() {
    local pid=${1:-}
    [[ -n $pid ]] || return 0
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# Waits for a background fixture to report FILE. A fixture that exits without
# reporting, or that outlives LIMIT seconds, is the failure: a regression must
# fail this script instead of hanging it.
await_fixture() {
    local limit=$1 file=$2 pid=$3 i
    for ((i = 0; i < 10 * limit; i++)); do
        [[ -s $file ]] && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 0.1
    done
    return 1
}

# A read end nobody drains: `sleep` opens the fifo and never reads it, so a
# writer blocks once the pipe buffer is full.
undrained() {
    mkfifo "$1"
    sleep 300 <"$1" &
    undrained_pid=$!
}

# A PATH with the tools these tests need but without `setsid`/`timeout`, which is
# the macOS situation.
shim=$tmp_dir/bin
mkdir -p "$shim"
for tool in sh python3 sleep kill ps awk mktemp rm cat ls grep dirname chmod tr pgrep mkdir env head wc mkfifo; do
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
status=0
output=$(bounded 1 sh -c "sleep 30 & echo \$! >$leak; echo banner") || status=$?
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

# P3 (issue 43): the caller stops reading. The child writes ~4 MiB, the reader
# takes 1 KiB and then holds the pipe open without reading, so the replay has
# nowhere to go. A synchronous `cat` returned only when the reader resumed
# (measured at ee71e40: 6.2 s for a reader away 6 s, 33 s for this fixture's
# holder); the replay carries its own bound now, so the run ends inside
# bound + grace and the rest is dropped with a notice.
ret_file=$tmp_dir/stopped-reader
started=$SECONDS
{
    status=0
    bounded 1 head -c 4194304 /dev/zero || status=$?
    printf '%s %s\n' "$((SECONDS - started))" "$status" >"$ret_file"
} | {
    head -c 1024 >/dev/null
    # Hold the read end without draining it; give up long after the bound so a
    # broken run fails on the elapsed check instead of hanging this test.
    for ((i = 0; i < 300; i++)); do
        [[ -s $ret_file ]] && break
        sleep 0.1
    done
}
[[ -s $ret_file ]] || fail 'the stopped-reader fixture never saw `bounded` return'
read -r elapsed status <"$ret_file"
printf '== caller that stops reading: %ss, status %s (1s bound, 4 MiB child)\n' \
    "$elapsed" "$status"
((status == 0)) || fail "a command that succeeded reported $status"
((elapsed <= 1 + bounded_grace)) ||
    fail "the replay outlived the bound: ${elapsed}s for a 1s bound"

# P3 (issue 43, review finding 1): only *stderr* is backed up. The caller drains
# its stdout and leaves its stderr on a pipe nobody reads, so the stderr replay is
# cut at its deadline. The truncation notice used to be a synchronous `printf` on
# that same stderr: at 3a5e1e2 `bounded 1` was still blocked after 6 s.
undrained "$tmp_dir/stderr-sink"
err_sink=$undrained_pid
ret_file=$tmp_dir/stderr-backpressure
rm -f "$ret_file"
started=$SECONDS
(
    status=0
    bounded 1 python3 -c 'import sys; sys.stderr.write("x" * 4194304)' \
        >/dev/null 2>"$tmp_dir/stderr-sink" || status=$?
    printf '%s\n' "$status" >"$ret_file"
) &
fixture=$!
if ! await_fixture 8 "$ret_file" "$fixture"; then
    stop_fixture "$fixture"
    stop_fixture "$err_sink"
    fail 'stderr backpressure blocked `bounded 1`: the truncation notice is not bounded'
fi
elapsed=$((SECONDS - started))
status=$(cat "$ret_file")
stop_fixture "$fixture"
stop_fixture "$err_sink"
printf '== caller that stops reading stderr: %ss, status %s (1s bound, 4 MiB child)\n' \
    "$elapsed" "$status"
((status == 0)) || fail "a command that succeeded reported $status"
((elapsed <= 3 * bounded_replay_secs + 1)) ||
    fail "stderr backpressure held the caller: ${elapsed}s for a $((3 * bounded_replay_secs))s replay budget"

# P3 (issue 43, review finding 2): an ordinary `set -e` caller — no
# `|| status=$?`, the shape `run_direct()` has — with its stdout on a pipe nobody
# reads. The replay is cut at its deadline, so its killer has already fired and
# exited; the unguarded `kill` of that dead killer aborted the caller (measured at
# 3a5e1e2: exit 1 after ~1.28 s, never reaching the next statement).
undrained "$tmp_dir/stdout-sink"
out_sink=$undrained_pid
ret_file=$tmp_dir/set-e-caller
rm -f "$ret_file"
started=$SECONDS
(
    set -e
    bounded 1 head -c 4194304 /dev/zero >"$tmp_dir/stdout-sink"
    printf 'reached\n' >"$ret_file"
) 2>/dev/null &
fixture=$!
if ! await_fixture 8 "$ret_file" "$fixture"; then
    status=0
    if kill -0 "$fixture" 2>/dev/null; then
        stop_fixture "$fixture"
        stop_fixture "$out_sink"
        fail "stdout backpressure held a plain \`set -e\` caller past $((SECONDS - started))s"
    fi
    wait "$fixture" 2>/dev/null || status=$?
    stop_fixture "$out_sink"
    fail "a plain \`set -e\` caller aborted with status $status although \`bounded\` succeeded"
fi
elapsed=$((SECONDS - started))
stop_fixture "$fixture"
stop_fixture "$out_sink"
printf '== plain `set -e` caller, stdout not drained: %ss, reached its next statement\n' \
    "$elapsed"
((elapsed <= 3 * bounded_replay_secs + 1)) ||
    fail "the replay outlived its budget: ${elapsed}s"

# P3 (issue 43, review finding 3): the replay's own killer is what holds the
# caller. `bounded` cancels it once the replay is done, but its `sleep` child kept
# the inherited stdout, so a substitution waited for that sleep to expire
# (measured at 3a5e1e2: 5 s for a 5 s replay window, on a run that took 0 s).
saved_replay_secs=$bounded_replay_secs
bounded_replay_secs=5
started=$SECONDS
status=0
output=$(bounded 1 python3 -c 'import sys; sys.stdout.write("x" * 8388608)') || status=$?
elapsed=$((SECONDS - started))
bounded_replay_secs=$saved_replay_secs
printf '== command substitution with a truncation: %ss, status %s (5s replay window, %s bytes)\n' \
    "$elapsed" "$status" "${#output}"
((status == 0)) || fail "a command that succeeded reported $status"
((${#output} == bounded_replay_cap)) ||
    fail "the substitution did not receive the capped replay: ${#output} bytes"
((elapsed < 5)) ||
    fail "the substitution waited for the replay's killed-off \`sleep\`: ${elapsed}s for a 5s replay window"

# P1, eighth reproduction: the escaped descendant starts a *new process group*
# and its parent exits at once, so neither a group kill nor a tree walk from the
# (dead, and therefore reparented) child can find it. The session does: `setpgrp`
# changes the group, not the session.
started=$SECONDS
status=0
output=$(bounded 2 python3 -c '
import os, subprocess
child = subprocess.Popen(["sleep", "30"], preexec_fn=os.setpgrp)
print(child.pid)
') || status=$?
elapsed=$((SECONDS - started))
printf '== descendant in a new process group, parent exits: %ss, status %s\n' \
    "$elapsed" "$status"
((status == 0)) || fail "a command that succeeded reported $status"
((elapsed <= 2 + bounded_grace)) ||
    fail "the caller waited for a new-group descendant: ${elapsed}s for a 2s bound"
new_group=$(printf '%s' "$output" | tr -d '[:space:]')
[[ $new_group =~ ^[0-9]+$ ]] || fail "the fixture did not report its descendant: '$output'"
if kill -0 "$new_group" 2>/dev/null; then
    kill -KILL "$new_group" 2>/dev/null || true
    # Killing it is only guaranteed where the session can be scanned; see the
    # header of `bounded-run.sh`. Everywhere else the bound must still end on
    # time, which the assertions above cover.
    [[ ${HATAORI_BOUNDED_SESSION_SCAN:-none} != none ]] ||
        printf '== no session scan here (%s): new-group stray left running\n' \
            "${HATAORI_BOUNDED_SESSION_SCAN:-none}"
    [[ ${HATAORI_BOUNDED_SESSION_SCAN:-none} == none ]] ||
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
    [[ ${HATAORI_BOUNDED_SESSION_SCAN:-none} == none ]] ||
        fail "a new-group descendant started by the TERM handler survived ($escaped_group)"
fi

# The session scan is chosen by what actually works, not by whether `ps` accepts
# the keyword: macOS accepts `-o sess=` and prints 0, and a keyword probe then
# picked a scan that found nothing (the review measured the escape below at 30 s
# against a 1 s bound). This reruns the new-group escape through each backend and
# through a `ps` that emulates macOS.
escape_fixture=$tmp_dir/escape-fixture.sh
cat >"$escape_fixture" <<'FIXTURE'
set -u
source "$1/scripts/bounded-run.sh"
started=$SECONDS
status=0
output=$(bounded 2 python3 -c '
import os, subprocess
child = subprocess.Popen(["sleep", "30"], preexec_fn=os.setpgrp)
print(child.pid)
') || status=$?
elapsed=$((SECONDS - started))
printf 'backend=%s elapsed=%ss status=%s\n' \
    "$HATAORI_BOUNDED_SESSION_SCAN" "$elapsed" "$status"
# The bound and the caller's unblocking are platform independent; killing the
# escaped group is only promised where a session scan exists.
[[ $status == 0 ]] || exit 1
((elapsed <= 2 + bounded_grace)) || exit 1
if kill -0 "$output" 2>/dev/null; then
    kill -KILL "$output" 2>/dev/null
    [[ ${HATAORI_BOUNDED_SESSION_SCAN:-none} == none ]] || exit 1
fi
FIXTURE
shell=${BASH:-bash}

macos_ps=$tmp_dir/macos-ps
mkdir -p "$macos_ps"
real_ps=$(command -v ps)
cat >"$macos_ps/ps" <<MACOS
#!/bin/sh
case " \$* " in
  *sess*) "$real_ps" "\$@" 2>/dev/null | awk '{ if (NF >= 2) print \$1, 0; else print \$0 }' ;;
  *) exec "$real_ps" "\$@" ;;
esac
MACOS
chmod +x "$macos_ps/ps"

if ! line=$("$shell" "$escape_fixture" "$root_dir" 2>&1); then
    fail "the auto-selected session scan did not clean up the escape: $line"
fi
printf '== session scan, auto: %s\n' "$line"

# Does *this host* enumerate sessions with SCAN? macOS answers no to both (its
# `ps -o sess=` prints 0 for every process and its `pgrep` rejects `-s`), so a
# forced backend that cannot work is skipped rather than failed — the deadline
# assertions in the fixtures above are what the platform must satisfy, and the
# kill of an escaped group is only promised where a scan exists.
session_scan_works() {
    local scan=$1 sid
    sid=$(python3 -c 'import os; print(os.getsid(0))' 2>/dev/null) || return 1
    case $scan in
        ps)
            ps -A -o pid= -o sess= 2>/dev/null |
                awk -v pid="$$" -v sid="$sid" '$1 == pid && $2 == sid { found = 1 } END { exit(found ? 0 : 1) }'
            ;;
        pgrep)
            command -v pgrep >/dev/null 2>&1 || return 1
            pgrep -s "$sid" 2>/dev/null | awk -v pid="$$" '$1 == pid { found = 1 } END { exit(found ? 0 : 1) }'
            ;;
        *)
            return 1
            ;;
    esac
}

for scan in ps pgrep; do
    if ! session_scan_works "$scan"; then
        printf '== session scan, forced %s: skipped (this host cannot enumerate sessions with %s)\n' \
            "$scan" "$scan"
        continue
    fi
    if ! line=$(HATAORI_BOUNDED_SESSION_SCAN=$scan "$shell" "$escape_fixture" "$root_dir" 2>&1); then
        fail "the forced $scan scan did not clean up the escape: $line"
    fi
    [[ $line == "backend=$scan "* ]] || fail "the forced $scan scan was not used: $line"
    printf '== session scan, forced %s: %s\n' "$scan" "$line"
done

# macOS as measured by the review: `ps -o sess=` reports 0 and `pgrep` has no
# `-s`, so no session scan exists. The bound must still end and must not wait for
# the escaped group.
if ! line=$(HATAORI_BOUNDED_SESSION_SCAN=none "$shell" "$escape_fixture" "$root_dir" 2>&1); then
    fail "without a session scan the bound did not end on time: $line"
fi
[[ $line == "backend=none "* ]] || fail "the forced no-session run did not take effect: $line"
printf '== session scan, none (macOS): %s\n' "$line"

if ! command -v pgrep >/dev/null 2>&1; then
    printf '== session scan, macOS-like ps: skipped (no pgrep)\n'
else
    # Ask the probe itself, in a shell where `ps` reports 0 for the session id.
    probe_line=$(
        # `/usr/bin` is already on this PATH so a login/module hook does not
        # re-prepend it ahead of the fake.
        env PATH="$macos_ps:$PATH:/usr/bin" "$shell" -c \
            'source "$1/scripts/bounded-run.sh"
printf "ps=%s backend=%s\n" "$(command -v ps)" "$(bounded_detect_session_scan)"' \
            _ "$root_dir" 2>&1
    )
    if [[ $probe_line != "ps=$macos_ps/ps backend="* ]]; then
        printf '== session scan, macOS-like ps: skipped (ps not shadowed here: %s)\n' "$probe_line"
    else
        # A ps that reports 0 must fall through to `pgrep -s`; on a host whose
        # `pgrep` has no `-s` (macOS) neither scan exists, and `none` is the honest
        # answer.
        expected=pgrep
        session_scan_works pgrep || expected=none
        [[ $probe_line == *"backend=$expected" ]] ||
            fail "a ps that reports sess=0 gave $probe_line, expected backend=$expected"
        printf '== session scan, macOS-like ps: %s\n' "$probe_line"
    fi
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
