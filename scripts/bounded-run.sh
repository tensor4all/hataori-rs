#!/usr/bin/env bash
# `bounded SECONDS COMMAND...`: run COMMAND and return its status, terminating
# it once SECONDS have elapsed. Sourced, never executed:
#
#   source "$root_dir/scripts/bounded-run.sh"
#
# An arm without a bound holds the whole job to its job-level timeout (issue 41: a
# rank blocked in the collective). This is that bound, and it is one implementation
# on every platform: `setsid timeout --kill-after` cannot provide it, because it
# returns as soon as the direct child exits, so the KILL it schedules never
# reaches a descendant that ignores SIGTERM (measured: `setsid timeout
# --signal=TERM --kill-after=2s 1s sh -c 'sh -c "trap \"\" TERM; exec sleep 30" &
# wait'` returns 124 after 1 s and leaves the `sleep` under PPID 1).
#
# - the command runs in its own **session** (`python3` and `os.setsid`, or job
#   control as the fallback), and the deadline signals every process in that
#   session. A descendant that only changes its process group — `setpgrp`/
#   `setpgid`, or a TERM handler that starts a new group — still inherits the
#   session, and a descendant that is reparented to PID 1 keeps it, so neither
#   escape hides it. The PID tree and the process group are signalled as well, for
#   the fallback without a session and for anything that left it. Without
#   `python3` there is no session to start, so the fallback (job control) reaches
#   only the group and the tree: a descendant that calls `setpgrp` can escape it;
# - the descendant list is taken once, before the first signal: a TERM can
#   reparent a grandchild to PID 1, and a later tree traversal would miss it;
# - a command that ignores SIGTERM is escalated to SIGKILL after
#   `bounded_grace` — the same TERM-then-KILL contract as
#   `timeout --kill-after` — so the arm cannot outlive the deadline it was
#   supposed to enforce;
# - the deadline is recorded, so a command that traps TERM and exits 0 is
#   reported as a timeout (124, as `timeout` reports it) instead of a success;
# - whatever is left in the session when the command returns is killed too: a
#   background child that outlives the command still holds this function's stdout,
#   and the caller's command substitution then waits for it long after the bound
#   (`output=$(bounded 1 sh -c "sleep 4 & echo banner")` took 4 s and reported
#   success).

if [[ -n ${HATAORI_BOUNDED_RUN_SH:-} ]]; then
    return 0
fi
HATAORI_BOUNDED_RUN_SH=1

# Seconds between the deadline's SIGTERM and the escalation to SIGKILL.
bounded_grace=${HATAORI_BOUNDED_GRACE:-2}

# Prints PID and its descendants, children first.
bounded_tree_pids() {
    local parent=$1 pid ppid
    ps -A -o pid= -o ppid= 2>/dev/null |
        while read -r pid ppid; do
            [[ $ppid == "$parent" ]] || continue
            bounded_tree_pids "$pid"
            echo "$pid"
        done
    echo "$parent"
}

# Prints the PIDs in session SID (the session leader's pid), the leader included.
# Empty when `ps` cannot report sessions, which leaves the group and tree sweeps.
bounded_session_pids() {
    local sid=$1 pid sess
    ps -A -o pid= -o sess= 2>/dev/null |
        while read -r pid sess; do
            if [[ $sess == "$sid" ]]; then
                echo "$pid"
            fi
        done
    # Callers run under `set -e`: a scan that matched nothing is not a failure.
    return 0
}

# Signals every PID given; a process that already exited is not an error.
bounded_signal_pids() {
    local signal=$1 pid
    shift
    for pid in "$@"; do
        kill -"$signal" "$pid" 2>/dev/null || true
    done
}

# Signals the process group led by PID, which catches anything the command
# started, including processes created while it handled the TERM.
bounded_signal_group() {
    kill -"$1" -- "-$2" 2>/dev/null || true
}

# Signals every process in session SID, in a fresh enumeration each time: a TERM
# handler can start members that no earlier snapshot or group list knows about.
bounded_signal_session() {
    local signal=$1 sid=$2
    local pids
    pids=$(bounded_session_pids "$sid" || true)
    # shellcheck disable=SC2086
    bounded_signal_pids "$signal" $pids
}

bounded() {
    local secs=$1
    shift
    local child status=0 watchdog marker timed_out=0 monitor=0 session=0
    local targets=
    marker=$(mktemp "${TMPDIR:-/tmp}/hataori-bounded.XXXXXX")
    # A session is what survives setpgrp/setpgid and reparenting. `python3` starts
    # one with `os.setsid` in place of the command (`execvp`), so the pid we get is
    # the session leader; job control is the fallback, and it only reaches the
    # group.
    if command -v python3 >/dev/null 2>&1; then
        session=1
        python3 -c 'import os, sys
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])' "$@" &
        child=$!
    else
        # Job control puts the command in its own process group; without it the
        # group would be this shell's, which the deadline cannot signal.
        case $- in *m*) monitor=1 ;; esac
        set -m
        "$@" &
        child=$!
        ((monitor)) || set +m
    fi
    (
        sleep "$secs"
        # Record the deadline before signalling, so the caller cannot read a
        # command that exits zero after its TERM as a success.
        echo fired >"$marker"
        targets=$(bounded_tree_pids "$child")
        if ((session)); then
            bounded_signal_session TERM "$child"
        fi
        bounded_signal_group TERM "$child"
        # shellcheck disable=SC2086
        bounded_signal_pids TERM $targets
        sleep "$bounded_grace"
        # Re-enumerate: a TERM handler may have started a new group since.
        if ((session)); then
            bounded_signal_session KILL "$child"
        fi
        bounded_signal_group KILL "$child"
        # shellcheck disable=SC2086
        bounded_signal_pids KILL $targets
    ) >/dev/null 2>&1 &
    watchdog=$!
    # The shell reports a signalled background job on its own stderr here.
    wait "$child" 2>/dev/null || status=$?
    if [[ -s $marker ]]; then
        timed_out=1
        # The deadline fired: let the watchdog finish its escalation instead of
        # cancelling it. The direct child often dies on the TERM while a
        # TERM-ignoring descendant is still inside the grace period.
        wait "$watchdog" >/dev/null 2>&1 || true
        # The escalation cannot have run after its own KILL, but a member of the
        # session can still be starting, so sweep once more.
        if ((session)); then
            bounded_signal_session KILL "$child"
        fi
        bounded_signal_group KILL "$child"
        targets=$(bounded_tree_pids "$child")
        # shellcheck disable=SC2086
        bounded_signal_pids KILL $targets
    else
        # The command finished. Anything still in its session outlived it and can
        # still hold this function's stdout, which would keep the caller's command
        # substitution open past the bound, so it is killed rather than inherited.
        if ((session)); then
            bounded_signal_session KILL "$child"
        fi
        bounded_signal_group KILL "$child"
        # Take the watchdog's `sleep` with it rather than leaving it to expire.
        targets=$(bounded_tree_pids "$watchdog")
        # shellcheck disable=SC2086
        bounded_signal_pids KILL $targets
        wait "$watchdog" >/dev/null 2>&1 || true
    fi
    rm -f "$marker"
    if [[ $timed_out == 1 && $status == 0 ]]; then
        status=124
    fi
    # A killed command reports 143 (TERM) or 137 (KILL), like `timeout` does for
    # the same escalation; callers only require "non-zero, within the deadline".
    return "$status"
}
