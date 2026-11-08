#!/usr/bin/env bash
# `bounded SECONDS COMMAND...`: run COMMAND and return its status, terminating
# it once SECONDS have elapsed. Sourced, never executed:
#
#   source "$root_dir/scripts/bounded-run.sh"
#
# `setsid timeout` is the Linux form. macOS has neither, and an arm without a
# bound holds the whole job to its job-level timeout (issue 41: a rank blocked in
# the collective), so the portable path has to bound and clean up as well:
#
# - the command runs as the leader of its own process group (job control), so the
#   deadline signals the *group*: a process started inside a TERM handler is in no
#   snapshot taken before the signal, but it does inherit the group. The PID tree
#   is signalled as well, which covers a shell that never entered monitor mode;
# - the descendant list is taken once, before the first signal: a TERM can
#   reparent a grandchild to PID 1, and a second traversal at KILL time would
#   miss exactly the TERM-ignoring descendants this fallback exists for;
# - a command that ignores SIGTERM is escalated to SIGKILL after
#   `bounded_grace` — the same TERM-then-KILL contract as
#   `timeout --kill-after` — so the arm cannot outlive the deadline it was
#   supposed to enforce;
# - the deadline is recorded, so a command that traps TERM and exits 0 is
#   reported as a timeout (124, as `timeout` reports it) instead of a success.

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

bounded() {
    local secs=$1
    shift
    if command -v setsid >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
        setsid timeout --signal=TERM --kill-after="${bounded_grace}s" "$secs" "$@"
        return
    fi
    if command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM --kill-after="${bounded_grace}s" "$secs" "$@"
        return
    fi

    local child status=0 watchdog marker timed_out=0 monitor=0
    local targets=
    marker=$(mktemp "${TMPDIR:-/tmp}/hataori-bounded.XXXXXX")
    # Job control puts the command in its own process group; without it the group
    # would be this shell's, which the deadline cannot signal.
    case $- in *m*) monitor=1 ;; esac
    set -m
    "$@" &
    child=$!
    ((monitor)) || set +m
    (
        sleep "$secs"
        # Record the deadline before signalling, so the caller cannot read a
        # command that exits zero after its TERM as a success.
        echo fired >"$marker"
        targets=$(bounded_tree_pids "$child")
        bounded_signal_group TERM "$child"
        # shellcheck disable=SC2086
        bounded_signal_pids TERM $targets
        sleep "$bounded_grace"
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
        bounded_signal_group KILL "$child"
        targets=$(bounded_tree_pids "$child")
        # shellcheck disable=SC2086
        bounded_signal_pids KILL $targets
    else
        # The command finished first; take its `sleep` with it rather than
        # leaving the watchdog to expire.
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
