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
# - the timeout signals the command *and every descendant*, not just the
#   immediate child, so MPI ranks cannot outlive the arm (`sh -c 'sleep 30 & wait'`
#   left its `sleep` alive when the fallback killed only `sh`);
# - a command that ignores SIGTERM is escalated to SIGKILL after
#   `bounded_grace` — the same TERM-then-KILL contract as
#   `timeout --kill-after` — so the arm cannot outlive the deadline it was
#   supposed to enforce.

if [[ -n ${HATAORI_BOUNDED_RUN_SH:-} ]]; then
    return 0
fi
HATAORI_BOUNDED_RUN_SH=1

# Seconds between the deadline's SIGTERM and the escalation to SIGKILL.
bounded_grace=${HATAORI_BOUNDED_GRACE:-2}

# Signals PID and its descendants, children first.
bounded_signal_tree() {
    local signal=$1 pid=$2 child
    for child in $(ps -A -o pid= -o ppid= 2>/dev/null | awk -v parent="$pid" '$2 == parent {print $1}'); do
        bounded_signal_tree "$signal" "$child"
    done
    kill -"$signal" "$pid" 2>/dev/null || true
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

    local child status=0 watchdog
    "$@" &
    child=$!
    (
        sleep "$secs"
        kill -0 "$child" 2>/dev/null || exit 0
        bounded_signal_tree TERM "$child"
        sleep "$bounded_grace"
        bounded_signal_tree KILL "$child"
    ) >/dev/null 2>&1 &
    watchdog=$!
    # The shell reports a signalled background job on its own stderr here.
    wait "$child" 2>/dev/null || status=$?
    # The watchdog outlives a command that finished first; take its `sleep` with
    # it rather than leaving it to expire.
    bounded_signal_tree KILL "$watchdog"
    wait "$watchdog" >/dev/null 2>&1 || true
    # A killed command reports 143 (TERM) or 137 (KILL), like `timeout` does for
    # the same escalation; callers only require "non-zero, within the deadline".
    return "$status"
}
