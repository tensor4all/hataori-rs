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
# - the command's stdout and stderr go to temporary files that are replayed when
#   the bound ends, so no descendant can keep the caller waiting by holding the
#   pipe: without this, a stray that outlives the command made a command
#   substitution block for the stray's whole lifetime (measured: 6 s for a 1 s
#   bound), and on a platform with no session scan that stray cannot be found at
#   all (macOS: `ps -o sess=` reports 0 and `pgrep` has no `-s`);
# - where a session can be scanned (Linux: `ps -o sess=`), the command runs as the
#   leader of its own session with `python3` and `os.setsid`, and the deadline
#   signals every member of that session — `setpgrp` changes the group, not the
#   session, so that escape is caught there too. Where no session scan exists
#   (measured on macOS: `ps -A -o sess=` reports 0 for every process and `pgrep`
#   rejects `-s`), the bound still ends on time and the caller never waits, but a
#   descendant that changes its process group and is reparented before the deadline
#   can survive. Polling the tree does not close that gap either: a poller observed
#   the child of a `setpgrp`-spawning parent in 0 of 10 attempts at 20, 50 and
#   200 ms, because the parent exits first. That leftover is a leak of the arm's
#   own processes, not a stalled gate, and the regression test asserts the kill
#   only where a scan exists;
# - the descendant list is taken once, before the first signal: a TERM can
#   reparent a grandchild to PID 1, and a later tree traversal would miss it;
# - a command that ignores SIGTERM is escalated to SIGKILL after
#   `bounded_grace` — the same TERM-then-KILL contract as
#   `timeout --kill-after` — so the arm cannot outlive the deadline it was
#   supposed to enforce;
# - the replay is bounded too: it writes on *your* stdout, so a caller that stops
#   reading parks it on a full pipe, and a synchronous `cat` then held `bounded 1`
#   for as long as the reader stayed away (measured: 4 MiB of output, 6.8 s). Each
#   stream is replayed up to `bounded_replay_cap` bytes and for at most
#   `bounded_replay_secs`, in the order stdout-then-stderr it always had, and
#   whatever is left is dropped with a notice. The notice is replayed under the
#   same bound, on the same stderr: a synchronous notice there blocked on the
#   pipe whose replay had just timed out (measured: still blocked after 6 s with
#   a 1 s bound). A stopped caller therefore costs at most three
#   `bounded_replay_secs` beyond the deadline and never more;
# - the deadline is recorded, so a command that traps TERM and exits 0 is
#   reported as a timeout (124, as `timeout` reports it) instead of a success.

if [[ -n ${HATAORI_BOUNDED_RUN_SH:-} ]]; then
    return 0
fi
HATAORI_BOUNDED_RUN_SH=1

# Seconds between the deadline's SIGTERM and the escalation to SIGKILL.
bounded_grace=${HATAORI_BOUNDED_GRACE:-2}

# Bytes replayed from each spooled stream, and the seconds each replay may take
# before the rest is dropped (see `bounded_replay_stream`). Two streams plus the
# truncation notice, so a caller that stops reading costs at most three times
# this beyond the deadline.
bounded_replay_cap=${HATAORI_BOUNDED_REPLAY_CAP:-1048576}
bounded_replay_secs=${HATAORI_BOUNDED_REPLAY_SECS:-1}

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

# Which scan can actually report a numeric session id here? The *semantics* are
# what matter: macOS `ps -o sess=` accepts the keyword and prints 0, so probing
# the keyword selected a scan that matched nothing and never fell through to
# `pgrep` (an escaping descendant then took 30 s against a 1 s bound). This probes
# both scans against this process's own session id and keeps the one that finds it.
bounded_detect_session_scan() {
    local sid
    sid=$(python3 -c 'import os; print(os.getsid(0))' 2>/dev/null) || {
        printf 'none'
        return
    }
    if ps -A -o pid= -o sess= 2>/dev/null |
        awk -v pid="$$" -v sid="$sid" '$1 == pid && $2 == sid { found = 1 } END { exit(found ? 0 : 1) }'; then
        printf 'ps'
        return
    fi
    if command -v pgrep >/dev/null 2>&1 &&
        pgrep -s "$sid" 2>/dev/null |
        awk -v pid="$$" '$1 == pid { found = 1 } END { exit(found ? 0 : 1) }'; then
        printf 'pgrep'
        return
    fi
    printf 'none'
}

# Auto by default; `HATAORI_BOUNDED_SESSION_SCAN=ps|pgrep|none` pins it, which is
# how the regression test exercises each backend.
if [[ -z ${HATAORI_BOUNDED_SESSION_SCAN:-} ]]; then
    if command -v python3 >/dev/null 2>&1; then
        HATAORI_BOUNDED_SESSION_SCAN=$(bounded_detect_session_scan)
    else
        HATAORI_BOUNDED_SESSION_SCAN=none
    fi
fi

# Prints the PIDs in session SID (the session leader's pid), the leader included.
bounded_session_pids() {
    local sid=$1 pid sess
    case ${HATAORI_BOUNDED_SESSION_SCAN:-none} in
        ps)
            ps -A -o pid= -o sess= 2>/dev/null |
                while read -r pid sess; do
                    if [[ $sess == "$sid" ]]; then
                        echo "$pid"
                    fi
                done
            ;;
        pgrep)
            pgrep -s "$sid" 2>/dev/null || true
            ;;
    esac
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

# Replays spooled FILE on FD without ever blocking the caller. The replay writes
# on the caller's pipe, so a caller that stops reading parks it once the pipe is
# full — a synchronous `cat` held `bounded 1` for the reader's whole absence
# (measured: 4 MiB of output, 6.8 s). Dropping the rest is the correct outcome
# here; hanging is the bug. The notice is left in `bounded_replay_cut` for the
# caller to replay: printing it here would block on the very pipe whose replay
# just timed out.
bounded_replay_stream() {
    local file=$1 fd=$2 label=$3 size pid killer status=0
    bounded_replay_cut=
    [[ -s $file ]] || return 0
    size=$(wc -c <"$file" 2>/dev/null) || size=0
    head -c "$bounded_replay_cap" "$file" >&"$fd" 2>/dev/null &
    pid=$!
    # The killer gets its own descriptors: a `sleep` that inherited the caller's
    # stdout kept a command substitution open for its whole life (measured: 5 s
    # for `output=$(bounded 1 printf banner)`).
    (sleep "$bounded_replay_secs"; bounded_signal_pids KILL "$pid") >/dev/null 2>&1 &
    killer=$!
    wait "$pid" 2>/dev/null || status=$?
    # Reap the killer and its `sleep`. `bounded_signal_pids` ignores a process
    # that already exited, and the killer has when it just fired: an unguarded
    # `kill` there aborted a plain `set -e` caller after a successful run
    # (measured: exit 1 after 1.28 s, never reaching the next statement).
    # shellcheck disable=SC2046
    bounded_signal_pids KILL $(bounded_tree_pids "$killer")
    wait "$killer" 2>/dev/null || true
    if ((size > bounded_replay_cap)); then
        printf -v bounded_replay_cut '%s truncated after %s bytes\n' "$label" "$bounded_replay_cap"
    elif ((status == 137)); then
        # 137 is the kill above: the replay never reached the end of its spool.
        printf -v bounded_replay_cut '%s truncated after %ss\n' "$label" "$bounded_replay_secs"
    fi
}

bounded() {
    local secs=$1
    shift
    local child status=0 watchdog marker timed_out=0 monitor=0 session=0
    local targets= stdout_file= stderr_file= notice_file= notices=
    marker=$(mktemp "${TMPDIR:-/tmp}/hataori-bounded.XXXXXX")
    # Own copies of the output: a descendant holding the caller's pipe is what
    # kept a substitution waiting past the bound.
    stdout_file=$(mktemp "${TMPDIR:-/tmp}/hataori-bounded-out.XXXXXX")
    stderr_file=$(mktemp "${TMPDIR:-/tmp}/hataori-bounded-err.XXXXXX")
    # A session is what survives setpgrp/setpgid and reparenting. `python3` starts
    # one with `os.setsid` in place of the command (`execvp`), so the pid we get is
    # the session leader; job control is the fallback, and it only reaches the
    # group.
    if [[ ${HATAORI_BOUNDED_SESSION_SCAN:-none} != none ]] && command -v python3 >/dev/null 2>&1; then
        session=1
        python3 -c 'import os, sys
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])' "$@" >"$stdout_file" 2>"$stderr_file" &
        child=$!
    else
        # Job control puts the command in its own process group; without it the
        # group would be this shell's, which the deadline cannot signal.
        case $- in *m*) monitor=1 ;; esac
        set -m
        "$@" >"$stdout_file" 2>"$stderr_file" &
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
    # Replay what the command wrote. Its own descendants may still hold these
    # files, which is harmless: nothing the caller reads is shared with them.
    bounded_replay_stream "$stdout_file" 1 output
    notices+=$bounded_replay_cut
    bounded_replay_stream "$stderr_file" 2 stderr
    notices+=$bounded_replay_cut
    rm -f "$marker" "$stdout_file" "$stderr_file"
    if [[ -n $notices ]]; then
        notice_file=$(mktemp "${TMPDIR:-/tmp}/hataori-bounded-notice.XXXXXX")
        printf '%s' "$notices" >"$notice_file"
        bounded_replay_stream "$notice_file" 2 notice
        rm -f "$notice_file"
    fi
    if [[ $timed_out == 1 && $status == 0 ]]; then
        status=124
    fi
    # A killed command reports 143 (TERM) or 137 (KILL), like `timeout` does for
    # the same escalation; callers only require "non-zero, within the deadline".
    return "$status"
}
