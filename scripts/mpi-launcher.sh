#!/usr/bin/env bash
# Shared MPI launcher helpers for the acceptance scripts. This file is sourced,
# never executed:
#
#   source "$root_dir/scripts/mpi-launcher.sh"
#
# `mpi_launcher_supports_oversubscribe LAUNCHER` reports whether LAUNCHER takes
# Open MPI's `--oversubscribe`. The banner alone is not enough to decide a
# hosted runner runs Open MPI 5, whose launcher reports the PRRTE runtime: the
# `Open MPI|OpenRTE` match silently dropped the flag there, so `-n 4` failed to
# allocate slots on a three-CPU runner (the same runner whose affinity mask
# omits cpu 0, issue 41). When the banner does not identify the launcher, the
# launcher is asked directly.
#
# `mpi_rank_counts CAP` prints the rank counts a lane may use: 1, 2 and
# min(CAP, available CPUs). Four ranks on a three-CPU runner is a
# slot-allocation failure, not a test.

# The probe below is bounded by `scripts/bounded-run.sh`, which also covers the
# platforms without `setsid` — otherwise a supported launcher whose banner this
# side does not recognize would be reported as unsupported on macOS.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/bounded-run.sh"

mpi_launcher_supports_oversubscribe() {
    local launcher=$1 banner
    # Bounded as well: a site wrapper whose --version never returns would block
    # every script that sources this helper, before the bounded probe below runs.
    banner=$(bounded 10s "$launcher" --version 2>&1 || true)
    if grep -Eqi 'open[ -]?mpi|openrte|prrte|prte' <<<"$banner"; then
        return 0
    fi
    bounded 30s "$launcher" --oversubscribe -n 1 true >/dev/null 2>&1
}

# The CPUs this process may run on. `nproc` and `getconf _NPROCESSORS_ONLN`
# answer for the host on some systems, so read the mask the launcher's
# children inherit instead.
mpi_available_cpus() {
    local count
    if count=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null); then
        printf '%s\n' "$count"
    elif count=$(getconf _NPROCESSORS_ONLN 2>/dev/null); then
        printf '%s\n' "$count"
    else
        printf 'mpi-launcher: cannot determine the CPUs available to this process\n' >&2
        exit 1
    fi
}

mpi_rank_counts() {
    local cap=$1 available max
    # Callers assign this in a checked command; a failed CPU query must fail
    # the gate rather than sweep a silently reduced set of ranks.
    available=$(mpi_available_cpus) || return 1
    max=$cap
    if ((available < max)); then
        max=$available
    fi
    if ((max < 2)); then
        max=2
    fi
    printf '1 2'
    if ((max > 2)); then
        printf ' %s' "$max"
    fi
    printf '\n'
}
