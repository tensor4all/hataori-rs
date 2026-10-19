#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 1)); then
    printf 'usage: %s ABSOLUTE_MPIWRAPPER_LIBRARY\n' "$0" >&2
    exit 2
fi
mpi_rt_lib=$1
[[ $mpi_rt_lib = /* && -f $mpi_rt_lib ]] || {
    printf 'check-hybrid: MPIwrapper path must be an existing absolute file\n' >&2
    exit 2
}

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
cd "$root_dir"

export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}
cargo build --example mpi_pmap_smoke --no-default-features --features mpi,rayon
cp target/debug/examples/mpi_pmap_smoke "$tmp_dir/upstream"
cargo build --example mpi_pmap_smoke --no-default-features --features mpi
cp target/debug/examples/mpi_pmap_smoke "$tmp_dir/upstream-mpi-only"
cargo build --example rsmpi_rt_pmap_smoke --no-default-features --features rsmpi-rt,rayon
cp target/debug/examples/rsmpi_rt_pmap_smoke "$tmp_dir/runtime"

launcher=${MPIEXEC:-mpiexec}
launcher_flags=()
if "$launcher" --version 2>&1 | grep -Eq 'Open MPI|OpenRTE'; then
    launcher_flags+=(--oversubscribe)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi

run_case() {
    local backend=$1
    local count=$2
    local trace=$tmp_dir/trace-$backend-$count
    local binary=$tmp_dir/$backend
    local output=$tmp_dir/$backend-$count.log
    local -a environment=(env HATAORI_HYBRID_TRACE=$trace)
    if [[ $backend == runtime ]]; then
        environment+=(MPI_RT_LIB=$mpi_rt_lib)
    fi
    rm -f "$trace"
    if ! setsid timeout --signal=TERM --kill-after=5s 120s \
        "${environment[@]}" "$launcher" "${launcher_flags[@]}" -n "$count" "$binary" \
        >"$output" 2>&1; then
        cat "$output" >&2
        printf 'check-hybrid: %s backend failed at n=%s\n' "$backend" "$count" >&2
        return 1
    fi
    [[ ! -e $trace ]] || {
        printf 'check-hybrid: trace cleanup failed for %s n=%s\n' "$backend" "$count" >&2
        return 1
    }
}

# Runs one abort fixture at n=2 and requires the exact MPI abort code.
#
# The code must be observed either in the launcher's abort banner or as the
# launcher's exit status: Open MPI's banner rendering intermittently emits its
# own "Data unpack" show_help error instead of the banner for the same abort, so
# the exit status is the equivalent, equally exact evidence.
expect_abort() {
    local backend=$1 binary=$2 code=$3 deadline=$4 label=$5
    shift 5
    local output=$tmp_dir/$backend-$label.log
    local -a environment=(env "$@")
    if [[ $backend == runtime ]]; then
        environment+=(MPI_RT_LIB=$mpi_rt_lib)
    fi
    set +e
    setsid timeout --signal=TERM --kill-after=5s "$deadline" \
        "${environment[@]}" "$launcher" "${launcher_flags[@]}" -n 2 "$binary" \
        >"$output" 2>&1
    local status=$?
    set -e
    if ((status == 0 || status == 124 || status == 137)); then
        cat "$output" >&2
        printf 'check-hybrid: %s %s did not abort promptly\n' "$backend" "$label" >&2
        return 1
    fi
    if grep -Eq 'MPI_ABORT|MPI_Abort' "$output" \
        && grep -Eq "errorcode[^0-9]*$code|code[^0-9]*$code" "$output"; then
        return 0
    fi
    if ((status == code)); then
        return 0
    fi
    cat "$output" >&2
    printf 'check-hybrid: %s %s lacked abort(%s) evidence\n' "$backend" "$label" "$code" >&2
    return 1
}

for count in 1 2 4; do
    output=$tmp_dir/upstream-mpi-only-$count.log
    if ! setsid timeout --signal=TERM --kill-after=5s 120s \
        "$launcher" "${launcher_flags[@]}" -n "$count" "$tmp_dir/upstream-mpi-only" \
        >"$output" 2>&1; then
        cat "$output" >&2
        printf 'check-hybrid: upstream MPI-only backend failed at n=%s\n' "$count" >&2
        exit 1
    fi
done

for backend in upstream runtime; do
    for count in 1 2 4; do
        run_case "$backend" "$count"
    done

    binary=$tmp_dir/$backend
    if ! expect_abort "$backend" "$binary" 70 60s panic HATAORI_HYBRID_PANIC=1; then
        exit 1
    fi

    for count in 1 2; do
        output=$tmp_dir/$backend-single-lane-$count.log
        environment=(env HATAORI_SINGLE_LANE_FALLBACK=1)
        if [[ $backend == runtime ]]; then
            environment+=(MPI_RT_LIB=$mpi_rt_lib)
        fi
        if ! setsid timeout --signal=TERM --kill-after=5s 120s \
            "${environment[@]}" "$launcher" "${launcher_flags[@]}" -n "$count" "$binary" \
            >"$output" 2>&1; then
            cat "$output" >&2
            printf 'check-hybrid: %s single-lane coordinator fallback failed at n=%s\n' \
                "$backend" "$count" >&2
            exit 1
        fi
    done

    if ! expect_abort "$backend" "$binary" 71 15s single-lane-prefetch-panic \
        HATAORI_SINGLE_LANE_PREFETCH_PANIC=1; then
        exit 1
    fi

    trace=$tmp_dir/$backend-prefetch-transfer-abort
    if ! expect_abort "$backend" "$binary" 75 15s prefetch-transfer-abort \
        HATAORI_PREFETCH_TRANSFER_ABORT="$trace"; then
        exit 1
    fi
done

printf 'MPI-only preflight, hybrid lanes, single-lane fallback, and single-lane prefetch panic checks passed for mpi and rsmpi-rt (n=1,2,4)\n'
