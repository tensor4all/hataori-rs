#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 1)); then
    printf 'usage: %s ABSOLUTE_MPIWRAPPER_LIBRARY\n' "$0" >&2
    exit 2
fi
mpi_rt_lib=$1
[[ $mpi_rt_lib = /* && -f $mpi_rt_lib ]] || {
    printf 'check-placement: MPIwrapper path must be an existing absolute file\n' >&2
    exit 2
}

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
cd "$root_dir"

export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}
cargo build --example mpi_placement_smoke --no-default-features --features mpi
cp target/debug/examples/mpi_placement_smoke "$tmp_dir/upstream"
cargo build --example rsmpi_rt_placement_smoke --no-default-features --features rsmpi-rt
cp target/debug/examples/rsmpi_rt_placement_smoke "$tmp_dir/runtime"

source "$root_dir/scripts/mpi-launcher.sh"
launcher=${MPIEXEC:-mpiexec}
launcher_flags=()
if mpi_launcher_supports_oversubscribe "$launcher"; then
    launcher_flags+=(--oversubscribe)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi

rank_counts=$(mpi_rank_counts 4)

for backend in upstream runtime; do
    for count in $rank_counts; do
        environment=(env)
        if [[ $backend == runtime ]]; then
            environment+=(MPI_RT_LIB=$mpi_rt_lib)
        fi
        output=$tmp_dir/$backend-$count.log
        if ! setsid timeout --signal=TERM --kill-after=5s 180s \
            "${environment[@]}" "$launcher" "${launcher_flags[@]}" -n "$count" \
            "$tmp_dir/$backend" >"$output" 2>&1; then
            cat "$output" >&2
            printf 'check-placement: %s backend failed at n=%s\n' "$backend" "$count" >&2
            exit 1
        fi
    done
done

printf 'placement checks passed for mpi and rsmpi-rt (n=%s)\n' "${rank_counts// /,}"
