#!/usr/bin/env bash
# Builds and runs the Mandelbrot examples that the tutorials quote, at a
# small image size, for every supported feature set.
#
# Usage: scripts/check-tutorial-examples.sh [ABSOLUTE_MPIWRAPPER_LIBRARY]
#
# Without an argument the `rsmpi-rt` lanes are skipped. `mpiexec` must be on
# PATH for the MPI lanes (override with HATAORI_MPIEXEC; rank count with
# HATAORI_MPI_RANKS, default 2). The `tenferro` feature needs Rust >= 1.96.
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
mpi_rt_lib=${1:-}
mpiexec_bin=${HATAORI_MPIEXEC:-mpiexec}
ranks=${HATAORI_MPI_RANKS:-2}
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

# Small enough to finish in seconds, large enough to reach `max_iter`.
size_args=(--width 256 --height 256 --max-iter 200)

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

launcher=("$mpiexec_bin")
if "$mpiexec_bin" --version 2>&1 | grep -Eq 'Open MPI|OpenRTE'; then
    # Small CI machines: allow ranks x workers to exceed the core count and do
    # not bind each rank to one core, which would make the managed Rayon
    # domains fail their CPU-set validation.
    launcher+=(--oversubscribe --bind-to none)
fi

build_lane() {
    local features=$1
    shift
    echo "== build: features='$features' examples: $*"
    local args=()
    for example in "$@"; do
        args+=(--example "$example")
    done
    cargo build --release --no-default-features --features "$features" "${args[@]}"
    cargo clippy --release --no-default-features --features "$features" "${args[@]}" -- -D warnings
}

run_direct() {
    local example=$1
    shift
    echo "== run: $example $*"
    (cd "$work_dir" && "$root_dir/target/release/examples/$example" "${size_args[@]}" "$@")
    [[ -f $work_dir/$example.png ]]
}

run_mpi() {
    local example=$1
    shift
    echo "== run: $example (mpiexec -n $ranks) $*"
    (cd "$work_dir" && "${launcher[@]}" -n "$ranks" \
        "$root_dir/target/release/examples/$example" "${size_args[@]}" "$@")
    [[ -f $work_dir/$example.png ]]
}

run_mpi_fails_fast() {
    local example=$1
    shift
    echo "== expect a bounded failure: $example (mpiexec -n $ranks) $*"
    local started=$SECONDS
    if (cd "$work_dir" && timeout 60 "${launcher[@]}" -n "$ranks" \
        "$root_dir/target/release/examples/$example" "${size_args[@]}" "$@"); then
        echo "check-tutorial-examples: $example $* was expected to fail" >&2
        return 1
    fi
    if ((SECONDS - started > 45)); then
        echo "check-tutorial-examples: $example $* took $((SECONDS - started))s; the abort path is not bounded" >&2
        return 1
    fi
    [[ ! -f $work_dir/$example.png ]] \
        || { echo "check-tutorial-examples: $example wrote a PNG after a failure" >&2; return 1; }
}

build_lane tenferro serial_mandelbrot
run_direct serial_mandelbrot --output serial_mandelbrot.png

build_lane rayon,tenferro rayon_mandelbrot
run_direct rayon_mandelbrot --threads 2 --output rayon_mandelbrot.png

build_lane mpi,tenferro mpi_mandelbrot_raw mpi_mandelbrot_pmap mpi_mandelbrot_placement
run_mpi mpi_mandelbrot_raw --output mpi_mandelbrot_raw.png
run_mpi mpi_mandelbrot_pmap --output mpi_mandelbrot_pmap.png
run_mpi mpi_mandelbrot_placement --output mpi_mandelbrot_placement.png

build_lane mpi,rayon,tenferro mpi_mandelbrot_hybrid
run_mpi mpi_mandelbrot_hybrid --workers 2 --output mpi_mandelbrot_hybrid.png
# #41: an inadmissible worker request must abort the job instead of leaving the other
# rank in the collective until the job timeout. The value also overflows a naive
# `size * workers * factor` product.
run_mpi_fails_fast mpi_mandelbrot_hybrid --workers 288230376151711744

if [[ -n $mpi_rt_lib ]]; then
    [[ $mpi_rt_lib = /* && -f $mpi_rt_lib ]] || {
        printf 'check-tutorial-examples: MPIwrapper path must be an existing absolute file\n' >&2
        exit 2
    }
    export MPI_RT_LIB=$mpi_rt_lib
    build_lane rsmpi-rt,tenferro rsmpi_rt_mandelbrot_pmap rsmpi_rt_mandelbrot_placement
    run_mpi rsmpi_rt_mandelbrot_pmap --output rsmpi_rt_mandelbrot_pmap.png
    run_mpi rsmpi_rt_mandelbrot_placement --output rsmpi_rt_mandelbrot_placement.png
    build_lane rsmpi-rt,rayon,tenferro rsmpi_rt_mandelbrot_hybrid
    run_mpi rsmpi_rt_mandelbrot_hybrid --workers 2 --output rsmpi_rt_mandelbrot_hybrid.png
else
    echo "== rsmpi-rt lanes skipped (no MPIwrapper library given); checking they compile"
    cargo check --no-default-features --features rsmpi-rt,rayon,tenferro --examples
fi

echo "hataori tutorial examples passed"
