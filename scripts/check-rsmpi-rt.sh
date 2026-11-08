#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf 'usage: %s ABSOLUTE_MPIWRAPPER_LIBRARY [FIXTURE_COMMIT]\n' "$0" >&2
}

die() {
    printf 'check-rsmpi-rt: %s\n' "$*" >&2
    exit 1
}

if (($# < 1 || $# > 2)); then
    usage
    exit 2
fi

mpi_rt_lib=$1
fixture_commit=${2-}
[[ $mpi_rt_lib = /* ]] || die 'MPIwrapper library path must be absolute'
[[ -f $mpi_rt_lib ]] || die "MPIwrapper library does not exist: $mpi_rt_lib"
if [[ -n $fixture_commit ]]; then
    printf 'MPIwrapper fixture commit: %s\n' "$fixture_commit"
fi

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root_dir/scripts/mpi-launcher.sh"

launcher=${MPIEXEC:-mpiexec}
command -v "$launcher" >/dev/null 2>&1 || die "MPI launcher not found: $launcher"
launcher_flags=()
if mpi_launcher_supports_oversubscribe "$launcher"; then
    # `--oversubscribe` lets a job ask for more ranks than the affinity mask
    # has CPUs; `--bind-to none` keeps these count sweeps off the launcher's
    # one-core-per-rank default, which the smoke does not test.
    launcher_flags+=(--oversubscribe --bind-to none)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi
original_home=$HOME
export RUSTUP_HOME=${RUSTUP_HOME:-$original_home/.rustup}
export CARGO_HOME=${CARGO_HOME:-$original_home/.cargo}
rustup_toolchain=${RUSTUP_TOOLCHAIN-}
if [[ -z $rustup_toolchain ]] && command -v rustup >/dev/null 2>&1; then
    rustup_toolchain=$(rustup show active-toolchain | awk 'NR == 1 {print $1}')
fi
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/home"

export HOME=$tmp_dir/home
if [[ -n $rustup_toolchain ]]; then
    export RUSTUP_TOOLCHAIN=$rustup_toolchain
fi
export CARGO_TARGET_DIR=$tmp_dir/target
export CC=$tmp_dir/nonexistent/cc
export CXX=$tmp_dir/nonexistent/cxx
export LIBCLANG_PATH=$tmp_dir/nonexistent/libclang
export MPI_ROOT=$tmp_dir/nonexistent/mpi-root
export MPI_HOME=$tmp_dir/nonexistent/mpi-home
export MPI_CC=$tmp_dir/nonexistent/mpi-cc
export OMPI_CC=$tmp_dir/nonexistent/ompi-cc
export MPICC=$tmp_dir/nonexistent/mpicc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$(command -v gcc)
cd "$root_dir"

cargo check --no-default-features --features rsmpi-rt
cargo build --no-default-features --features rsmpi-rt --example rsmpi_rt_pmap_smoke

tree=$(cargo tree --no-default-features --features rsmpi-rt)
for package in mpi-sys bindgen build-probe-mpi; do
    if grep -Eq "(^|[[:space:]])${package}([[:space:]]|v|$)" <<<"$tree"; then
        die "runtime dependency tree contains forbidden package: $package"
    fi
done

binary=$CARGO_TARGET_DIR/debug/examples/rsmpi_rt_pmap_smoke
[[ -x $binary ]] || die "runtime smoke binary was not built: $binary"

run_mpi() {
    local count=$1
    shift
    setsid timeout --signal=TERM --kill-after=5s 120s \
        "$launcher" "${launcher_flags[@]}" -n "$count" "$@"
}

negative_output=$tmp_dir/negative.log
if (unset MPI_RT_LIB; run_mpi 1 "$binary") >"$negative_output" 2>&1; then
    cat "$negative_output" >&2
    die 'unset MPI_RT_LIB unexpectedly succeeded'
fi
if ! grep -q 'MPI_RT_LIB' "$negative_output"; then
    cat "$negative_output" >&2
    die 'unset MPI_RT_LIB did not reach the runtime-path failure'
fi

invalid_lib=$tmp_dir/nonexistent/libmpiwrapper.so
negative_output=$tmp_dir/invalid.log
if (MPI_RT_LIB=$invalid_lib; export MPI_RT_LIB; run_mpi 1 "$binary") >"$negative_output" 2>&1; then
    cat "$negative_output" >&2
    die 'invalid MPI_RT_LIB unexpectedly succeeded'
fi
if ! grep -q 'Failed to load MPIwrapper library' "$negative_output"; then
    cat "$negative_output" >&2
    die 'invalid MPI_RT_LIB did not fail in the runtime loader'
fi

export MPI_RT_LIB=$mpi_rt_lib
rank_counts=$(mpi_rank_counts 4)
for count in $rank_counts; do
    output=$tmp_dir/positive-$count.log
    if ! run_mpi "$count" "$binary" >"$output" 2>&1; then
        cat "$output" >&2
        die "runtime MPI smoke failed at n=$count"
    fi
done

printf 'rsmpi-rt checks passed (n=%s)\n' "${rank_counts// /,}"
