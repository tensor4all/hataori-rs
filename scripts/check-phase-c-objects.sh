#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

run_group() {
    local seconds=$1
    shift
    setsid timeout --signal=TERM --kill-after=2s "$seconds" "$@"
}

cargo fmt --check
run_group 30 cargo test -p hataori-runtime --lib
run_group 30 cargo test -p hataori-runtime --no-default-features --features mpi --lib
run_group 30 cargo clippy -p hataori-runtime --all-targets -- -D warnings
run_group 30 cargo clippy -p hataori-runtime --all-targets --no-default-features --features mpi -- -D warnings
run_group 30 env RUSTDOCFLAGS=-Dwarnings cargo doc -p hataori-runtime --no-deps
run_group 30 env RUSTDOCFLAGS=-Dwarnings cargo doc -p hataori-runtime --no-deps --no-default-features --features mpi
run_group 30 cargo +1.85.0 check -p hataori-runtime
run_group 30 cargo +1.85.0 check -p hataori-runtime --no-default-features --features mpi

if grep -REn 'unsafe impl[[:space:]]+(Send|Sync)' crates/hataori-runtime/src; then
    printf 'check-phase-c-objects: unsafe thread-safety implementation found\n' >&2
    exit 1
fi
if grep -REn 'mpi::|SimpleCommunicator|TcpStream|SocketAddr' crates/hataori-runtime/src/*.rs; then
    printf 'check-phase-c-objects: backend type leaked into runtime/object layers\n' >&2
    exit 1
fi

run_group 20 cargo run -p hataori-runtime --bin tcp_runtime_smoke
run_group 30 cargo build -p hataori-runtime --no-default-features --features mpi --bin mpi_runtime_smoke
mpi_flags=()
source "$root_dir/scripts/mpi-launcher.sh"
if mpi_launcher_supports_oversubscribe mpiexec; then
    mpi_flags+=(--oversubscribe)
fi
for n in $(mpi_rank_counts 4); do
    run_group 15 mpiexec "${mpi_flags[@]}" -n "$n" target/debug/mpi_runtime_smoke
done

git diff --check
printf 'Phase C fixed-placement remote objects passed\n'
