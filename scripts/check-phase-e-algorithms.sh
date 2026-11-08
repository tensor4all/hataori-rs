#!/usr/bin/env bash
set -Eeuo pipefail
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}
run(){ local seconds=$1; shift; setsid timeout --signal=TERM --kill-after=2s "$seconds" "$@"; }
cargo fmt --check
run 30 cargo test -p hataori-runtime --lib
run 30 cargo test -p hataori-algorithms
run 30 cargo clippy -p hataori-runtime -p hataori-algorithms --all-targets -- -D warnings
run 30 env RUSTDOCFLAGS=-Dwarnings cargo doc -p hataori-runtime -p hataori-algorithms --no-deps
run 30 cargo +1.85.0 check -p hataori-runtime -p hataori-algorithms
run 30 cargo check --features runtime
run 30 cargo run -p hataori-runtime --bin tcp_runtime_smoke
run 30 cargo build -p hataori-runtime --features mpi --bin mpi_runtime_smoke
mpi_flags=()
source "$root_dir/scripts/mpi-launcher.sh"
if mpi_launcher_supports_oversubscribe mpiexec; then
    mpi_flags+=(--oversubscribe)
fi
rank_counts=$(mpi_rank_counts 4)
for n in $rank_counts; do run 20 env -u DISPLAY mpiexec "${mpi_flags[@]}" -n "$n" target/debug/mpi_runtime_smoke; done
if grep -REn 'todo!|unimplemented!|TODO|FIXME|unsafe impl[[:space:]]+(Send|Sync)' crates/hataori-algorithms/src crates/hataori-runtime/src; then exit 1; fi
scripts/check-performance-manifest.py
git diff --check
printf 'Phase E algorithms and facade correctness passed\n'
