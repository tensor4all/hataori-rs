#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

cargo fmt --all -- --check
cargo test -p hataori-runtime-foundation --no-default-features
cargo test -p hataori-runtime-foundation --no-default-features --features mpi --lib
cargo clippy -p hataori-runtime-foundation --all-targets --no-default-features -- -D warnings
cargo clippy -p hataori-runtime-foundation --all-targets --no-default-features --features mpi -- -D warnings
cargo doc -p hataori-runtime-foundation --no-deps --no-default-features
cargo doc -p hataori-runtime-foundation --no-deps --no-default-features --features mpi
cargo +1.85.0 check -p hataori-runtime-foundation --no-default-features
cargo +1.85.0 check -p hataori-runtime-foundation --no-default-features --features mpi

if cargo tree -p hataori-runtime-foundation --no-default-features --prefix none |
    grep -Eq '^(mpi|mpi-sys) '; then
    printf 'runtime foundation: default feature tree contains MPI\n' >&2
    exit 1
fi
if rg -n 'mpi::|mpi_upstream|mpi-runtime|SimpleCommunicator|TcpStream|SocketAddr' \
    crates/hataori-runtime-foundation/src/protocol.rs \
    crates/hataori-runtime-foundation/src/transport.rs \
    crates/hataori-runtime-foundation/src/memory.rs; then
    printf 'runtime foundation: backend type leaked above its owner\n' >&2
    exit 1
fi
if rg -n 'unsafe impl (Send|Sync)' crates/hataori-runtime-foundation/src; then
    printf 'runtime foundation: unsafe thread-safety wrapper is forbidden\n' >&2
    exit 1
fi

cargo build -p hataori-runtime-foundation --no-default-features --bin tcp_transport_smoke
setsid timeout --signal=TERM --kill-after=2s 30s target/debug/tcp_transport_smoke

cargo build -p hataori-runtime-foundation --no-default-features --features mpi \
    --bin mpi_transport_smoke
source "$root_dir/scripts/mpi-launcher.sh"
launcher=${MPIEXEC:-mpiexec}
launcher_flags=()
if mpi_launcher_supports_oversubscribe "$launcher"; then
    launcher_flags+=(--oversubscribe)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi
for count in $(mpi_rank_counts 4); do
    output=$(mktemp)
    set +e
    env -u DISPLAY setsid timeout --signal=TERM --kill-after=2s 45s \
        "$launcher" "${launcher_flags[@]}" -n "$count" \
        target/debug/mpi_transport_smoke >"$output" 2>&1
    status=$?
    set -e
    if ((status != 0)); then
        cat "$output" >&2
        rm -f "$output"
        printf 'runtime foundation: MPI contract failed at n=%s (status=%s)\n' \
            "$count" "$status" >&2
        exit 1
    fi
    rm -f "$output"
done

git diff --check
printf 'Phase A protocol/memory/TCP/MPI transport foundation passed\n'
