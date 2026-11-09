#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s ABSOLUTE_MPIWRAPPER_LIBRARY MPIWRAPPER_COMMIT\n' "$0" >&2
    exit 2
fi
mpi_rt_lib=$1
fixture_commit=$2
[[ $mpi_rt_lib = /* && -f $mpi_rt_lib ]] || {
    printf 'check-core: MPIwrapper path must be an existing absolute file\n' >&2
    exit 2
}

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

# The committed Cargo.lock is the merge gate for this lane. Fail before any
# unlocked cargo run below can resolve a newer dependency set and silently
# rewrite it; `deps-fresh` is the one lane allowed to move the lock.
cargo metadata --locked --format-version 1 >/dev/null

# The `tenferro` feature (Mandelbrot examples only) needs Rust >= 1.96 and is
# therefore outside the 1.85 matrix below; it is built and run separately.
scripts/check-tutorial-examples.sh "$mpi_rt_lib"
# The arm watchdog and the launcher probe have portable fallbacks; both are
# exercised here with `setsid`/`timeout` hidden from PATH.
scripts/check-bounded-run.sh

cargo fmt --check
for features in '' rayon mpi mpi,rayon rsmpi-rt rsmpi-rt,rayon; do
    if [[ -z $features ]]; then
        args=(--no-default-features)
    else
        args=(--no-default-features --features "$features")
    fi
    cargo test --locked "${args[@]}"
    cargo clippy --all-targets --locked "${args[@]}" -- -D warnings
    cargo doc --no-deps --locked "${args[@]}"
    cargo +1.85.0 check --locked "${args[@]}"
done

for features in '' rayon; do
    if [[ -z $features ]]; then
        tree=$(cargo tree --no-default-features --prefix none)
    else
        tree=$(cargo tree --no-default-features --features "$features" --prefix none)
    fi
    if grep -Eq '^(mpi|mpi-sys|mpi-rt-sys|serde|bincode|tenferro|tensor4all) ' <<<"$tree"; then
        printf 'check-core: forbidden dependency in %s tree\n%s\n' "${features:-default}" "$tree" >&2
        exit 1
    fi
done

if cargo check --no-default-features --features mpi,rsmpi-rt > /tmp/hataori-both-features.log 2>&1; then
    printf 'check-core: mutually exclusive MPI backends compiled\n' >&2
    exit 1
fi
grep -q 'mutually exclusive' /tmp/hataori-both-features.log

scripts/check-mpi-call-boundaries.py
scripts/check-compile-boundaries.sh
scripts/check-rsmpi-rt.sh "$mpi_rt_lib" "$fixture_commit"
scripts/check-hybrid.sh "$mpi_rt_lib"
scripts/check-placement.sh "$mpi_rt_lib"
scripts/check-faults.sh "$mpi_rt_lib"

if rustup target list --installed | grep -qx 'x86_64-apple-darwin'; then
    cargo check --no-default-features --features rayon --target x86_64-apple-darwin
fi

git diff --check
printf 'complete Hataori core acceptance matrix passed\n'
