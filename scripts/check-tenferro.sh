#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

pin=e7edd899def8b80df81f2249a7c3b79d64ea592d

# Every stage runs in its own process group under a short watchdog. A cold
# tenferro dependency build does not fit one window, so `bounded` re-runs the
# same command: cargo resumes from its own cache, each attempt is independently
# bounded, and the number of attempts is fixed. A genuine failure therefore
# costs at most these bounded attempts and still fails the gate.
run() { local seconds=$1; shift; setsid timeout --signal=TERM --kill-after=2s "$seconds" "$@"; }

bounded() {
    local seconds=$1
    shift
    local attempts=${HATAORI_TENFERRO_BUILD_ATTEMPTS:-12}
    local attempt=1
    while true; do
        if run "$seconds" "$@"; then
            return 0
        fi
        if ((attempt >= attempts)); then
            echo "check-tenferro: $* did not finish within $attempts bounded attempts" >&2
            return 1
        fi
        attempt=$((attempt + 1))
    done
}

bounded 30 cargo fmt --all -- --check

bounded 30 cargo test -p hataori-tenferro --all-targets --no-run
run 30 cargo test -p hataori-tenferro --all-targets
bounded 30 cargo test -p hataori-tenferro --doc
bounded 30 cargo test -p hataori --no-default-features --features rayon --doc
bounded 30 cargo clippy -p hataori-tenferro --all-targets -- -D warnings
bounded 30 env RUSTDOCFLAGS="-D warnings" cargo doc -p hataori-tenferro --no-deps

bounded 30 cargo +1.85.0 check -p hataori --no-default-features
bounded 30 cargo +1.85.0 check -p hataori --no-default-features --features rayon

for features in "" rayon; do
    bounded 30 cargo tree -p hataori --no-default-features ${features:+--features "$features"} >/dev/null
    tree=$(bounded 30 cargo tree -p hataori --no-default-features ${features:+--features "$features"})
    if grep -Eq '(^|[[:space:]])(hataori-tenferro|tenferro-(cpu|tensor)|tensor4all)([[:space:]]|$)' <<<"$tree"; then
        echo "check-tenferro: core tree unexpectedly contains adapter/tensor dependencies" >&2
        exit 1
    fi
done

# The adapter tree must carry the exact pinned tenferro revision, and it must
# not pull any tensor4all crate. tenferro-tensor arrives both through
# tenferro-cpu and as the adapter's own dev-dependency for tests and examples.
adapter_tree=$(bounded 30 cargo tree -p hataori-tenferro --prefix none)
for name in tenferro-cpu tenferro-tensor; do
    if ! grep -Eq "^${name} v[0-9][^ ]* \(https://github\.com/tensor4all/tenferro-rs\?rev=${pin}#[0-9a-f]+\)( \(\*\))?$" <<<"$adapter_tree"; then
        echo "check-tenferro: ${name} is not pinned to ${pin}" >&2
        exit 1
    fi
done

metadata=$(mktemp)
trap 'rm -f "$metadata"' EXIT
bounded 30 cargo metadata --format-version 1 >"$metadata"
python3 - "$metadata" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    packages = json.load(stream)["packages"]

by_name = {package["name"]: package for package in packages}
if "hataori-tenferro" not in by_name:
    raise SystemExit("check-tenferro: missing package hataori-tenferro")

pin = "e7edd899def8b80df81f2249a7c3b79d64ea592d"
for name in ("tenferro-cpu", "tenferro-tensor"):
    source = by_name[name].get("source") or ""
    if f"rev={pin}" not in source or not source.endswith(f"#{pin}"):
        raise SystemExit(f"check-tenferro: {name} is not pinned to {pin}: {source}")

for package in packages:
    if package["name"] == "tensor4all" or package["name"].startswith("tensor4all-"):
        raise SystemExit(
            f"check-tenferro: adapter unexpectedly includes {package['name']}"
        )
PY

# `pmap_lanes` over caller-held tenferro engines at world size one. The lane
# count is explicit, not inferred from the launcher's core binding: one lane
# exercises the coordinator against a single lane, and two lanes exercise it
# against a pair.
launcher=${MPIEXEC:-mpiexec}
if command -v "$launcher" >/dev/null 2>&1 && command -v mpicc >/dev/null 2>&1; then
    bounded 30 cargo build -p hataori-tenferro --features mpi --example pmap_lanes_smoke
    bounded 30 cargo clippy -p hataori-tenferro --features mpi --all-targets -- -D warnings
    mpi_flags=(--oversubscribe)
    if ((EUID == 0)); then
        mpi_flags+=(--allow-run-as-root)
    fi
    if "$launcher" --version 2>&1 | grep -Eq 'Open MPI|OpenRTE'; then
        # Let the rank see every online CPU; the lanes are the rank's own
        # threads.
        mpi_flags+=(--bind-to none)
    fi
    run 60 env -u DISPLAY "$launcher" "${mpi_flags[@]}" -n 1 \
        target/debug/examples/pmap_lanes_smoke --lanes 1
    # A lane is a thread the adapter starts, not a pool worker the launcher
    # pins, so two lanes need no second CPU and no unbinding launcher.
    run 60 env -u DISPLAY "$launcher" "${mpi_flags[@]}" -n 1 \
        target/debug/examples/pmap_lanes_smoke --lanes 2
else
    echo "check-tenferro: MPI toolchain unavailable; skipped the pmap_lanes world-size-1 smoke" >&2
fi

bounded 30 git diff --check
echo "Hataori tenferro engine lane adapter checks passed"
