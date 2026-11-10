#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 1)); then
    printf 'usage: %s ABSOLUTE_MPIWRAPPER_LIBRARY\n' "$0" >&2
    exit 2
fi
mpi_rt_lib=$1
[[ $mpi_rt_lib = /* && -f $mpi_rt_lib ]] || {
    printf 'check-faults: MPIwrapper path must be an existing absolute file\n' >&2
    exit 2
}

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
cd "$root_dir"
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

source "$root_dir/scripts/mpi-launcher.sh"
launcher=${MPIEXEC:-mpiexec}
launcher_flags=()
if mpi_launcher_supports_oversubscribe "$launcher"; then
    launcher_flags+=(--oversubscribe)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi

build_test_binary() {
    local backend=$1
    local json=$tmp_dir/$backend.json
    cargo test --no-run --no-default-features --features "$backend" --message-format=json >"$json"
    python3 - "$json" <<'PY'
import json
import sys

paths = []
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        target = record.get("target", {})
        profile = record.get("profile", {})
        if (
            record.get("reason") == "compiler-artifact"
            and target.get("name") == "hataori"
            and "lib" in target.get("kind", [])
            and profile.get("test") is True
            and record.get("executable")
        ):
            paths.append(record["executable"])
if len(paths) != 1:
    raise SystemExit(f"expected one hataori unit-test executable, got {paths!r}")
print(paths[0])
PY
}

run_exact() {
    local backend=$1
    local count=$2
    local test_name=$3
    shift 3
    local output=$tmp_dir/$backend-${test_name##*::}.log
    local -a environment=(env "$@")
    if [[ $backend == rsmpi-rt ]]; then
        environment+=(MPI_RT_LIB=$mpi_rt_lib)
    fi
    if ! setsid timeout --signal=TERM --kill-after=2s 60s \
        "${environment[@]}" "$launcher" "${launcher_flags[@]}" -n "$count" \
        "$binary" --exact "$test_name" --nocapture --test-threads=1 >"$output" 2>&1; then
        cat "$output" >&2
        printf 'check-faults: %s %s failed\n' "$backend" "$test_name" >&2
        return 1
    fi
}

for backend in mpi rsmpi-rt; do
    binary=$(build_test_binary "$backend")
    marker=$tmp_dir/$backend-marker
    run_exact "$backend" 4 \
        pmap::mpi_fault_tests::simultaneous_user_and_decode_faults_drain_and_reuse \
        HATAORI_FAULT_MARKER="$marker"
    run_exact "$backend" 1 pmap::mpi_fault_tests::size_one_root_paths_bypass_codec \
        HATAORI_CODEC_TEST=1
    run_exact "$backend" 2 direct::tests::direct_payload_roundtrip \
        HATAORI_DIRECT_PAYLOAD=1
    run_exact "$backend" 3 direct::tests::direct_payload_roundtrip \
        HATAORI_DIRECT_PAYLOAD=1
done

printf 'fault, codec-bypass, and direct data-plane checks passed for mpi and rsmpi-rt\n'
