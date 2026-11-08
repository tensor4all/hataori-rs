#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
manifest=$root_dir/benchmarks/legacy-runner/Cargo.toml
lockfile=$root_dir/benchmarks/legacy-runner/Cargo.lock
baseline=34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT

[[ $(grep -c "rev = \"$baseline\"" "$manifest") == 1 ]]
[[ $(grep -c "#${baseline}\"" "$lockfile") == 1 ]]
"$root_dir/scripts/check-performance-manifest.py"

export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

build() {
    local name=$1
    local features=$2
    local -a args=(--release --locked --manifest-path "$manifest" --no-default-features)
    [[ -z $features ]] || args+=(--features "$features")
    cargo build "${args[@]}"
    cp "$root_dir/benchmarks/legacy-runner/target/release/hataori-legacy-runner" "$tmp_dir/$name"
}

check_records() {
    local output=$1
    local case_name=$2
    local repetitions=$3
    local items=$4
    local ranks=$5
    awk -F '\t' -v case_name="$case_name" -v repetitions="$repetitions" \
        -v items="$items" -v ranks="$ranks" -v baseline="$baseline" '
        BEGIN { seen = 0 }
        /^HATAORI_BENCH\t/ {
            if ($2 != "case=" case_name || $3 != "repetition=" seen ||
                $5 == "checksum=0" || $6 != "items=" items ||
                $9 != "ranks=" ranks || $14 != "warmups=1" ||
                $15 != "baseline_commit=" baseline) exit 1
            seen++
        }
        END { if (seen != repetitions) exit 1 }
    ' "$output"
}

run_local() {
    local binary=$1
    local case_name=$2
    local items=$3
    shift 3
    local output=$tmp_dir/local-$case_name-$RANDOM.log
    setsid timeout --signal=TERM --kill-after=2s 30s "$binary" "$case_name" \
        --items "$items" --payload-bytes 8 --work 2 --warmups 1 --repetitions 2 "$@" \
        >"$output" 2>&1
    check_records "$output" "$case_name" 2 "$items" 1
}

build serial ''
run_local "$tmp_dir/serial" map 4
invalid=$tmp_dir/invalid.log
set +e
setsid timeout --signal=TERM --kill-after=2s 10s "$tmp_dir/serial" map --unknown 1 >"$invalid" 2>&1
status=$?
set -e
if ((status == 0 || status == 124 || status == 137)) || grep -q '^HATAORI_BENCH' "$invalid"; then
    printf 'legacy runner invalid CLI check failed (status=%s)\n' "$status" >&2
    exit 1
fi

build rayon rayon
for mode in sequential outer inner; do
    run_local "$tmp_dir/rayon" map-in 4 --threads 1 --mode "$mode"
done

source "$root_dir/scripts/mpi-launcher.sh"
launcher=${MPIEXEC:-mpiexec}
command -v "$launcher" >/dev/null || {
    printf 'legacy runner local smoke passed; MPI launcher unavailable\n'
    exit 0
}
launcher_flags=()
if mpi_launcher_supports_oversubscribe "$launcher"; then
    launcher_flags+=(--oversubscribe)
    if ((EUID == 0)); then
        launcher_flags+=(--allow-run-as-root)
    fi
fi

run_mpi() {
    local binary=$1
    local ranks=$2
    local case_name=$3
    local items=$4
    local reported_items=$5
    shift 5
    local output=$tmp_dir/mpi-$case_name-$ranks-$RANDOM.log
    env -u DISPLAY setsid timeout --signal=TERM --kill-after=2s 30s \
        "$launcher" "${launcher_flags[@]}" -n "$ranks" "$binary" "$case_name" \
        --items "$items" --payload-bytes 8 --work 2 --warmups 1 --repetitions 2 "$@" \
        >"$output" 2>&1
    check_records "$output" "$case_name" 2 "$reported_items" "$ranks"
}

build mpi mpi
for ranks in 1 2; do
    run_mpi "$tmp_dir/mpi" "$ranks" pmap 4 4 --batch-size 2
    run_mpi "$tmp_dir/mpi" "$ranks" broadcast 3 3
    run_mpi "$tmp_dir/mpi" "$ranks" scatter 3 3
    run_mpi "$tmp_dir/mpi" "$ranks" gather 3 "$((3 * ranks))"
done

build hybrid mpi,rayon
run_mpi "$tmp_dir/hybrid" 2 pmap 8 8 \
    --threads 1 --mode sequential --batch-size 2 --prefetch true

printf 'immutable legacy performance runner smoke passed\n'
