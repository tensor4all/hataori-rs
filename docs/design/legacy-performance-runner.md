# Legacy performance runner

## Status

Implemented Phase A-1 design. The completed pre- and post-implementation review
gates are recorded in `docs/review-log.md`.

## Purpose

Freeze a buildable runner for the immutable P0/P1 performance reference
`34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e`. The runner gives later paired
experiments one stable way to execute legacy `map`, `map_in`, `pmap`, and
placement-helper workloads without checking out or modifying the reference
commit.

This runner produces no performance claim. The complete case manifest, host
validity gates, repetitions, statistics, and acceptance thresholds are frozen
in `benchmarks/performance/manifest.toml` before any candidate measurement.

## Location and source pin

Add one standalone crate at `benchmarks/legacy-runner/`. Its manifest declares
its own `[workspace]`, so it does not enter Hataori's normal workspace or CI
feature matrix. It depends on Hataori through:

```toml
hataori = {
    git = "https://github.com/tensor4all/hataori-rs.git",
    rev = "34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e",
    default-features = false,
}
```

Runner features forward only `rayon` and `mpi` to that dependency. The MPI
feature also uses the same upstream `mpi = 0.8.1` API required to initialize
MPI. The runner adds no benchmark framework or async runtime.

Commit identity is checked twice:

1. `Cargo.toml` contains the full accepted revision;
2. the checked `Cargo.lock` resolves the Hataori git package to that same full
   revision.

A verification script fails if either value differs. Updating the baseline
requires a new accepted design change; ordinary dependency updates must not
move it.

## Executable contract

One binary, `hataori-legacy-runner`, accepts exactly one case per invocation:

```text
hataori-legacy-runner CASE [OPTIONS]
```

Initial case names are:

- `map`: serial `hataori::map`;
- `map-in`: `hataori::map_in` with an explicit managed Rayon domain;
- `pmap`: collective `hataori::pmap` with root-owned input;
- `broadcast`, `scatter`, and `gather`: the corresponding placement helpers.

The runner uses a small hand-written argument parser. Unknown options,
duplicate options, missing values, zero counts, unsupported feature/case
combinations, and invalid MPI or thread settings fail before the timed region.
No general configuration abstraction is introduced.

Shared options are limited to values needed by the final manifest:

- `--items N`;
- `--payload-bytes N`;
- `--work N`, a deterministic integer-loop count per item;
- `--repetitions N`;
- `--warmups N`.

Rayon cases additionally accept `--threads N` and
`--mode sequential|outer|inner`. `pmap` additionally accepts `--batch-size N`
and `--prefetch true|false`. MPI process count remains launcher-owned rather
than being duplicated in runner arguments.

Each invocation constructs its domain once. Because the APIs consume owned
input, it prepares fresh equivalent input outside the timed region before each
warmup and repetition. Setup, MPI initialization, and shutdown are outside
operation timing; the frozen manifest requires them to be recorded separately.

## Workload and correctness

All cases use deterministic integer work and byte payloads generated from the
item index and rank. Inputs are value-dependent and every measured result is
consumed into a deterministic 64-bit wrapping checksum. The checksum and
expected item/byte counts are validated before a repetition is reported.
This prevents a wrong-but-shaped or optimized-away workload from becoming
benchmark evidence.

Same-locality inputs remain owned values. The legacy runner does not add
serialization, copies, threads, barriers, or queues around the operation being
measured. MPI ranks execute the same collective sequence; only rank zero emits
records.

## Output

Each successful measured repetition prints one machine-readable TSV record to
stdout with a literal `HATAORI_BENCH` prefix and fixed keys:

```text
HATAORI_BENCH	case=...	repetition=...	elapsed_ns=...	checksum=...	items=...	payload_bytes=...	work=...	ranks=...	threads=...	mode=...	batch_size=...	prefetch=...	warmups=...	baseline_commit=34cb1b...
```

Values are decimal integers or closed ASCII tokens, so escaping is unnecessary.
Diagnostics go to stderr. Partial, malformed, duplicate, or nonzero-exit runs
are invalid evidence; a later orchestrator fails the whole experiment rather
than retaining successful repetitions.

The runner reports raw observations only. It does not compute confidence
intervals, classify PASS/FAIL, retry cases, discard outliers, inspect host
noise, or choose thresholds.

## Verification

Add `scripts/check-legacy-runner.sh` with bounded smoke coverage:

1. verify the manifest and lockfile contain the exact baseline revision;
2. build the serial runner in release mode and run a small `map` case;
3. build with Rayon and run small `map-in` cases for all three modes;
4. when `mpiexec` is available, build with MPI and run `pmap`, `broadcast`,
   `scatter`, and `gather` at world sizes one and two;
5. validate record count, fixed prefix, baseline revision, nonzero checksum,
   expected item counts, and ordered repetition numbers;
6. prove one invalid CLI invocation exits nonzero without a benchmark record.

Open MPI smoke launches add `--oversubscribe` and, only when running as root,
`--allow-run-as-root`, matching existing repository scripts. Every subprocess
has a process-group timeout. Absence of MPI may skip only the local developer
smoke lane; hosted CI installs MPI and runs the complete script.

## Non-goals

This slice does not:

- benchmark the current or future runtime;
- run baseline-versus-baseline calibration;
- implement paired interleaving or statistical analysis;
- add TCP, segmented parcels, remote objects, migration, or observability;
- copy the existing π or Mandelbrot examples into another benchmark suite;
- modify P0/P1 source or its measured fast paths.

## Frozen experiment manifest

`benchmarks/performance/manifest.toml` is the machine-readable source of truth
for the complete case families, host validity observations, paired statistics,
noise-derived tolerance capped at 2%, acceptance thresholds, instrumentation,
and predeclared TCP targets. `scripts/check-performance-manifest.py` fails
closed if required coverage or a hard threshold is removed. Candidate results
must not modify this manifest; every report records its SHA-256.

The manifest deliberately contains future-runtime cases which the legacy runner
cannot execute. Their presence freezes their acceptance contract; it does not
claim those facilities are implemented. A later experiment orchestrator must
refuse candidate execution unless it can execute every case selected for the
candidate's declared comparison lane.

Protocol/runtime implementation may start only from a commit containing both
this checked manifest and the immutable runner.
