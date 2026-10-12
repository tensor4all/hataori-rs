# Phase E algorithms, facade, and policy

**Status:** Proposed implementation contract; implementation must not start until
its recorded design review gate passes

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Depends on:** Phase B typed actions/futures, Phase C placement, Phase D
migration, and the immutable Phase A performance manifest

## 1. Scope

Phase E adds algorithms above the runtime without changing transport or object
lifetime ownership:

- a canonical bounded batch controller for initiator-only ordered `pmap`;
- action/future-based broadcast, scatter, and gather;
- exact-place and hard/preferred object-colocated batch placement;
- current-style async and blocking facade entry points that call the canonical
  controller directly;
- same-locality typed action dispatch without codec or transport work;
- candidate and paired-experiment tooling for the frozen performance manifest.

The existing P0/P1 implementation remains available until every frozen
legacy-continuity and sugar-transparency case is `PASS`. Phase E does not add
automatic migration: no measured workload currently justifies policy state,
and Issue #14 explicitly defers it until one does.

## 2. Ownership and crate boundary

A new unpublished `hataori-algorithms` crate depends on `hataori-runtime` and
owns batch planning, submission, ordered collection, collective aggregation,
and algorithm errors. It owns no executor, transport, pending table, resolver,
retry state, or lease.

The top-level `hataori` crate exposes the new facade behind a `runtime` feature.
Its facade functions are re-exports or one-call blocking adapters; they do not
wrap results in another scheduler or state machine. The old MPI collective API
continues to compile when `runtime` is disabled and is not removed by this
phase.

## 3. Registered batch actions

`BatchActionToken<A>` is a typed zero-sized descriptor for an item action `A`.
`A` uses the existing stable `Action::ID`, `WireValue` input, and
`Action::Output`. `AlgorithmRegistryExt::register_pmap_action` installs one
batch wrapper action derived mechanically from `A::ID`; registry collision and
schema checks remain fail-closed in `RuntimeBuilder`.

One batch parcel contains:

```text
batch header: version, start index, local mode, item count, nested segment counts
item segments: original WireValue segments without concatenating payload bytes
```

The response contains the same start index and one ordered success value or
bounded error string per item. Checked counts, lengths, integer conversions,
and a maximum error-message size are enforced before allocation. A callback
failure is data in the successful batch response so the controller can select
the globally lowest failing input index deterministically. Codec, transport,
peer, deadline, and runtime failures remain typed runtime failures.

The batch handler executes `Sequential` and `Inner` batches serially. `Outer`
uses Rayon from the currently executing runtime-owned domain; it does not
construct a pool or use a second scheduler. Results remain ordered.

## 4. Canonical `pmap`

`PmapOptions` validates nonzero batch size and bounded per-target depth.
Recognizable fields are:

- `batch_size`;
- `local_mode: LocalMode::{Sequential, Outer, Inner}`;
- boolean `prefetch`, selecting one admitted batch per target when false and
  two when true;
- one action deadline.

The controller allocates the input index plan and one `Vec<Option<U>>` ordered
result table once. It creates one owned batch and one runtime future per
in-flight batch, never per item unless `batch_size == 1`. Initial slots are
spread over eligible localities; when a slot completes, its locality receives
the next batch, preserving bounded dynamic scheduling without a coordinator
queue outside the controller. Pending calls are therefore bounded by
`eligible_targets * (1 + prefetch)` and by runtime limits.

The initiator alone calls `pmap` and receives `Vec<U>`. Service localities only
run registered actions. Empty input returns immediately. Runtime failures drop
outstanding futures, using existing cancellation. Callback failures are fully
collected and the lowest input index is returned.

## 5. Placement

The canonical controller has three submission strategies sharing the same
planning and collection code:

1. `Any`: one or two slots per runtime locality in membership order;
2. exact `Place`: one or two slots at that place;
3. hard or preferred colocation: each batch calls the existing
   `spawn_colocated` or `spawn_preferred_colocated` path.

Object placement is resolved and accounted once per batch. No item creates a
future, resolver lookup, handle, lease, placement ticket, or allocation.
Migration redirects reuse the runtime's existing request/epoch contract. Soft
placement uses the caller's explicit bounded fallback and never changes queue
priority.

## 6. Same-locality typed fast path

The current runtime encodes every ordinary typed action before deciding whether
it is local. Phase E adds a typed local action path at the owning action/runtime
boundary:

- action registration stores both the existing segment codec handler and a
type-erased checked local move handler;
- local requests/jobs/completions carry either checked segments or one owned
typed value;
- local `spawn_on` moves the action into the domain and downcasts the typed
  output into `RemoteFuture<T>` without invoking `WireValue::encode/decode`;
- remote and already-encoded internal operations retain the segment path.

The same pending promise, domain admission, cancellation, completion, and
future are used. This is not a second scheduler or promise table. Runtime stats
count typed local dispatches and action codec calls so tests and benchmarks can
prove zero local serialization and zero transport messages.

## 7. Collectives

`AlgorithmRegistryExt::register_collective::<T>()` registers one typed echo
action for `T`. The algorithms are initiator-only:

- broadcast clones one value into one action per logical locality and gathers
  ordered acknowledgements;
- scatter requires exactly one value per logical locality and sends each value
  once;
- gather consumes an ordered vector of existing runtime futures and returns one
  ordered vector.

These are reference parcel semantics over actions/futures. They add no native
MPI collective requirement and work unchanged over memory, TCP, and MPI.
Membership size bounds retained futures through the runtime's fixed bounded
membership and pending limits.

## 8. Facades

The async top-level facade exports the canonical types and functions directly.
Local `map` and `map_in` keep their scoped borrowed-closure implementation.
Distributed `pmap` accepts only a registered `BatchActionToken`; it never
claims to serialize a closure.

`hataori::blocking` takes `&mut Runtime`, constructs the same owned canonical
future, and calls the existing `Runtime::block_on`. It adds no executor or
queue. Runtime-worker re-entry is structurally rejected by the runtime owner
contract and remains a typed runtime error if that contract is violated.

## 9. Errors and observability

`AlgorithmError` distinguishes invalid options, input/member shape, callback
failure with the lowest input index and bounded message, and wrapped
`RuntimeError`. Controller statistics expose item count, batch count,
peak in-flight batches, and placement submissions. Runtime statistics expose
local typed dispatches and action encode/decode counts.

No algorithm table or cache survives its future. Dropping the controller drops
all batch futures and owned yet-unsubmitted inputs. Successful completion leaves
no pending promise, placement ticket, resolver pin, or retained batch buffer.

## 10. Performance evidence

A candidate runner emits the same raw TSV identity fields as the immutable
legacy runner plus candidate commit, facade/canonical mode, backend, and
instrumentation counters. A standard-library paired orchestrator:

- verifies the frozen manifest digest and fixed baseline commit;
- records every required host observation before running a candidate;
- expands selected manifest matrices fail-closed;
- runs 30 baseline-only calibration pairs and 30 alternating paired
  baseline/candidate repetitions with no retries or outlier removal;
- computes the frozen log-ratio point estimates and one-sided 95% bounds;
- rejects missing/malformed records and classifies host/noise failures as
  `INCONCLUSIVE`;
- checks correctness, boundedness, retained resources, and zero-overhead
  counters;
- writes a complete machine-readable report tied to one exact candidate commit.

Local smoke checks validate runner/orchestrator mechanics with tiny repetitions;
they are not performance evidence. Promotion or deletion of P0/P1 requires a
full valid-host report in which every frozen required case is `PASS`. If the
available host is invalid or cannot execute a required matrix, implementation
may be complete but Phase E acceptance remains blocked rather than weakening
or relabeling the gate.

## 11. Verification

Required deterministic checks are:

- batch codec round trips segmented values and rejects malformed counts/lengths;
- ordered empty/single/multiple-batch `pmap`, bounded in-flight depth, callback
  error selection, dropped future cancellation, and saturated admission;
- all three local modes on runtime-owned domains;
- same-locality zero codec and zero transport counters;
- remote memory/TCP/MPI ordered execution at 1/2/4 localities and at least one
  segmented payload larger than 1 MiB;
- exact, hard-colocated, preferred fallback, migration redirect, and
  per-batch resolver/future accounting;
- broadcast/scatter/gather on memory, TCP, and MPI;
- async facade versus canonical counter equality and blocking facade reuse;
- clean reusable shutdown with zero pending calls and retained algorithm work;
- Rust 1.85, rustfmt, clippy `-D warnings`, rustdoc, docs-site build, and Phase
  A-D regressions under repository watchdogs.

## 12. Non-goals

Phase E does not add automatic migration, per-item affinity, barriers,
reductions, a task graph, dataflow, native transport collectives, a new async executor, dynamic
membership, persistence, replication, topology solving, TLS, or crash recovery.
It does not remove P0/P1 before the frozen performance gate passes.
