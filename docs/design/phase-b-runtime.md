# Phase B long-lived runtime and structured execution

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Status:** implementation contract for Phase B. Phase A remains the transport
foundation; Phase C-E object, migration, and algorithm APIs remain unimplemented.

## 1. Ownership boundary

Phase B adds one unpublished `hataori-runtime` crate above
`hataori-runtime-foundation`. The lower crate continues to own identities,
parcels, bootstrap, and transport drivers. The runtime crate owns:

- lifecycle and owner-thread progress;
- immutable typed-action and domain registries;
- bounded pending-call, send-ticket, action, completion, cancellation, and
  deduplication state;
- `RemoteFuture`, structured remote scopes, deadlines, and statistics.

The public `Runtime` is not generic over a backend. It stores an object-safe
transport handle/driver pair. MPI communicators remain inside the Phase A MPI
driver, which is progressed only by the MPI main thread; no unsafe `Send` or
`Sync` implementation is introduced.

## 2. Construction and handshake

Actions must be registered before transport bootstrap because the Phase A
hello exchanges an action-registry fingerprint:

```rust
let mut builder = Runtime::builder(run_id, protocol_limits, runtime_limits)?;
builder.register::<Add, _>(|action| Ok(action.left + action.right))?;
let hello = builder.hello()?;
let (handle, driver) = TcpTransport::connect(membership, hello)?;
let mut runtime = builder.start(handle, driver)?;
```

Registration is keyed by stable nonzero `Action::ID` plus explicit input and
output schema IDs. The fingerprint sorts registrations by action ID and uses a
stable repository-owned non-cryptographic digest; it is a compatibility guard,
not authentication. Duplicate IDs and an empty/invalid schema are typed build
errors.

Phase B intentionally uses a small codec trait rather than adding a general
serialization dependency:

```rust
trait WireValue: Sized + Send + 'static {
    const SCHEMA_ID: u64;
    fn encode(self) -> Result<Vec<Vec<u8>>, ActionError>;
    fn decode(segments: Vec<Vec<u8>>) -> Result<Self, ActionError>;
}

trait Action: WireValue {
    const ID: u128;
    type Output: WireValue;
}
```

Codecs preserve segmented payloads. Registry dispatch is type-erased only
inside the runtime; callers receive `RemoteFuture<A::Output>`.

## 3. Lifecycle and progress

The checked state machine is:

```text
Created -> Bootstrapping -> Running -> Draining -> Stopped
                            |             |
                            +-----------> Failed
```

`Runtime::progress` is the one owner of the driver. It drains control/action
transport events, worker completions, deadlines, abandoned waiters, and expired
dedup entries in bounded batches. `Runtime::block_on` is a minimal owner-thread
future loop that polls the target future, calls bounded progress, and parks
briefly when no progress was made. It is not a second async executor.

Action handlers execute on fixed-size runtime-owned worker domains. Phase C
backs those domains with dedicated Rayon pools while retaining Phase B's
bounded admission and completion contract. Result capacity is sized from the
validated total worker/queue bound, so a worker never needs an unbounded
fallback. Handler panics are caught and converted to bounded typed action
failures; Rust does not forcibly stop a handler already running.

## 4. Requests, futures, scopes, and cancellation

`RuntimeClient::spawn_on` allocates a stable origin-scoped `RequestId`, inserts
one bounded pending promise, and then either submits locally or sends one
action parcel. Transport send completion only releases transport ownership; it
does not complete the future.

Every public spawn has a deadline. The default is configured in
`RuntimeLimits`; `spawn_on_with` accepts an explicit nonzero deadline and
optional trace ID. Deadline expiry removes the pending waiter, completes its
future with `DeadlineExceeded`, and sends best-effort cancellation.

`RemoteFuture` is `#[must_use]`. Dropping it removes the local pending waiter
immediately and sends best-effort cancellation. A validated late result is
counted and discarded; it never recreates a promise.

`Runtime::scope` uses a higher-ranked scope lifetime so scoped futures cannot
escape. The scope can spawn and `block_on` children. On scope exit it cancels
all still-tracked requests and pumps the cancellation queue once. This is the
Phase B structured remote-work contract. Explicit detached work remains
unimplemented rather than exposing an unbounded or fake API.

Cancellation is cooperative:

- a queued action observes its token and does not call user code;
- a running action may finish unless user code returns;
- result framing already announced to a transport is still drained;
- cancellation never claims that Rust user code was forcibly stopped.

## 5. Runtime action protocol

The first parcel segment is one fixed checked runtime header. It contains magic,
version, message kind, `RequestId`, `ActionId`, `DomainId`, and relative deadline
metadata. Remaining segments are action input/output data. Request/result/error
messages use the action channel; cancellation uses the control channel.

The decoder rejects wrong magic/version, unknown kind, zero IDs, malformed
header length, channel/kind mismatch, source/origin mismatch, unexpected
payload, and checked-length overflow before registry dispatch. Error text is
UTF-8 and capped at 4096 bytes.

Phase B does not add object IDs, directory traffic, leases, migration, or bulk
algorithm messages.

## 6. Deduplication and bounds

The receiver inserts `(origin, RequestId)` before queueing user code. Entries
are `Running`, `Completed`, or `Cancelled`:

- duplicate `Running` requests never execute again;
- duplicate `Completed` requests resend the cached response when retained;
- if the bounded result-byte budget could not retain a response, the completion
  marker remains and a duplicate receives `DuplicateResultUnavailable` rather
  than re-executing user code;
- cancellation before or during queueing leaves a bounded cancellation marker.

Completed/cancelled entries have a configured TTL. Running entries remain until
completion and are bounded by the dedup-entry/action-queue limits. A full table
returns typed resource exhaustion without executing the action. No eviction
silently permits duplicate execution inside the retained window.

`RuntimeLimits` validates nonzero bounds for pending calls, action queues,
workers/domains, dedup entries/bytes, progress events, default deadline, dedup
TTL, and shutdown timeout. Checked arithmetic derives total completion capacity.

## 7. Failure and shutdown

Protocol corruption, unknown actions/domains, remote user failures,
cancellation, deadlines, resource exhaustion, transport failure, and runtime
state errors are distinct variants.

Explicit shutdown:

1. atomically enters `Draining` and rejects new work;
2. removes pending waiters and sends best-effort cancellation;
3. progresses queued/running work until idle or the shutdown deadline;
4. stops and joins idle domain workers;
5. clears run-scoped dedup state;
6. flushes and stops the transport driver;
7. returns a report containing all retained runtime/transport counts.

A successful shutdown requires zero pending calls, action jobs, send-ticket
mappings, dedup entries, pending events, and retained transport bytes. A timeout
reports exact outstanding counts. `Drop` never performs an MPI barrier or waits
indefinitely; it only rejects new work, abandons local waiters, and detaches any
handler that user code has blocked. Correct programs call explicit shutdown.

## 8. Acceptance evidence

Phase B is accepted when bounded checks demonstrate:

- pure lifecycle, wire-header, registry-fingerprint, pending-table, deadline,
  scope, dedup, and queue-limit state machines;
- deterministic in-memory pass, duplication, delay/deadline, saturation,
  cancellation, late-result discard, handler failure/panic, and disconnect;
- the same typed-action request/result contract over TCP and MPI at world sizes
  1, 2, and 4, including reuse and explicit clean shutdown;
- control cancellation still progresses while action traffic is saturated;
- statistics and debug output expose counts/bytes without dumping payloads;
- default/MPI clippy `-D warnings`, rustdoc, Rust 1.85, runnable examples,
  rendered docs, and the unchanged P0/P1/Phase A acceptance lanes.

Every local validation process uses the short watchdogs in `AGENTS.md`.
