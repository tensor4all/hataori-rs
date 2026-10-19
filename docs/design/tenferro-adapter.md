# Phase 20a: tenferro adapter

**Status:** Implemented (commit `c2cb8fb`, PR #2); this document records the design as reviewed.

**Scope:** A tenferro-only adapter over Hataori's existing whole-domain `Inner`
execution contract. tensor4all contexts, tensor transport, and tensor
reconstruction remain Phase 20b and are not part of this change.

**Upstream contract:** [`tenferro-rs` PR #1717](https://github.com/tensor4all/tenferro-rs/pull/1717), merge commit `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`.

## 1. Crate and dependency boundary

Add one non-publishable workspace member, `hataori-tenferro`, instead of
binding tenferro's backend inside the `hataori` package. (The root package
separately has an opt-in `tenferro` feature that adds only `tenferro-tensor`
for the Mandelbrot examples; it is not part of the adapter contract.) The
adapter depends on:

- local `hataori` with `rayon` enabled;
- `rayon = "1.10"` directly, for the `ThreadPool` handle it inspects;
- `tenferro-cpu` and `tenferro-tensor` at the exact upstream merge revision;
- no tensor4all crate and no MPI backend.

The root manifest becomes a package-plus-workspace whose `members` list gains
`adapters/tenferro`, with `default-members = ["."]` and resolver 2.
Consequently ordinary root `cargo build`/`cargo test` still select only
`hataori`. The root package remains dependency-free by default and its
Rust 1.85 feature matrix (default, `rayon`, `mpi`, `rsmpi-rt`, and hybrids)
remains free of tenferro, including a fresh Rust 1.85 check with the workspace
member present. The root package's opt-in `tenferro` example feature is outside
that matrix: it pulls `tenferro-tensor` and therefore also needs Rust >= 1.96;
`scripts/check-tutorial-examples.sh` builds and runs the Mandelbrot examples
with the `tenferro` feature. The adapter follows the pinned tenferro
revision's Rust 1.96 MSRV and has a separate current-stable CI/gate invocation.
A crates.io release is out of scope while its upstream dependency is a git
revision.

## 2. Public API

The adapter exposes one binding and one typed error:

```rust
pub struct TenferroDomain { /* private */ }

impl TenferroDomain {
    pub fn new(domain: hataori::Domain)
        -> Result<Self, TenferroAdapterError>;

    pub fn domain(&self) -> &hataori::Domain;

    pub fn with_backend<R>(
        &self,
        operation: impl FnOnce(&mut tenferro_cpu::CpuBackend) -> R,
    ) -> Result<R, TenferroAdapterError>;
}
```

`TenferroDomain::new` consumes one Rayon-backed Hataori domain. Ownership makes
one adapter entry gate canonical for that domain; two independently guarded
adapters cannot be constructed from the same `Domain`. Callers pass
`adapter.domain()` to `map_in`/`pmap`. Construction failure drops the supplied
domain normally and leaks no pool or backend resource.

Construction:

1. requires a Rayon-backed Hataori domain;
2. retains the exact `Arc<rayon::ThreadPool>` already owned or retained by that
   domain;
3. maps `hataori::DomainId` losslessly to `tenferro_tensor::CpuDomainId`;
4. uses the domain's complete nonzero `worker_count` as the tenferro thread
   budget;
5. constructs `RayonCpuDomainExecutor`,
   `ExternalCpuDomain::new_caller_managed`, and
   `CpuBackend::from_external_managed_domains` from the pinned public tenferro
   API.

No second pool, executor registry, global backend, or Hataori-specific tenferro
type is introduced.

`with_backend` is the scoped entry used inside a Hataori `Inner` callback. It
checks that the calling thread belongs to the bound Rayon pool before cloning the
cheap `CpuBackend` handle and invoking the borrowed operation. The clone shares
tenferro's per-domain active-entry flag and does not create a runtime, pool, or
provider thread. The operation and result need not be `'static`.

The Hataori admission is held by `map_in`/`pmap`, but their established callback
shape intentionally does not expose `DomainAdmission`. Therefore the adapter
does not pretend that an atomic observation is a non-aliasing admission witness.
Calling `with_backend` from an admitted `LocalMode::Inner` callback is a safe API
contract, not an unsafe precondition. Pool membership is checked locally. A
caller that directly schedules unrelated work onto a caller-owned pool is
already outside Hataori's admission contract.

Exact-revision runtime testing found that tenferro's shared caller-managed flag
panics with `BACKEND_REENTRY_PANIC` on collision rather than returning a typed
error. The domain-owning adapter therefore owns one nonblocking RAII `AtomicBool`
entry gate.
It is acquired only around `with_backend`, returns typed `ConcurrentEntry` before
the user closure on collision, and releases on success, error return, or unwind.
This is an API panic firewall, not capacity arbitration or coarse admission: it
never waits, never enters `ResourceArbiter`, and does not replace Hataori's
running slot. tenferro's own flag remains the final guard against a caller who
clones and recursively enters the supplied backend inside one operation.

The core adds one narrow read-only integration observation required by the
separate adapter: cloning the exact selected Rayon pool `Arc`. The accessor grants
no Hataori admission and its rustdoc states that direct pool submission bypasses
Hataori scheduling. This deliberate public seam is the cost of keeping tenferro
out of the core crate; no admission-state reader, TLS marker, executor trait, or
global registry is added.

## 3. Admission and fan-out ownership

The required call shape is:

```text
Hataori map_in/pmap(LocalMode::Inner)
  -> Hataori DomainAdmission (sole coarse admission)
    -> one callback on the selected Rayon pool
      -> TenferroDomain::with_backend
        -> tenferro caller-managed per-domain public-entry guard
          -> same Rayon pool for provider-owned inner fork/join
```

Hataori's running slot is the only cross-operation/coarse admission mechanism.
At exact tenferro commit `a21a4c6`, `CpuBackend::acquire_execution_permit`
matches `ResolvedCpuExecution::ExternalCallerManaged` and constructs
`ResourcePermit::caller_managed` directly from the domain's shared active flag;
it does not call any `ResourceArbiter::acquire*` method or wait for capacity.
The adapter's local entry gate likewise never waits and carries no CPU-set or
capacity policy.

`LocalMode::Outer` is not a supported adapter entry mode, but the adapter cannot
pre-identify the mode from the unchanged callback signature. Concurrent calls
are rejected by the adapter's typed `ConcurrentEntry` gate before a backend
clone or user closure is produced; they cannot reach tenferro's panic or create
multiple simultaneous provider entries. Rustdoc and examples use only `Inner`,
and a regression test makes this behavior explicit. The adapter does not change
`LocalMode` or add a partial worker budget.

## 4. Provider and placement policy

The adapter uses tenferro's standard caller-managed constructor with
`tenferro-cpu` default features disabled and only `cpu-faer` enabled. Independently,
at exact commit `a21a4c6`, `external_domain_backend_kind` forces
`CpuBackendKind::Faer` whenever any descriptor has
`CpuAdmissionMode::CallerManaged`, rather than using `default_compiled`.
Construction then validates the provider bundle before returning the backend. A
provider with external workers, uncontrolled thread count, or no caller-managed
placement control is rejected by tenferro's typed provider validation. The
adapter neither duplicates that capability matrix nor manipulates environment
thread counts.

Hataori remains the source of its managed/external affinity status. The generic
tenferro Rayon executor reports no independent affinity claim and caller-owned
shutdown, as specified upstream. Dropping the adapter never shuts down the
Hataori pool.

## 5. Errors

`TenferroAdapterError` preserves typed sources for:

- missing Hataori Rayon pool;
- tenferro external-domain construction;
- tenferro backend construction;
- entry from outside the bound domain pool;
- simultaneous or recursive adapter entry (`ConcurrentEntry`).

Construction and entry failures run no user operation. Error and unwind paths
release the upstream tenferro public-entry guard; Hataori's existing RAII guard
releases coarse admission when `map_in` or `pmap` leaves its scope.

## 6. Acceptance evidence

The implementation must leave one checked entry point that proves:

- default and `rayon` Hataori dependency trees contain no tenferro or tensor4all;
- the adapter tree contains the exact tenferro revision and no tensor4all crate;
- adapter construction reports caller-managed admission, full worker budget,
  caller-owned executor shutdown, no fabricated placement, and Faer selection;
- a known-value tenferro tensor operation runs from `LocalMode::Inner` on only
  the supplied Rayon team's named workers;
- a pool `start_handler` count remains exactly the declared worker count, a
  global-pool sentinel remains untouched, and observed worker names all belong
  to the supplied team; upstream exact-revision tests separately exercise all
  workers inside tenferro's native parallel region;
- entry from a foreign thread is typed and runs no operation;
- Hataori contention remains nonblocking, and success, callback error, and
  unwind all permit later reuse;
- incompatible `Outer` fan-out returns typed `ConcurrentEntry`, runs only one
  closure, never reaches tenferro's collision panic, and allows later reuse;
- borrowed non-`'static` input/output state compiles and runs;
- `cargo fmt`, tests, all-target clippy with `-D warnings`, rustdoc/doctests,
  dependency-boundary checks, and `git diff --check` pass.

MPI transport and tensor serialization are unchanged. Phase 20b owns explicit
tensor4all contexts and logical tensor reconstruction after
`tensor4all-rs#663`; Phase 20c owns the three-repository MPI integration gate.

## 7. Follow-up: lanes instead of the external-domain chain

This document records the implemented Phase 20a state, which is pinned to a
tenferro revision that still accepts a caller-managed external domain. Upstream
tenferro-rs #2004 removed borrowing an external pool, so a follow-up rewrite
keeps tenferro as the sole owner of its pool and admission and binds the two
layers through the core lanes contract instead (`docs/design.md` §6.1,
[`tenferro-rs#2201`](https://github.com/tensor4all/tenferro-rs/issues/2201)):
Hataori schedules items onto tenferro phase lanes (`CpuPhase::run` /
`run_with_coordinator`), and the adapter maps tenferro's `PhaseControl` into
Hataori's `LaneControl`. `LocalMode::Outer` then becomes a supported mode
because each lane carries its own owner-inheriting child session. Nothing in
this section is implemented in the adapter yet.
