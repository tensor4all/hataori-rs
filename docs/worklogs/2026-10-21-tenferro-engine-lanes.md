# tenferro adapter: lanes over caller-held engines

Upstream changed the execution model under the adapter: tenferro-rs
[#2206](https://github.com/tensor4all/tenferro-rs/issues/2206) replaced sessions
with caller-held engines, deleted `CpuBackend`, `CpuHeldSession`, `CpuPhase`,
`PhaseLane`, `PhaseControl`, `SessionEntryError` and `CpuContext`, and stopped
building threads of any kind (merged as #2209, `5f0ae0a3`). The held-session
adapter could not be repaired; it was replaced.

## Chosen approach

The adapter now runs one lane per engine the caller hands it:

- `TenferroLanes::new(Vec<CpuEngine>)` — one lane per engine, in the caller's
  order, rejecting an empty list with `NoEngine`. `Lanes::Lane` is the engine
  itself, so a lane body gets `&mut CpuEngine`.
- `Lanes::run` spawns one scoped thread per engine and joins them in index
  order; `run_with_coordinator` spawns the same lanes and runs the coordinator
  on the calling thread.
- `close()` is gone: with sessions gone there is nothing to release except the
  engines, which the caller owns.

Two properties follow from the thread layout rather than from checks:

- a lane thread is never a worker of the engine pool it uses, which is
  tenferro's own precondition for issuing an operation, so no runtime
  pool-membership check is needed;
- a single lane can still overlap a coordinator, so `LanesError::SingleLane` is
  no longer produced and hybrid `pmap` takes the lane path at every width.

`Self::Error` is `LaneThreadError`, and it exists for exactly one case: a lane
thread that could not be started, after which the started lanes are cancelled and
joined. Everything else a run can report is a lane or coordinator failure, so the
tenferro error mapping table disappears with the vocabulary it translated.

## Alternatives rejected

- **A Hataori `Domain` (lane pool) plus `RayonLanes`.** `Domain::external` and
  `Domain::managed` both require a declared CPU set, which the adapter would
  have to invent; placement is the caller's decision, and inventing a CPU list
  in a library is wrong on any machine the caller did not describe.
- **Long-lived lane threads with a job protocol.** That is the phase/child-
  session machinery tenferro just deleted, moved into Hataori; it buys nothing
  over one scoped thread per run, because a run is a whole wave of items rather
  than one item.
- **`Mutex<CpuEngine>` per lane to satisfy the `Sync` closure bound.** Unneeded:
  engines are `Send`, so a scoped thread takes `&mut` to its own engine
  directly.

## Verification

`scripts/check-tenferro.sh` passes in full, including the `mpi` example: the
`pmap_lanes` smoke runs at world size one with `--lanes 1` and `--lanes 2`, and
an Open MPI launcher on one online CPU skips the two-lane stage as before.

Behaviour that the tests pin down, beyond the existing fixtures (known values,
lowest failing index, panic-then-reuse, peer release on coordinator return,
coordinator error/panic, lane error/unwind):

- each lane is lent engine `i` on its own thread, the engine pool reports
  `current_thread_index() == None` inside the lane, and a real `add_read`
  succeeds there;

A lane thread also gets tenferro's 16 MiB worker stack rather than std's 2 MiB,
because the provider code a lane body runs can recurse deeply on the lane thread.

An independent review raised two points that changed the implementation: a panic
during thread start used to leave started lanes waiting for a cancellation that
never came (now the thread is started fallibly, the lanes are cancelled and
joined, and the failure is reported as `LanesError::Executor`), and the lane
stack was std's default.
- a single lane overlaps a coordinator (barrier fixture), where the old
  held-session adapter returned `SingleLane`;
- an empty engine vector is `NoEngine`.

## Remaining

tensor4all-rs, `tenferro-benchmark` and `gaugefields-rs` still pin the previous
tenferro revision; each needs its own follow-up against this adapter and
against tenferro `5f0ae0a3`.
