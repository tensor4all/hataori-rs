# Direct data plane for contiguous `f32`/`f64` buffers (issue #16, slice A)

**Scope:** hataori-rs #16 item 2, primitive-buffer half only. One contiguous,
one-dimensional `f32`/`f64` buffer can cross MPI as a raw typed buffer instead of
a serde-encoded `Vec<u8>`. Tensor adapters and the downstream-defined adapters
(item 3) are not part of this slice, and nothing in the public facade uses the new
path yet.

## What changed

- `src/wire.rs` gains `MessageKind::Direct` (code 8), `ElemType`,
  `ByteOrder`, `DirectMeta`, `encode_direct_control`, `Header::new_direct`, and
  `Header::decode_control`. Framing, `Header::encode`/`decode`, the existing
  kinds, and their byte layout are untouched: pmap and placement still send the
  28-byte header and the coded payload are unchanged; `pmap` keeps its 0/1 tag
  pair, and placement keeps 2/3, which direct frames also use.
- `src/direct.rs` gains the transport: `send`/`receive` over one typed MPI
  message, with a `DirectElem` impl for `f32` and `f64`.
- `scripts/check-faults.sh` launches the MPI `#[test]` for both backends at
  `-n 2` and `-n 3`, reusing its existing test-binary selection, launcher flags,
  and watchdog instead of a second copy of that machinery. That script is
  already called by `scripts/check-core.sh`, so the hosted core lane reaches the
  new cases without a further CI change. Its MPI launches now use a `60s`
  watchdog with `--kill-after=2s`, matching the repository's MPI bound; the
  script's `cargo test --no-run` build step is not covered by that watchdog.

## Decisions

- **Descriptor in the control message, not the data message.** The data message
  must remain exactly the caller's buffer, so the element type and the byte-order
  marker travel in two bytes appended to the header of `Direct` frames only.
  `Header::decode_control` requires the kind and the control length to agree, so
  no framed kind can grow a descriptor and a `Direct` frame cannot omit one.
- **Native byte order with a marker.** The data plane copies memory unchanged and
  records the sender's host order in byte 29. A receiver rejects a mismatching
  peer before allocating instead of decoding it silently wrong; mixed-endian
  peers use the codec path, which is explicitly little-endian.
- **Empty buffer sends no data message.** A zero-element send emits the 30-byte
  control message only, and the receiver returns an empty `Vec` without posting a
  data receive. This is the case that would otherwise leave an unmatched message
  or a zero-length receive.
- **Crate-private, dead-code-allowed, for now.** Adding public API is a
  maintainer decision and the issue's adapter trait is a later slice, so `send`,
  `receive`, and `DirectElem` are `pub(crate)` and the module carries a
  `dead_code` reason naming the slice that will use it.
- **MPI tests follow the existing pattern.** Like
  `src/pmap/mpi_fault_tests.rs`, the test no-ops unless `HATAORI_DIRECT_PAYLOAD`
  is set, and `scripts/check-faults.sh` launches it through the MPI launcher. A
  test that spawned `mpirun` from inside `cargo test` would add MPI launches to
  every row of the matrix.
- **A rejected frame is drained.** If the control message fails validation after
  the sender already sent its data message, the receiver still receives and
  discards that message before returning the error, so a blocking sender
  completes and the next exchange is not poisoned. The test covers this: a
  mismatched element type is rejected, and the frame sent after it arrives
  intact. Draining requires a length-consistent `Direct` frame with a readable
  element-type byte and a probed data count equal to `item_count`; anything else
  (undecodable header, invalid type byte, disagreeing count) means the wire
  formats disagree or the stream is corrupt; the rejection is then reported as
  `DirectDrainFailed` (or `DirectCountMismatch` from the drain probe) instead of
  the original recoverable error, and the exchange is fatal.
- **Counts and sizes are probed before allocating.** Both the control message and
  the data message are probed, so an unexpected control size is rejected without
  allocating it and the control message goes into a stack buffer, and a short or
  oversized data message becomes a typed `DirectCountMismatch` instead of leaving
  zero-filled values or aborting the job with an MPI truncation error.
- **The framed tag pairs differ.** `pmap` frames use tags 0/1; placement frames
  and direct frames use 2/3, so a direct exchange owns its communicator.

## Verification

- Red first: the round-trip test was written before `src/direct.rs` existed and
  failed with 40 compile errors naming the missing kind, descriptor, and
  transport.
- `cargo test --no-default-features --features mpi` (63 passed, 1 ignored)
  includes the unmodified `wire::tests::header_golden_and_roundtrip` and the new
  control-plane byte-identity test.
- `scripts/check-faults.sh /tmp/mpiwrapper-install/lib/libmpiwrapper.so` passes
  for `mpi` and `rsmpi-rt` at `-n 2` and `-n 3`, including the direct round trip.
- The allocation table in `docs/design.md` §10.1 comes from the `#[ignore]`d
  `direct_payload_measurement` test with a counting allocator, run on one test
  thread; it covers the Hataori-side encode window, not MPI internals.

## Deferred

- Item 3's adapter trait for downstream buffer types (`tenferro`, `ndarray`,
  sparse matrices, several buffers per message).
- Any caller: `pmap` task/result payloads and placement collectives still use the
  codec path, so the direct frame kind is currently exercised only by tests.
- Throughput: not measured. Loopback MPI timing on this shared host is not a
  measurement, and the repository's benchmark gates own timing claims.
