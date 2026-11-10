# Pluggable payload codec for the MPI wire (issue #16, step 1)

**Scope:** hataori-rs #16 step 1. This splits the payload encoder out of the
framed MPI wire so the codec can be replaced without touching framing, and adds
`bitcode` as an opt-in evaluation backend. It keeps bincode as the default with
unchanged bytes and errors, and it does **not** add a data plane: large numeric
buffers still travel as one serialized payload.

## Why

`src/wire.rs` mixed three concerns: the 28-byte frame header, payload-length
bounds, and a hard-coded `bincode::serde` call. Evaluating a faster codec meant
editing the transport, and the same file owned both framing and encoding, so the
two could not be reviewed or changed separately.

## Chosen design

- `src/codec.rs` owns encoding only, behind `pub(crate) trait PayloadCodec`
  (`encode`/`decode`). Framing, the header, and `MAX_WIRE_BYTES` stay in
  `src/wire.rs`.
- `wire::encode_payload`/`wire::decode_payload` keep their signatures and
  behaviour: they delegate to the default backend and remain the only functions
  the scheduler and placement code call. `encode_payload_with`/
  `decode_payload_with` are the codec-parameterized forms used by tests and
  measurement.
- `Bincode` reproduces the previous configuration exactly (fixed-size integers,
  little-endian, full byte consumption checked with `TrailingBytes`), so wire
  bytes and error variants are unchanged.
- `Bitcode` is behind the new `bitcode` feature, which requires `mpi` or
  `rsmpi-rt`; it cannot silently do nothing, and it never enters a default
  build. The codec is chosen at compile time because both ends of an exchange
  must agree on it.
- The MPI path still selects bincode. Picking a codec per exchange, and the
  direct numeric data plane, are later steps of #16.

## Verification

- `codec::tests` round-trips a small struct, a nested value, empty values, and a
  10 000-element `Vec<u64>` through both backends, and rejects truncation,
  trailing bytes, and garbage for both.
- The measurement harness is an `#[ignore]`d test in `src/codec.rs`; it is not
  part of the test suite. It is not wired into CI.
