# API Reference

The rustdoc for the `hataori` crate is published alongside this site:

- [`hataori` crate documentation](hataori/index.html)
- [`hataori-runtime` Phase B-D documentation](hataori_runtime/index.html)

It is built with `--features mpi,rayon`, so the MPI, hybrid, and Rayon items
are all visible; feature-gated items are annotated with the feature that
enables them. The `rsmpi-rt` backend exposes the same items as `mpi`.

## Public surface by model

| Model | Items |
| --- | --- |
| Serial | `map`, `MapError` |
| Rayon | `map_in`, `MapInError`, `allowed_cpus`, `Domain::managed`, `Domain::external`, `PoolOwnership`, `PlacementStatus`, `DomainBuildError` |
| MPI and hybrid | `pmap`, `PmapOptions`, `PmapError`, `PmapErrorKind`, `Domain::sequential` |
| Placement | `broadcast`, `scatter`, `gather`, `PlacementError`, `PlacementErrorKind` |
| Common | `Domain`, `DomainId`, `LocalMode`, `Place`, `DomainAdmission`, `DomainBusy`, `NegativeRank`, `UnsupportedDomainId` |
| Phase B-D runtime (unpublished) | `Runtime`, `RuntimeClient`, `Action`, `WireValue`, `RemoteFuture`, `RuntimeScope`, `DistributedObject`, `Remote`, `ObjectRoot`, `WeakRemote`, `MobileObject`, `MigrationFuture`, typed object actions, limits/stats |

Build it locally with:

```bash
cargo doc --no-deps --no-default-features --features mpi,rayon --open
```
