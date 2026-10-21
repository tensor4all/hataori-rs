# Choosing Features and Backends

## Feature matrix

| Features | `map` | `map_in` | `pmap` | placement collectives |
| --- | --- | --- | --- | --- |
| none | serial | — | — | — |
| `rayon` | serial | Rayon | — | — |
| `mpi` or `rsmpi-rt` | serial | — | MPI-only (sequential domain) | yes |
| `mpi`+`rayon` or `rsmpi-rt`+`rayon` | serial | Rayon | hybrid (pooled domain) | yes |

Without any MPI feature the dependency tree contains no `mpi`, `serde`, or
`bincode` crates; the CI enforces this so serial and Rayon users pay nothing
for the distributed backend.

## `mpi` vs `rsmpi-rt`

Both features expose the same Hataori API on top of the rsmpi crate, so user
code is identical apart from the crate alias it imports.

| | `mpi` | `rsmpi-rt` |
| --- | --- | --- |
| rsmpi crate | upstream `mpi = 0.8.1` | [tensor4all/rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) |
| Linking | against the system MPI at build time | none; the MPIABI library named by `MPI_RT_LIB` is `dlopen`ed at start |
| Build requirements | MPI headers, `mpicc`, libclang (bindgen) | a Rust toolchain only |
| Typical use | HPC clusters where the module system provides MPI | wheels/binaries shared with MPI.jl or mpi4py, sandboxed builds |

`MPI_RT_LIB` should be an absolute path to an MPIABI-compatible library (the
loader passes it to `dlopen` verbatim; the project scripts require an absolute
path), such as
one built from [MPIwrapper](https://github.com/eschnett/MPIwrapper). See the
[rsmpi-rt tutorial](../tutorials/rsmpi-rt.md).

## The `tenferro` feature

`tenferro` pulls in `tenferro-tensor` for the Mandelbrot examples under
`examples/`; because the pinned tenferro-rs workspace sets
`rust-version = "1.96"`, this feature needs Rust >= 1.96 (the rest of the
core matrix keeps MSRV 1.85). The separate `hataori-tenferro` workspace
adapter, which likewise declares `rust-version = "1.96"`, runs one lane per
caller-held tenferro engine for `map_in_lanes` and `pmap_lanes`, so Hataori
schedules items across the caller's engines; see the
[adapter design](../design/tenferro-adapter.md).
