# Tutorials

Every tutorial renders the same image — a Mandelbrot set, computed one
column at a time — with a different execution model. Read them in order:
each later model keeps the calling convention of the previous one and adds
only the coordination that the new model needs.

| Tutorial | Model | Example | Features | Run with |
| --- | --- | --- | --- | --- |
| [0. The workload](workload.md) | — | `examples/support/mandelbrot_common.rs` | `tenferro` | — |
| [1. Serial map](serial-map.md) | one thread | `serial_mandelbrot` | `tenferro` | `cargo run` |
| [2. Rayon map_in](rayon-map-in.md) | rank-local thread pool, three `LocalMode`s | `rayon_mandelbrot` | `rayon,tenferro` | `cargo run` |
| [3. MPI pmap](mpi-pmap.md) | dynamic scheduling across ranks | `mpi_mandelbrot_pmap` | `mpi,tenferro` | `mpiexec` |
| [4. MPI placement collectives](mpi-placement.md) | `broadcast`, `scatter`, `gather` | `mpi_mandelbrot_placement` | `mpi,tenferro` | `mpiexec` |
| [5. Hybrid MPI + Rayon pmap](hybrid-pmap.md) | a Rayon pool on every rank, optional prefetch | `mpi_mandelbrot_hybrid` | `mpi,rayon,tenferro` | `mpiexec` |
| [6. Runtime-loaded MPI](rsmpi-rt.md) | tutorials 3–5 without linking MPI | `rsmpi_rt_mandelbrot_*` | `rsmpi-rt,tenferro` | `MPI_RT_LIB=… mpiexec` |

The hand-written MPI baseline,
[`examples/mpi_mandelbrot_raw.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/mpi_mandelbrot_raw.rs),
distributes the same columns round-robin with explicit sends and receives;
tutorials 3 and 4 compare against it. The π examples
(`mpi_pi_raw`, `mpi_pi_pmap`) apply the same `pmap` pattern to a second
workload.

## Running the tutorial code

The code on these pages is quoted verbatim from
[`examples/`](https://github.com/tensor4all/hataori-rs/tree/main/examples)
and re-synchronized by `scripts/check-doc-snippets.py`. Every example
asserts its own results and writes a PNG, so a successful exit is the test.

All examples accept `--width N`, `--height N`, `--max-iter N` (default
4096 × 4096 × 500) and `--output PATH`. The default image takes tens of
seconds on one core; pass a small size while reading along:

```bash
# one process, one thread
cargo run --release --no-default-features --features tenferro \
  --example serial_mandelbrot -- --width 512 --height 512

# every tutorial example, at a small size, with the MPI ones launched
# through `mpiexec -n 2` (needs Rust >= 1.96 for the tenferro feature)
scripts/check-tutorial-examples.sh

# additionally exercise the rsmpi-rt backend
scripts/check-tutorial-examples.sh /absolute/path/to/libmpiwrapper.so
```

Set `HATAORI_MPIEXEC` to use a different launcher (for example `srun`) and
`HATAORI_MPI_RANKS` to change the rank count used by the script.

::: {.callout-note}
The `tenferro` feature only provides the tensor type the examples use to
assemble and save the image; Hataori itself does not depend on it. Because
the pinned tenferro release requires Rust 1.96, so do these examples; the
library keeps its 1.85 minimum.
:::
