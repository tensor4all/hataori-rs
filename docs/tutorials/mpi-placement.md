# 4. MPI placement collectives

**Model:** moving owned values between ranks. **Features:** `mpi,tenferro`
(or `rsmpi-rt,tenferro`). **Example:** `mpi_mandelbrot_placement`.

Besides `pmap`, the MPI backends provide three collectives for serde values.
They complement `pmap`: use `broadcast` for parameters every rank needs,
`scatter` to give each rank its own shard, and `gather` to collect one value
per rank on the root.

| Collective | Root passes | Every rank receives |
| --- | --- | --- |
| `broadcast(world, root, Option<T>)` | `Some(value)` | an owned copy of `value` |
| `scatter(world, root, Option<Vec<T>>)` | `Some(shards)` with exactly one shard per rank | its rank-indexed shard |
| `gather(world, root, T)` | (every rank passes one value) | root: `Some(Vec<T>)` in rank order; others: `None` |

This tutorial renders the Mandelbrot image with a **static** split: the
root broadcasts the parameters, scatters one contiguous block of column
indices to each rank, every rank computes its block, and the root gathers
the blocks.

## Serializable value types

<!-- snippet-source: examples/mpi_mandelbrot_placement.rs#placement-types -->
```rust
/// Values that cross rank boundaries need serde. `Param` itself lives in the
/// shared example module and derives nothing, so wrap the fields the ranks
/// need.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Region {
    xmin: f64,
    xmax: f64,
    width: usize,
    ymin: f64,
    ymax: f64,
    height: usize,
    max_iter: i64,
}

impl From<Param> for Region {
    fn from(param: Param) -> Self {
        Self {
            xmin: param.xmin,
            xmax: param.xmax,
            width: param.width,
            ymin: param.ymin,
            ymax: param.ymax,
            height: param.height,
            max_iter: param.max_iter,
        }
    }
}

impl From<Region> for Param {
    fn from(region: Region) -> Self {
        Self {
            xmin: region.xmin,
            xmax: region.xmax,
            width: region.width,
            ymin: region.ymin,
            ymax: region.ymax,
            height: region.height,
            max_iter: region.max_iter,
        }
    }
}

/// One rank's share of the image: a contiguous block of column indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Block {
    first_column: usize,
    columns: Vec<Vec<i64>>,
}
```
<!-- end-snippet-source -->

Only the values that cross rank boundaries need serde: the shared `Param`
is mirrored into a `Region`, and a rank's result is a `Block` of columns
with the index of its first column.

## The three collectives

<!-- snippet-source: examples/mpi_mandelbrot_placement.rs#placement-call -->
```rust
/// Static distribution with the three collectives. Every rank calls each
/// collective in the same order with the same root.
fn render<C: Communicator>(world: &C, root_param: Option<Param>) -> Option<Tensor> {
    let rank = world.rank();
    let size = world.size() as usize;
    let root = 0;

    // broadcast: the root supplies `Some`, every rank receives an owned copy.
    let region: Region =
        broadcast(world, root, root_param.map(Region::from)).expect("broadcast must succeed");
    let param = Param::from(region);
    let (x, y) = param.make_axes();

    // scatter: the root supplies exactly one shard per rank; each rank
    // receives the shard at its own index. Here a shard is a contiguous
    // range of column indices, so neighbouring ranks get neighbouring
    // columns — and, near the set, very different amounts of work.
    let shards = (rank == root).then(|| {
        (0..size)
            .map(|target| {
                let first = target * param.width / size;
                let last = (target + 1) * param.width / size;
                (first..last).collect::<Vec<usize>>()
            })
            .collect::<Vec<_>>()
    });
    let my_columns: Vec<usize> = scatter(world, root, shards).expect("scatter must succeed");

    // Compute locally; no Hataori call is involved in this step.
    let block = Block {
        first_column: my_columns.first().copied().unwrap_or(0),
        columns: my_columns
            .iter()
            .map(|&col_idx| mandelbrot_common::compute_column(&param, x[col_idx], &y))
            .collect(),
    };

    // gather: every rank supplies one value; the root receives them in rank
    // order and every other rank receives `None`.
    let blocks = gather(world, root, block).expect("gather must succeed");
    blocks.map(|blocks| {
        let mut full_data = vec![0_i64; param.width * param.height];
        for block in blocks {
            for (offset, column) in block.columns.iter().enumerate() {
                let col_idx = block.first_column + offset;
                mandelbrot_common::place_column(&mut full_data, param.height, col_idx, column);
            }
        }
        mandelbrot_common::tensor_from_columns(param.width, param.height, full_data)
    })
}
```
<!-- end-snippet-source -->

- Only the root parses the command line; `broadcast` delivers the
  parameters to everyone.
- `scatter` requires exactly one shard per rank on the root. Each shard here
  is a contiguous range, so neighbouring ranks get neighbouring columns — and,
  near the set, very different amounts of work.
- The compute step in the middle is plain Rust; no Hataori call is involved.
- `gather` returns the blocks in rank order on the root, which then places
  each block by its `first_column`.

Errors are `PlacementError` values with a `PlacementErrorKind`; like `pmap`,
a failure (for example a serialization error on one rank) converges to the
same error on every rank.

Source: [`examples/mpi_mandelbrot_placement.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/mpi_mandelbrot_placement.rs)

## Build and run

```bash
cargo build --release --no-default-features --features mpi,tenferro \
  --example mpi_mandelbrot_placement
mpiexec -n 4 target/release/examples/mpi_mandelbrot_placement --width 1024 --height 1024
```

## Static versus dynamic

Compare the total time with [tutorial 3](mpi-pmap.md) at the same size and
rank count. The collectives make the static version as short to write as
the `pmap` one, but nothing rebalances the uneven columns: the rank whose
block crosses the set finishes last while the others wait in `gather`.
That gap is what `pmap`'s dynamic scheduling removes.

Next: [5. Hybrid MPI + Rayon pmap](hybrid-pmap.md).
