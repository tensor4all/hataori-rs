# Hataori

Hataori is a Rust engine for simple serial, Rayon, MPI, and hybrid data-parallel execution. It is inspired by Distributed.jl's dynamically scheduled `pmap` while retaining Rust's scoped execution and MPI's SPMD model.

## Documentation

The online documentation — execution-model guide, runnable tutorials for the
serial, Rayon, MPI, hybrid, and runtime-loaded-MPI modes, and the API
reference — is published from `docs/` at
<https://tensor4all.org/hataori-rs/>. Build it locally with
`scripts/build_docs_site.sh` (requires [Quarto](https://quarto.org)); the
tutorials quote the Mandelbrot examples under `examples/`, which run with
`scripts/check-tutorial-examples.sh`.

Coding agents and readers who want one machine-oriented entry point should
start from the [llms.txt index](docs/llms.txt) (published at
<https://tensor4all.org/hataori-rs/llms.txt>): it states the calling
conventions every model shares and lists every guide, tutorial, API page, and
design document with a one-line description, so each is one fetch away.

## Name

**Hataori** comes from the Japanese word **機織り** (*hataori*), meaning weaving on a loom. The name reflects the engine's job: weave independent strands of work across MPI ranks and Rayon threads into one ordered result.

## Development status

The public `hataori` crate implements the synchronous P0/P1 `map`, `map_in`,
collective `pmap`, and placement helpers. The unpublished
`hataori-runtime-foundation` workspace crate implements Issue #14 Phase A:
transport-independent protocol/framing, deterministic memory faults, bounded
TCP rendezvous/transport, and MPI transport with segmented payloads. The
unpublished `hataori-runtime` crate implements Phase B above that boundary:
long-lived owner-thread progress, bounded worker domains, typed actions,
futures, scopes, cancellation, backpressure, statistics, and reusable shutdown.
The same crate implements Phase C remote objects and Phase D manual migration.
The unpublished `hataori-algorithms` crate implements Phase E bounded `pmap`,
action/future collectives, per-batch placement, and direct async/blocking
facades. Automatic migration remains deferred, and performance promotion
awaits a valid-host PASS report.

Run the five correctness acceptance lanes with:

```bash
scripts/check-runtime-foundation.sh
scripts/check-phase-b-runtime.sh
scripts/check-phase-c-objects.sh
scripts/check-phase-d-migration.sh
scripts/check-phase-e-algorithms.sh
```

See the Phase A-E acceptance ledgers and design documents under `docs/`; Phase
E's ledger records the outstanding performance acceptance explicitly.

## Features

Hataori has no default dependencies. Optional execution backends are selected explicitly:

| Feature | Backend |
|---|---|
| `rayon` | rank-local thread parallelism |
| `mpi` | upstream rsmpi with a build/link-time MPI implementation |
| `rsmpi-rt` | [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) with MPIABI runtime loading |

`mpi` and `rsmpi-rt` expose the same Hataori API and are mutually exclusive. The `rsmpi-rt` feature supports build environments without MPI headers, a C compiler, or libclang and can share the MPI runtime used by MPI.jl or mpi4py.

The non-publishable `hataori-tenferro` workspace adapter binds an admitted
whole-domain `Inner` callback to tenferro's caller-managed Faer backend without
adding tenferro to ordinary Hataori builds; it declares `rust-version = "1.96"`
following the pinned tenferro-rs. See its
[design and usage contract](docs/design/tenferro-adapter.md).

### MPI backend prerequisites

The `mpi` feature links against the system MPI implementation at build time. On macOS, install Open MPI first, for example with Homebrew:

```bash
brew install open-mpi
```

Then verify that `mpicc` and `mpiexec` are on your `PATH` before building with `--features mpi`.

## Running the examples

Build and run the MPI smoke tests with `mpiexec`. The smoke examples exit silently on success (except `rsmpi_rt_pmap_smoke`, which prints the loaded `MPI_RT_LIB`) because they verify behaviour through assertions.

### Upstream MPI backend (`mpi`)

```bash
cargo build --example mpi_pmap_smoke --no-default-features --features mpi,rayon
mpiexec -n 4 target/debug/examples/mpi_pmap_smoke && echo "pmap smoke OK"

cargo build --example mpi_placement_smoke --no-default-features --features mpi
mpiexec -n 4 target/debug/examples/mpi_placement_smoke && echo "placement smoke OK"
```

### Runtime-loaded MPI backend (`rsmpi-rt`)

```bash
cargo build --example rsmpi_rt_pmap_smoke --no-default-features --features rsmpi-rt,rayon
MPI_RT_LIB=/absolute/path/to/libmpiwrapper.so mpiexec -n 4 target/debug/examples/rsmpi_rt_pmap_smoke && echo "pmap smoke OK"
```

## Developer guide: working on the documentation

The site under `docs/` is a [Quarto](https://quarto.org) website. The
tutorial pages quote Rust code verbatim from the Mandelbrot examples under
`examples/`, and CI rejects pages that drift from that code.

### Preview locally

From the repository root:

```bash
quarto preview docs
```

Quarto opens a browser (default `http://localhost:4xxx/`) and re-renders and
reloads the page whenever a file under `docs/**/*.md` is saved. To keep the
browser closed or pin the port:

```bash
quarto preview docs --no-browser --port 4321
```

The rendered site goes to `target/docs-site/`, as configured in
`docs/_quarto.yml`.

### Include the API reference in the preview

`quarto preview` only regenerates the Quarto pages. The rustdoc served under
`api/hataori/…` must be placed there once by a full build; after that the
preview keeps serving it while you edit:

```bash
scripts/build_docs_site.sh   # snippet check -> rustdoc -> Quarto -> copy into api/
quarto preview docs          # live reload from here on
```

`scripts/build_docs_site.sh` is the same script CI runs to publish the site.

### Edit tutorial code

The Rust blocks in `docs/tutorials/` and `docs/index.md` are copies of the
regions between `// snippet-start:NAME` and `// snippet-end:NAME` in
`examples/*.rs` and `examples/support/mandelbrot_common.rs`. After changing a
`.rs` file, re-sync the Markdown:

```bash
python3 scripts/check-doc-snippets.py
```

CI runs the same script with `--check` and fails on stale snippets or on any
` ```rust ` fence in the guides and tutorials that is not backed by a
snippet marker. Run the examples themselves (small image, every feature
lane, MPI ones via `mpiexec -n 2`; needs Rust >= 1.96 for `tenferro`) with:

```bash
scripts/check-tutorial-examples.sh                                   # link-time MPI lanes
scripts/check-tutorial-examples.sh /abs/path/to/libmpiwrapper.so     # plus rsmpi-rt lanes
```

### Keep `docs/llms.txt` in sync

`docs/llms.txt` is the machine-oriented index of the site. When a page is
added, renamed, or removed under `docs/`, update its entry there;
`scripts/check-llms-index.py` (run by `scripts/build_docs_site.sh` and CI)
fails when an entry points at a page that no longer exists, when a page
rendered by `docs/_quarto.yml` has no entry, when a URL is repeated, or when
the README stops linking the index:

```bash
python3 scripts/check-llms-index.py
```

### Publishing

`.github/workflows/docs.yml` runs the tutorial examples, builds the site, and
deploys it to GitHub Pages on every push to `main`; pull requests only build.
Deployment requires the repository's Pages source to be set to "GitHub
Actions" once by an administrator.

## Status

Hataori's P0 core and tenferro-only adapter foundation are implemented. P1 adds
an opt-in, default-off bounded prefetch of one batch per remote hybrid domain.
The tensor4all explicit-context and tensor-reconstruction layer remains blocked
on [tensor4all-rs#663](https://github.com/tensor4all/tensor4all-rs/issues/663).

- [P0 design](docs/design.md)
- [P1 bounded-prefetch design](docs/design/bounded-prefetch.md)
- [Long-term distributed-runtime design](docs/design/distributed-runtime.md)
- [Object placement and locality-aware task API](docs/design/object-placement-api.md)
- [Implementation readiness and validation matrix](docs/implementation-readiness.md)
- [Review gate log](docs/review-log.md)
- [P0 implementation tracker](https://github.com/tensor4all/hataori-rs/issues/1)
- [Long-term distributed-runtime tracker](https://github.com/tensor4all/hataori-rs/issues/14)
