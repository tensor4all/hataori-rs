<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to hataori-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `hataori-rs`:

## Local facts

- **MSRV**: `rust-version = "1.85"` in the workspace manifest. The `core` job installs `1.85.0` and `scripts/check-core.sh`
  runs `cargo +1.85.0 check` in every feature configuration, so the claim is gated. The `tenferro` feature needs Rust
  1.96 and is outside that matrix by design.
- **Feature combinations CI builds**: `--no-default-features` with each of `""`, `rayon`, `mpi`, `mpi,rayon`,
  `mpi,bitcode`, `rsmpi-rt`, `rsmpi-rt,rayon`, each running `cargo test`, `cargo clippy --all-targets -D warnings`,
  `cargo doc --no-deps` and the 1.85.0 check, plus `cargo fmt --check` once.
- **Platforms claimed**: `ubuntu-latest` for all three jobs (`core`, `deps-fresh`, `tenferro-adapter`);
  `x86_64-apple-darwin` is installed as a check target only, not run. MPI comes from the system Open MPI
  (`libopenmpi-dev`, `openmpi-bin`) and the runtime-loaded backend from the pinned MPIwrapper fixture.
- **Local gate**: the matrix needs the MPIwrapper fixture built first, then the script that runs it:

  ```sh
  rustup toolchain install 1.85.0 --profile minimal
  rustup target add x86_64-apple-darwin
  sudo apt-get update && sudo apt-get install -y --no-install-recommends cmake libclang-dev libopenmpi-dev openmpi-bin python3
  git clone https://github.com/eschnett/MPIwrapper.git /tmp/MPIwrapper
  git -C /tmp/MPIwrapper checkout --detach 966f4231c96153a08295fc7d0bcbd65e916a73fd
  cmake -S /tmp/MPIwrapper -B /tmp/MPIwrapper/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/tmp/mpiwrapper-install
  cmake --build /tmp/MPIwrapper/build --parallel 2 && cmake --install /tmp/MPIwrapper/build
  scripts/check-core.sh /tmp/mpiwrapper-install/lib/libmpiwrapper.so \
        966f4231c96153a08295fc7d0bcbd65e916a73fd
  ```

  The MPIwrapper fixture is the step people forget: the `rsmpi-rt` rows load it at run time, and `check-core.sh`
  refuses a path that is not an existing absolute file. The separate adapter lane is `scripts/check-tenferro.sh`.

## Anything here that departs from the shared rules

The `docs` gate departs: `check-core.sh` builds rustdoc with `cargo doc --no-deps --locked` without denying warnings,
so the root crate's public documentation is not gated warning-free; the `check-phase-*` scripts deny warnings for
`hataori-runtime` and `hataori-algorithms`, and `check-tenferro.sh` does so for the adapter.

The workflow's other steps cover `fmt`, `clippy`, `test` and `msrv`. `check-core.sh` passes `--locked` to the initial
`cargo metadata` read and to the seven-row matrix (`cargo test`, `cargo clippy`, `cargo doc`, `cargo +1.85.0 check`);
the two `cargo tree` probes, the mutually exclusive feature probe, the Darwin target check, and the helper scripts it
calls do not pass `--locked`.
