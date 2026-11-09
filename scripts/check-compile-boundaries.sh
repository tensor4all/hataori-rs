#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

make_crate() {
    local name=$1
    local hataori_features=$2
    local extra=${3-}
    local dir=$tmp_dir/$name
    mkdir -p "$dir/src"
    cat >"$dir/Cargo.toml" <<EOF
[package]
name = "$name"
version = "0.0.0"
edition = "2021"

[dependencies]
hataori = { path = "$root_dir", default-features = false, features = [$hataori_features] }
$extra
EOF
    printf '%s\n' "$dir"
}

serial=$(make_crate serial_boundary "")
cat >"$serial/src/main.rs" <<'RS'
use std::{cell::RefCell, rc::Rc};
fn main() {
    let text = String::from("borrowed");
    let calls = Rc::new(RefCell::new(0));
    let seen = Rc::clone(&calls);
    let values = vec![Rc::new(text.as_str())];
    let out = hataori::map(values, |value| {
        *seen.borrow_mut() += 1;
        Ok::<_, Rc<&str>>(value)
    }).unwrap();
    assert_eq!(*out[0], "borrowed");
}
RS
cargo check --quiet --manifest-path "$serial/Cargo.toml"

rayon_case=$(make_crate rayon_boundary '"rayon"' 'rayon = "=1.12.0"')
cat >"$rayon_case/src/main.rs" <<'RS'
use std::sync::Arc;
fn main() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let domain = hataori::Domain::external(Arc::clone(&pool), vec![0], 1).unwrap();
    let bias = 4;
    let borrowed = &bias;
    let out = hataori::map_in(&domain, hataori::LocalMode::Outer, vec![1], |value| {
        Ok::<_, &'static str>(value + *borrowed)
    }).unwrap();
    assert_eq!(out, vec![5]);
}
RS
cargo check --quiet --manifest-path "$rayon_case/Cargo.toml"

mpi_case=$(make_crate mpi_boundary '"mpi"' 'mpi-upstream = { package = "mpi", version = "=0.8.1", default-features = false }
serde = { version = "=1.0.229", features = ["derive"] }')
cat >"$mpi_case/src/main.rs" <<'RS'
use mpi_upstream::traits::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{cell::Cell, num::NonZeroUsize, rc::Rc};
struct Local(Rc<i32>);
impl Serialize for Local {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.as_ref().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for Local {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        i32::deserialize(deserializer).map(|value| Self(Rc::new(value)))
    }
}
fn main() {
    let universe = mpi_upstream::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let state = Rc::new(Cell::new(0));
    let captured = Rc::clone(&state);
    let _ = hataori::pmap(
        &world,
        &hataori::Domain::sequential(),
        hataori::PmapOptions { root: 0, batch_size: NonZeroUsize::new(1).unwrap(), local_mode: hataori::LocalMode::Sequential, prefetch: false },
        (rank == 0).then(|| vec![Local(Rc::new(1))]),
        move |value| { captured.set(captured.get() + 1); Ok::<_, String>(value) },
    );
}
RS
cargo check --quiet --manifest-path "$mpi_case/Cargo.toml"

hybrid=$(make_crate hybrid_boundary '"mpi", "rayon"' 'mpi-upstream = { package = "mpi", version = "=0.8.1", default-features = false }
rayon = "=1.12.0"')
cat >"$hybrid/src/main.rs" <<'RS'
use mpi_upstream::traits::*;
use std::{num::NonZeroUsize, sync::{Arc, atomic::{AtomicUsize, Ordering}}};
fn main() {
    let (universe, _) = mpi_upstream::initialize_with_threading(mpi_upstream::environment::Threading::Funneled).unwrap();
    let world = universe.world();
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let domain = hataori::Domain::external(pool, vec![0], 1).unwrap();
    let bias = 3;
    let calls = AtomicUsize::new(0);
    let _ = hataori::pmap(
        &world, &domain,
        hataori::PmapOptions { root: 0, batch_size: NonZeroUsize::new(1).unwrap(), local_mode: hataori::LocalMode::Outer, prefetch: true },
        (world.rank() == 0).then(|| vec![1]),
        |value| { calls.fetch_add(1, Ordering::Relaxed); Ok::<_, String>(value + bias) },
    );
}
RS
cargo check --quiet --manifest-path "$hybrid/Cargo.toml"

not_send=$(make_crate communicator_not_send '"mpi"' 'mpi-upstream = { package = "mpi", version = "=0.8.1", default-features = false }')
cat >"$not_send/src/main.rs" <<'RS'
use mpi_upstream::traits::*;
fn main() {
    let universe = mpi_upstream::initialize().unwrap();
    let world = universe.world();
    std::thread::spawn(move || world.rank()).join().unwrap();
}
RS
if cargo check --manifest-path "$not_send/Cargo.toml" >"$tmp_dir/not-send.out" 2>&1; then
    echo 'communicator unexpectedly compiled as Send' >&2
    exit 1
fi
grep -Eq 'cannot be sent|Send|Sync' "$tmp_dir/not-send.out"

both=$(make_crate both_backends '"mpi", "rsmpi-rt"')
echo 'fn main() {}' >"$both/src/main.rs"
if cargo check --manifest-path "$both/Cargo.toml" >"$tmp_dir/both.out" 2>&1; then
    echo 'mutually exclusive backends unexpectedly compiled' >&2
    exit 1
fi
grep -q 'mutually exclusive' "$tmp_dir/both.out"

printf 'compile boundary checks passed\n'
