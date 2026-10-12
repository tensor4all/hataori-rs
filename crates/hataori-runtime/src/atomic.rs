//! Conditional atomic update that builds on both the MSRV and current stable.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// `fetch_update` under one name for every Rust version this crate supports.
///
/// Rust 1.99 deprecates `fetch_update` in favour of `try_update`, which does
/// not exist on the 1.85 MSRV. Call sites use this trait so the deprecation
/// allowance stays in one place; switch to `try_update` when the MSRV reaches
/// the release that stabilized it.
pub(crate) trait AtomicTryUpdate<T> {
    fn try_update_compat(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        f: impl FnMut(T) -> Option<T>,
    ) -> Result<T, T>;
}

macro_rules! impl_try_update_compat {
    ($atomic:ty, $value:ty) => {
        impl AtomicTryUpdate<$value> for $atomic {
            #[allow(deprecated)]
            fn try_update_compat(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                f: impl FnMut($value) -> Option<$value>,
            ) -> Result<$value, $value> {
                self.fetch_update(set_order, fetch_order, f)
            }
        }
    };
}

impl_try_update_compat!(AtomicU64, u64);
impl_try_update_compat!(AtomicUsize, usize);
