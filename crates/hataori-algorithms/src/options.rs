use hataori_runtime::DomainId;
use std::time::Duration;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalMode {
    Sequential = 0,
    Outer = 1,
    Inner = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PmapOptions {
    pub batch_size: usize,
    pub local_mode: LocalMode,
    pub prefetch: bool,
    pub deadline: Duration,
    pub domain: DomainId,
    pub(crate) inflight_limit: usize,
}
impl Default for PmapOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            local_mode: LocalMode::Sequential,
            prefetch: false,
            deadline: Duration::from_secs(30),
            domain: DomainId::DEFAULT,
            inflight_limit: 1,
        }
    }
}
impl PmapOptions {
    pub fn batch_size(mut self, batch_size: usize) -> Result<Self, crate::AlgorithmError> {
        self.batch_size = batch_size;
        self.validate()?;
        Ok(self)
    }
    pub const fn local_mode(mut self, mode: LocalMode) -> Self {
        self.local_mode = mode;
        self
    }
    pub const fn prefetch(mut self, enabled: bool) -> Self {
        self.prefetch = enabled;
        self
    }
    pub const fn domain(mut self, domain: DomainId) -> Self {
        self.domain = domain;
        self
    }
    pub fn deadline(mut self, deadline: Duration) -> Result<Self, crate::AlgorithmError> {
        self.deadline = deadline;
        self.validate()?;
        Ok(self)
    }
    pub fn validate(&self) -> Result<(), crate::AlgorithmError> {
        if self.batch_size == 0 || self.batch_size > 1_048_576 {
            return Err(crate::AlgorithmError::InvalidOptions(
                "batch_size must be between 1 and 1048576",
            ));
        }
        if self.deadline.is_zero() {
            return Err(crate::AlgorithmError::InvalidOptions(
                "deadline must be nonzero",
            ));
        }
        Ok(())
    }
    pub(crate) fn with_targets(mut self, targets: usize) -> Result<Self, crate::AlgorithmError> {
        if targets == 0 {
            return Err(crate::AlgorithmError::InputShape(
                "membership must be nonzero",
            ));
        }
        self.inflight_limit = targets.checked_mul(1 + usize::from(self.prefetch)).ok_or(
            crate::AlgorithmError::InvalidOptions("in-flight batch limit overflow"),
        )?;
        Ok(self)
    }
}
#[cfg(test)]
mod tests;
