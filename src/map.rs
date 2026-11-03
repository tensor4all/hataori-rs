use std::fmt::{self, Display, Formatter};

pub(crate) const MAX_ERROR_MESSAGE_BYTES: usize = 4096;

/// The first callback error returned by [`map`].
#[derive(Debug)]
pub struct MapError {
    index: usize,
    message: String,
}

impl MapError {
    /// Returns the zero-based index of the input that failed.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the callback error message, at most 4096 bytes and truncated at a UTF-8 character boundary.
    pub fn message(&self) -> &str {
        &self.message
    }

    fn new<E: Display>(index: usize, error: E) -> Self {
        Self {
            index,
            message: truncate_message(error.to_string()),
        }
    }
}

impl Display for MapError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "map failed at index {}: {}",
            self.index, self.message
        )
    }
}

impl std::error::Error for MapError {}

pub(crate) fn truncate_message(mut message: String) -> String {
    if message.len() > MAX_ERROR_MESSAGE_BYTES {
        let mut end = MAX_ERROR_MESSAGE_BYTES;
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        message.truncate(end);
    }
    message
}

/// Applies `f` to `items` sequentially and returns results in their original order.
///
/// Each successful input is evaluated exactly once. Evaluation stops at the first
/// callback error, and trailing unconsumed items are dropped naturally. Errors
/// report a zero-based input index and a message truncated at a UTF-8 character
/// boundary to at most 4096 bytes. This serial operation has no serde, `Send`,
/// `Sync`, or `'static` requirements.
///
/// # Example
///
/// ```
/// use hataori::map;
///
/// let doubled = map(vec![1, 2, 3], |item| {
///     Ok::<_, std::convert::Infallible>(item * 2)
/// })
/// .unwrap();
/// assert_eq!(doubled, vec![2, 4, 6]);
/// ```
pub fn map<T, U, E, F>(items: Vec<T>, mut f: F) -> Result<Vec<U>, MapError>
where
    F: FnMut(T) -> Result<U, E>,
    E: Display,
{
    let mut output = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        match f(item) {
            Ok(value) => output.push(value),
            Err(error) => return Err(MapError::new(index, error)),
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
