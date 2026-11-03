use crate::{AlgorithmError, LocalMode, PmapOptions};
use hataori_runtime::{
    Action, ActionError, Place, PlacementFallback, Remote, Runtime, RuntimeBuilder, RuntimeError,
    Segments, SpawnOptions, WireValue,
};
use rayon::prelude::*;
use std::{
    collections::VecDeque,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

const ACTION_MASK: u128 = 0xe500_0000_0000_0000_0000_0000_0000_0001;
const MAX_BATCH_ITEMS: usize = 1_048_576;
const MAX_ERROR_BYTES: usize = 4096;

type BatchFuture<U> = Pin<Box<dyn Future<Output = Result<BatchReply<U>, RuntimeError>>>>;
type Submit<A> =
    Box<dyn FnMut(BatchAction<A>) -> Result<BatchFuture<<A as Action>::Output>, RuntimeError>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct BatchActionToken<A>(PhantomData<fn() -> A>);
impl<A> BatchActionToken<A> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

pub struct BatchAction<A> {
    start: usize,
    mode: LocalMode,
    items: Vec<A>,
}
pub enum ItemReply<T> {
    Ok(T),
    Err(String),
}
pub struct BatchReply<T> {
    start: usize,
    items: Vec<ItemReply<T>>,
}

fn put(value: usize, out: &mut Vec<u8>) -> Result<(), ActionError> {
    let value = u64::try_from(value).map_err(|_| ActionError::codec("batch count overflow"))?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}
fn take(bytes: &[u8], at: &mut usize) -> Result<usize, ActionError> {
    let end = at
        .checked_add(8)
        .ok_or_else(|| ActionError::codec("batch header overflow"))?;
    let raw: [u8; 8] = bytes
        .get(*at..end)
        .ok_or_else(|| ActionError::codec("truncated batch header"))?
        .try_into()
        .unwrap();
    *at = end;
    usize::try_from(u64::from_le_bytes(raw)).map_err(|_| ActionError::codec("batch count overflow"))
}
fn encode_nested<T: WireValue>(
    values: Vec<T>,
    prefix: &mut Vec<u8>,
) -> Result<Segments, ActionError> {
    if values.len() > MAX_BATCH_ITEMS {
        return Err(ActionError::codec("batch item limit exceeded"));
    }
    put(values.len(), prefix)?;
    let mut out = vec![std::mem::take(prefix)];
    for value in values {
        let segments = value.encode()?;
        put(segments.len(), &mut out[0])?;
        for segment in segments {
            put(segment.len(), &mut out[0])?;
            out.push(segment);
        }
    }
    Ok(out)
}
fn decode_nested<T: WireValue>(
    mut segments: Segments,
    prefix: usize,
) -> Result<(Vec<u8>, Vec<T>), ActionError> {
    if segments.is_empty() {
        return Err(ActionError::codec("missing batch header"));
    }
    let header = segments.remove(0);
    let mut at = prefix;
    let count = take(&header, &mut at)?;
    if count > MAX_BATCH_ITEMS {
        return Err(ActionError::codec("batch item limit exceeded"));
    }
    let mut lengths = Vec::with_capacity(count);
    for _ in 0..count {
        let n = take(&header, &mut at)?;
        let mut item = Vec::with_capacity(n);
        for _ in 0..n {
            item.push(take(&header, &mut at)?);
        }
        lengths.push(item);
    }
    if at != header.len() {
        return Err(ActionError::codec("trailing batch header bytes"));
    }
    let expected: usize = lengths
        .iter()
        .map(Vec::len)
        .try_fold(0usize, |a, b| a.checked_add(b))
        .ok_or_else(|| ActionError::codec("segment count overflow"))?;
    if segments.len() != expected {
        return Err(ActionError::codec("batch segment count mismatch"));
    }
    let mut source = segments.into_iter();
    let mut values = Vec::with_capacity(count);
    for item in lengths {
        let mut encoded = Vec::with_capacity(item.len());
        for length in item {
            let bytes = source.next().unwrap();
            if bytes.len() != length {
                return Err(ActionError::codec("batch segment length mismatch"));
            }
            encoded.push(bytes);
        }
        values.push(T::decode(encoded)?);
    }
    Ok((header, values))
}
impl<A: Action> WireValue for BatchAction<A> {
    const SCHEMA_ID: u64 = A::SCHEMA_ID ^ 0xe501_0000_0000_0001;
    fn encode(self) -> Result<Segments, ActionError> {
        let mut h = Vec::new();
        put(self.start, &mut h)?;
        h.push(self.mode as u8);
        encode_nested(self.items, &mut h)
    }
    fn decode(segments: Segments) -> Result<Self, ActionError> {
        let header = segments
            .first()
            .ok_or_else(|| ActionError::codec("missing batch header"))?;
        let mut at = 0;
        let start = take(header, &mut at)?;
        let mode = match *header
            .get(at)
            .ok_or_else(|| ActionError::codec("missing local mode"))?
        {
            0 => LocalMode::Sequential,
            1 => LocalMode::Outer,
            2 => LocalMode::Inner,
            _ => return Err(ActionError::codec("invalid local mode")),
        };
        at += 1;
        let (_, items) = decode_nested(segments, at)?;
        Ok(Self { start, mode, items })
    }
}
impl<T: WireValue> WireValue for BatchReply<T> {
    const SCHEMA_ID: u64 = T::SCHEMA_ID ^ 0xe502_0000_0000_0001;
    fn encode(self) -> Result<Segments, ActionError> {
        let mut h = Vec::new();
        put(self.start, &mut h)?;
        let values = self
            .items
            .into_iter()
            .map(|item| match item {
                ItemReply::Ok(value) => Ok(ReplyWire::Ok(value)),
                ItemReply::Err(mut message) => {
                    while message.len() > MAX_ERROR_BYTES {
                        message.pop();
                    }
                    Ok(ReplyWire::Err(message))
                }
            })
            .collect::<Result<Vec<_>, ActionError>>()?;
        encode_nested(values, &mut h)
    }
    fn decode(segments: Segments) -> Result<Self, ActionError> {
        let header = segments
            .first()
            .ok_or_else(|| ActionError::codec("missing reply header"))?;
        let mut at = 0;
        let start = take(header, &mut at)?;
        let (_, values) = decode_nested::<ReplyWire<T>>(segments, at)?;
        Ok(Self {
            start,
            items: values
                .into_iter()
                .map(|v| match v {
                    ReplyWire::Ok(x) => ItemReply::Ok(x),
                    ReplyWire::Err(e) => ItemReply::Err(e),
                })
                .collect(),
        })
    }
}
enum ReplyWire<T> {
    Ok(T),
    Err(String),
}
impl<T: WireValue> WireValue for ReplyWire<T> {
    const SCHEMA_ID: u64 = T::SCHEMA_ID ^ 0xe503_0000_0000_0001;
    fn encode(self) -> Result<Segments, ActionError> {
        match self {
            Self::Ok(v) => {
                let mut s = v.encode()?;
                s.insert(0, vec![0]);
                Ok(s)
            }
            Self::Err(e) => Ok(vec![vec![1], e.into_bytes()]),
        }
    }
    fn decode(mut s: Segments) -> Result<Self, ActionError> {
        if s.is_empty() {
            return Err(ActionError::codec("missing reply tag"));
        }
        let tag = s.remove(0);
        match tag.as_slice() {
            [0] => Ok(Self::Ok(T::decode(s)?)),
            [1] => {
                if s.len() != 1 || s[0].len() > MAX_ERROR_BYTES {
                    return Err(ActionError::codec("invalid callback error"));
                }
                Ok(Self::Err(String::from_utf8(s.pop().unwrap()).map_err(
                    |_| ActionError::codec("callback error is not UTF-8"),
                )?))
            }
            _ => Err(ActionError::codec("invalid reply tag")),
        }
    }
}
impl<A: Action> Action for BatchAction<A> {
    const ID: u128 = A::ID ^ ACTION_MASK;
    type Output = BatchReply<A::Output>;
}

pub trait AlgorithmRegistryExt {
    fn register_pmap_action<A, F>(
        &mut self,
        handler: F,
    ) -> Result<BatchActionToken<A>, RuntimeError>
    where
        A: Action,
        F: Fn(A) -> Result<A::Output, ActionError> + Send + Sync + 'static;
}
impl AlgorithmRegistryExt for RuntimeBuilder {
    fn register_pmap_action<A, F>(
        &mut self,
        handler: F,
    ) -> Result<BatchActionToken<A>, RuntimeError>
    where
        A: Action,
        F: Fn(A) -> Result<A::Output, ActionError> + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        self.register::<BatchAction<A>, _>(move |batch| {
            let apply = |item| match handler(item) {
                Ok(v) => ItemReply::Ok(v),
                Err(e) => ItemReply::Err(e.to_string()),
            };
            let items = match batch.mode {
                LocalMode::Outer => batch.items.into_par_iter().map(apply).collect(),
                _ => batch.items.into_iter().map(apply).collect(),
            };
            Ok(BatchReply {
                start: batch.start,
                items,
            })
        })?;
        Ok(BatchActionToken::new())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControllerStats {
    pub item_count: usize,
    pub batch_count: usize,
    pub peak_in_flight: usize,
    pub placement_submissions: usize,
}
struct Slot<U> {
    future: BatchFuture<U>,
}
pub struct PmapFuture<A: Action> {
    pending: VecDeque<(usize, Vec<A>)>,
    slots: Vec<Slot<A::Output>>,
    submit: Submit<A>,
    options: PmapOptions,
    outputs: Vec<Option<A::Output>>,
    first_error: Option<(usize, String)>,
    stats: ControllerStats,
    done: bool,
}
impl<A: Action> Unpin for PmapFuture<A> {}
impl<A: Action> Future for PmapFuture<A> {
    type Output = Result<(Vec<A::Output>, ControllerStats), AlgorithmError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        loop {
            while this.slots.len() < this.options.inflight_limit && !this.pending.is_empty() {
                let (start, items) = this.pending.pop_front().unwrap();
                let action = BatchAction {
                    start,
                    mode: this.options.local_mode,
                    items,
                };
                let future = match (this.submit)(action) {
                    Ok(f) => f,
                    Err(e) => {
                        this.done = true;
                        return Poll::Ready(Err(e.into()));
                    }
                };
                this.slots.push(Slot { future });
                this.stats.peak_in_flight = this.stats.peak_in_flight.max(this.slots.len());
                this.stats.placement_submissions += 1;
            }
            let mut progress = false;
            let mut i = 0;
            while i < this.slots.len() {
                match this.slots[i].future.as_mut().poll(cx) {
                    Poll::Pending => i += 1,
                    Poll::Ready(Err(e)) => {
                        this.done = true;
                        return Poll::Ready(Err(e.into()));
                    }
                    Poll::Ready(Ok(reply)) => {
                        progress = true;
                        let Slot { future: _ } = this.slots.swap_remove(i);
                        for (offset, item) in reply.items.into_iter().enumerate() {
                            let index = reply.start + offset;
                            match item {
                                ItemReply::Ok(v) => {
                                    if index >= this.outputs.len() || this.outputs[index].is_some()
                                    {
                                        this.done = true;
                                        return Poll::Ready(Err(AlgorithmError::Protocol(
                                            "invalid batch result index",
                                        )));
                                    }
                                    this.outputs[index] = Some(v)
                                }
                                ItemReply::Err(message) => {
                                    if this
                                        .first_error
                                        .as_ref()
                                        .is_none_or(|(old, _)| index < *old)
                                    {
                                        this.first_error = Some((index, message));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if this.pending.is_empty() && this.slots.is_empty() {
                this.done = true;
                if let Some((index, message)) = this.first_error.take() {
                    return Poll::Ready(Err(AlgorithmError::Callback { index, message }));
                }
                let mut out = Vec::with_capacity(this.outputs.len());
                for value in &mut this.outputs {
                    out.push(
                        value
                            .take()
                            .ok_or(AlgorithmError::Protocol("missing batch result"))?,
                    );
                }
                return Poll::Ready(Ok((out, this.stats)));
            }
            if !progress {
                return Poll::Pending;
            }
        }
    }
}

fn build<A: Action>(
    items: Vec<A>,
    options: PmapOptions,
    submit: Submit<A>,
) -> Result<PmapFuture<A>, AlgorithmError> {
    options.validate()?;
    let len = items.len();
    let mut source = items.into_iter();
    let mut pending = VecDeque::new();
    let mut start = 0;
    while start < len {
        let chunk: Vec<_> = source.by_ref().take(options.batch_size).collect();
        let n = chunk.len();
        pending.push_back((start, chunk));
        start += n;
    }
    let batches = pending.len();
    Ok(PmapFuture {
        pending,
        slots: Vec::new(),
        submit,
        options,
        outputs: (0..len).map(|_| None).collect(),
        first_error: None,
        stats: ControllerStats {
            item_count: len,
            batch_count: batches,
            ..Default::default()
        },
        done: false,
    })
}
pub fn pmap<A: Action>(
    runtime: &Runtime,
    options: PmapOptions,
    items: Vec<A>,
    _: BatchActionToken<A>,
) -> Result<PmapFuture<A>, AlgorithmError> {
    let client = runtime.client();
    let places: Vec<_> = runtime
        .localities()
        .iter()
        .map(|&l| Place::new(l, options.domain))
        .collect();
    let options = options.with_targets(places.len())?;
    let deadline = options.deadline;
    let mut next = 0;
    build(
        items,
        options,
        Box::new(move |a| {
            let place = places[next % places.len()];
            next += 1;
            Ok(Box::pin(client.spawn_on_with(
                place,
                a,
                SpawnOptions {
                    deadline,
                    trace_id: None,
                },
            )?))
        }),
    )
}
pub fn pmap_at<A: Action>(
    runtime: &Runtime,
    place: Place,
    options: PmapOptions,
    items: Vec<A>,
    _: BatchActionToken<A>,
) -> Result<PmapFuture<A>, AlgorithmError> {
    let client = runtime.client();
    let options = options.with_targets(1)?;
    let deadline = options.deadline;
    build(
        items,
        options,
        Box::new(move |a| {
            Ok(Box::pin(client.spawn_on_with(
                place,
                a,
                SpawnOptions {
                    deadline,
                    trace_id: None,
                },
            )?))
        }),
    )
}
pub fn pmap_preferred_colocated<A: Action, T: hataori_runtime::DistributedObject>(
    runtime: &Runtime,
    remote: &Remote<T>,
    fallback: PlacementFallback,
    options: PmapOptions,
    items: Vec<A>,
    _: BatchActionToken<A>,
) -> Result<PmapFuture<A>, AlgorithmError> {
    let client = runtime.client();
    let remote = remote.clone();
    let options = options.with_targets(1)?;
    build(
        items,
        options,
        Box::new(move |a| {
            Ok(Box::pin(
                client.spawn_preferred_colocated(&remote, fallback, a)?,
            ))
        }),
    )
}

pub fn pmap_colocated<A: Action, T: hataori_runtime::DistributedObject>(
    runtime: &Runtime,
    remote: &Remote<T>,
    options: PmapOptions,
    items: Vec<A>,
    _: BatchActionToken<A>,
) -> Result<PmapFuture<A>, AlgorithmError> {
    let client = runtime.client();
    let remote = remote.clone();
    let options = options.with_targets(1)?;
    build(
        items,
        options,
        Box::new(move |a| Ok(Box::pin(client.spawn_colocated(&remote, a)?))),
    )
}

#[cfg(test)]
mod tests;
