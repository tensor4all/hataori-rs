mod support;

use hataori_runtime::{Place, Runtime, RuntimeError, RuntimeLimits};
use hataori_runtime_foundation::{
    protocol::{DomainId, LocalityId, ProtocolLimits, RunId},
    tcp::{TcpConfig, TcpRendezvous, TcpRendezvousConfig, TcpTransport},
};
use std::{
    future::Future,
    net::{SocketAddr, TcpListener},
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};
use support::{Counter, CounterAdd, Increment};

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn cooperative_block_on<T>(
    runtime: &mut Runtime,
    future: impl Future<Output = Result<T, RuntimeError>>,
    finished: &AtomicUsize,
) -> Result<T, RuntimeError> {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    let mut result = None;
    loop {
        runtime.progress(64)?;
        if result.is_none() {
            if let Poll::Ready(value) = Future::poll(Pin::as_mut(&mut future), &mut context) {
                result = Some(value);
                finished.fetch_add(1, Ordering::AcqRel);
            }
        }
        if finished.load(Ordering::Acquire) == 2 {
            return result.unwrap();
        }
        std::thread::yield_now();
    }
}

fn reserve_endpoint() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

fn run_round(run: u128) {
    let run_id = RunId::new(run).unwrap();
    let rendezvous_endpoint = reserve_endpoint();
    let endpoints = [reserve_endpoint(), reserve_endpoint()];
    let barrier = Arc::new(Barrier::new(2));
    let finished = Arc::new(AtomicUsize::new(0));
    let created = Arc::new(AtomicUsize::new(0));
    let called = Arc::new(AtomicUsize::new(0));
    let migrated = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        let joins: Vec<_> = endpoints
            .into_iter()
            .enumerate()
            .map(|(index, local_endpoint)| {
                let barrier = Arc::clone(&barrier);
                let finished = Arc::clone(&finished);
                let created = Arc::clone(&created);
                let called = Arc::clone(&called);
                let migrated = Arc::clone(&migrated);
                scope.spawn(move || {
                    let mut builder = Runtime::builder(
                        run_id,
                        ProtocolLimits::default(),
                        RuntimeLimits {
                            default_deadline: Duration::from_secs(2),
                            shutdown_timeout: Duration::from_secs(2),
                            ..RuntimeLimits::default()
                        },
                    )
                    .unwrap();
                    support::register(&mut builder).unwrap();
                    let membership = TcpRendezvous::join(TcpRendezvousConfig {
                        run_id,
                        rendezvous_endpoint,
                        local_endpoint,
                        expected_localities: 2,
                        coordinator: index == 0,
                        timeout: Duration::from_secs(2),
                    })
                    .unwrap();
                    let local_id = membership.local_id;
                    let (handle, driver) = TcpTransport::connect(TcpConfig {
                        local_id,
                        endpoints: membership.endpoints,
                        hello: builder.hello().unwrap(),
                        bootstrap_timeout: Duration::from_secs(2),
                        io_timeout: Duration::from_secs(2),
                    })
                    .unwrap();
                    let mut runtime = builder.start(handle, driver).unwrap();
                    let peer = LocalityId::new(1 - local_id.get());
                    let future = runtime
                        .spawn_on(Place::new(peer, DomainId::DEFAULT), Increment(run as u64))
                        .unwrap();
                    assert_eq!(
                        cooperative_block_on(&mut runtime, future, &finished).unwrap(),
                        run as u64 + 1
                    );
                    let create = runtime
                        .client()
                        .create_at(Place::new(peer, DomainId::DEFAULT), Counter(run as u64))
                        .unwrap();
                    let remote = cooperative_block_on(&mut runtime, create, &created).unwrap();
                    let migration = remote
                        .migrate_to(Place::new(local_id, DomainId::DEFAULT))
                        .unwrap();
                    let report = cooperative_block_on(&mut runtime, migration, &migrated).unwrap();
                    assert_eq!(report.to.locality(), local_id);
                    let call = remote.call_write(CounterAdd(5)).unwrap();
                    assert_eq!(
                        cooperative_block_on(&mut runtime, call, &called).unwrap(),
                        run as u64 + 5
                    );
                    drop(remote);
                    // The peer releases its handle asynchronously, so wait for the
                    // condition instead of spinning through a fixed iteration count.
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while runtime.stats().objects.live_objects != 0 && Instant::now() < deadline {
                        runtime.progress(64).unwrap();
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    assert_eq!(
                        runtime.stats().objects.live_objects,
                        0,
                        "live_objects did not reach 0 within 5 s of dropping the remote"
                    );
                    barrier.wait();
                    for _ in 0..128 {
                        runtime.progress(64).unwrap();
                        std::thread::yield_now();
                    }
                    barrier.wait();
                    let report = runtime.shutdown().unwrap();
                    assert_eq!(report.stats.pending_calls, 0);
                    assert_eq!(report.stats.transport.retained_bytes(), 0);
                })
            })
            .collect();
        for join in joins {
            join.join().unwrap();
        }
    });
}

fn main() {
    run_round(1);
    run_round(2);
}
