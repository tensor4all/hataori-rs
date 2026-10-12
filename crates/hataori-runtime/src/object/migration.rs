use super::*;

pub(super) const MIGRATION_SCHEMA_ID: u64 = 0x484d_4947;
pub(crate) const MIGRATION_HEADER_BYTES: usize = 133;
pub(crate) const MIGRATION_FREEZE_ACTION_ID: u128 = 0xc4ea_7e02_0000_0000_0000_0000_0000_0001;
pub(crate) const MIGRATION_PREPARE_ACTION_ID: u128 = 0xc4ea_7e02_0000_0000_0000_0000_0000_0002;
pub(crate) const MIGRATION_COMMIT_ACTION_ID: u128 = 0xc4ea_7e02_0000_0000_0000_0000_0000_0003;
pub(crate) const MIGRATION_ACTIVATE_ACTION_ID: u128 = 0xc4ea_7e02_0000_0000_0000_0000_0000_0004;
pub(crate) const MIGRATION_ROLLBACK_ACTION_ID: u128 = 0xc4ea_7e02_0000_0000_0000_0000_0000_0005;
pub(crate) const MIGRATION_RETIRE_ACTION_ID: u128 = 0xc4ea_7e02_0000_0000_0000_0000_0000_0006;

#[derive(Clone, Debug)]
pub(crate) struct MigrationPacket {
    pub(crate) migration: RequestId,
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) authority: LocalityId,
    pub(crate) from: ObjectLocation,
    pub(crate) destination: Place,
    pub(crate) snapshot_version: u32,
    pub(crate) snapshot_schema: u64,
    pub(crate) snapshot: Segments,
}

impl WireValue for MigrationPacket {
    const SCHEMA_ID: u64 = MIGRATION_SCHEMA_ID;

    fn encode(mut self) -> Result<Segments, ActionError> {
        let mut header = Vec::with_capacity(MIGRATION_HEADER_BYTES);
        header.extend_from_slice(b"HMIG");
        header.push(1);
        header.extend_from_slice(&self.migration.origin.get().to_le_bytes());
        header.extend_from_slice(&self.migration.sequence.to_le_bytes());
        header.extend_from_slice(&self.object.run().get().to_le_bytes());
        header.extend_from_slice(&self.object.unique().to_le_bytes());
        header.extend_from_slice(&self.object_type.get().to_le_bytes());
        header.extend_from_slice(&self.authority.get().to_le_bytes());
        header.extend_from_slice(&self.from.locality().get().to_le_bytes());
        header.extend_from_slice(&self.from.domain().get().to_le_bytes());
        header.extend_from_slice(&self.from.slot().to_le_bytes());
        header.extend_from_slice(&self.from.generation().to_le_bytes());
        header.extend_from_slice(&self.from.epoch().to_le_bytes());
        header.extend_from_slice(&self.destination.locality.get().to_le_bytes());
        header.extend_from_slice(&self.destination.domain.get().to_le_bytes());
        header.extend_from_slice(&self.snapshot_version.to_le_bytes());
        header.extend_from_slice(&self.snapshot_schema.to_le_bytes());
        self.snapshot.insert(0, header);
        Ok(self.snapshot)
    }

    fn decode(mut segments: Segments) -> Result<Self, ActionError> {
        if segments.is_empty() || segments[0].len() != MIGRATION_HEADER_BYTES {
            return Err(ActionError::codec("invalid migration header length"));
        }
        let header = segments.remove(0);
        if &header[0..4] != b"HMIG" || header[4] != 1 {
            return Err(ActionError::codec("invalid migration header"));
        }
        let migration = RequestId::new(
            LocalityId::new(u64::from_le_bytes(header[5..13].try_into().unwrap())),
            u64::from_le_bytes(header[13..21].try_into().unwrap()),
        )
        .map_err(|_| ActionError::codec("invalid migration id"))?;
        let run = RunId::new(u128::from_le_bytes(header[21..37].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid migration run"))?;
        let object = ObjectId::new(run, u128::from_le_bytes(header[37..53].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid migration object"))?;
        let object_type =
            ObjectTypeId::new(u128::from_le_bytes(header[53..69].try_into().unwrap()))
                .map_err(|_| ActionError::codec("invalid migration object type"))?;
        let authority = LocalityId::new(u64::from_le_bytes(header[69..77].try_into().unwrap()));
        let from = ObjectLocation::new(
            LocalityId::new(u64::from_le_bytes(header[77..85].try_into().unwrap())),
            DomainId::new(u32::from_le_bytes(header[85..89].try_into().unwrap())),
            u64::from_le_bytes(header[89..97].try_into().unwrap()),
            u32::from_le_bytes(header[97..101].try_into().unwrap()),
            u64::from_le_bytes(header[101..109].try_into().unwrap()),
        )
        .map_err(|_| ActionError::codec("invalid migration source"))?;
        let destination = Place::new(
            LocalityId::new(u64::from_le_bytes(header[109..117].try_into().unwrap())),
            DomainId::new(u32::from_le_bytes(header[117..121].try_into().unwrap())),
        );
        Ok(Self {
            migration,
            object,
            object_type,
            authority,
            from,
            destination,
            snapshot_version: u32::from_le_bytes(header[121..125].try_into().unwrap()),
            snapshot_schema: u64::from_le_bytes(header[125..133].try_into().unwrap()),
            snapshot: segments,
        })
    }
}

enum MigrationState {
    Ready(Option<Result<MigrationReport, RuntimeError>>),
    Freeze(RemoteFuture<MigrationPacket>),
    Prepare(RemoteFuture<MigrationPacket>),
    Commit(RemoteFuture<MigrationPacket>),
    Activate(RemoteFuture<MigrationPacket>),
    Retire(RemoteFuture<MigrationPacket>),
    RollbackSource {
        future: RemoteFuture<MigrationPacket>,
        error: RuntimeError,
        destination: bool,
    },
    RollbackDestination {
        future: RemoteFuture<MigrationPacket>,
        error: RuntimeError,
    },
    Done,
}

#[must_use = "a migration future must be driven or dropped"]
pub struct MigrationFuture<T: MobileObject> {
    client: Option<RuntimeClient>,
    service: Option<Arc<ObjectService>>,
    lease: Option<Arc<LocalLease>>,
    packet: MigrationPacket,
    authority: ObjectLocation,
    original: ObjectLocation,
    prepared: Option<ObjectLocation>,
    snapshot_bytes: usize,
    redirects: usize,
    committed: bool,
    state: MigrationState,
    marker: PhantomData<T>,
}

impl<T: MobileObject> std::fmt::Debug for MigrationFuture<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MigrationFuture")
            .field("object", &self.packet.object)
            .field("migration", &self.packet.migration)
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

impl<T: MobileObject> MigrationFuture<T> {
    pub(crate) fn new(
        client: RuntimeClient,
        service: Arc<ObjectService>,
        lease: Arc<LocalLease>,
        packet: MigrationPacket,
        authority: ObjectLocation,
        future: RemoteFuture<MigrationPacket>,
    ) -> Self {
        Self {
            client: Some(client),
            service: Some(service),
            lease: Some(lease),
            original: packet.from,
            packet,
            authority,
            prepared: None,
            snapshot_bytes: 0,
            redirects: 0,
            committed: false,
            state: MigrationState::Freeze(future),
            marker: PhantomData,
        }
    }

    pub(crate) fn ready(report: MigrationReport, object_type: ObjectTypeId) -> Self {
        // INVARIANT: this packet is an inert placeholder for the Ready state;
        // it is never encoded, registered, or submitted to an ObjectService.
        let packet = MigrationPacket {
            migration: RequestId::new(report.from.locality(), 1).unwrap(),
            object: report.object,
            object_type,
            authority: report.from.locality(),
            from: report.from,
            destination: Place::new(report.to.locality(), report.to.domain()),
            snapshot_version: 1,
            snapshot_schema: 1,
            snapshot: Vec::new(),
        };
        Self {
            client: None,
            service: None,
            lease: None,
            packet,
            authority: report.from,
            original: report.from,
            prepared: Some(report.to),
            snapshot_bytes: report.snapshot_bytes,
            redirects: 0,
            committed: true,
            state: MigrationState::Ready(Some(Ok(report))),
            marker: PhantomData,
        }
    }

    fn start_rollback(
        &mut self,
        error: RuntimeError,
        destination: bool,
    ) -> Result<(), RuntimeError> {
        let error = self.migration_error(error);
        let client = self.client.as_ref().expect("active migration has client");
        let future = client.migration_step(
            Place::new(self.original.locality(), self.original.domain()),
            MIGRATION_ROLLBACK_ACTION_ID,
            self.packet.clone(),
        )?;
        self.state = MigrationState::RollbackSource {
            future,
            error,
            destination,
        };
        Ok(())
    }

    fn migration_error(&self, error: RuntimeError) -> RuntimeError {
        match error {
            RuntimeError::Migration { .. } => error,
            error => RuntimeError::Migration {
                request: self.packet.migration,
                message: error.to_string(),
            },
        }
    }

    fn finish(&mut self, outcome: MigrationOutcome) {
        if let Some(service) = self.service.take() {
            service.finish_migration(self.packet.migration, outcome, self.snapshot_bytes);
        }
        self.client = None;
        self.lease = None;
    }
}

impl<T: MobileObject> Future for MigrationFuture<T> {
    type Output = Result<MigrationReport, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match &mut self.state {
                MigrationState::Ready(result) => {
                    let result = result.take().unwrap_or_else(|| {
                        Err(RuntimeError::Protocol(
                            "migration future polled after completion".into(),
                        ))
                    });
                    self.state = MigrationState::Done;
                    return Poll::Ready(result);
                }
                MigrationState::Freeze(future) => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(packet)) => {
                        let Some(snapshot_bytes) = packet
                            .snapshot
                            .iter()
                            .try_fold(0_usize, |total, segment| total.checked_add(segment.len()))
                        else {
                            let error = RuntimeError::SnapshotTooLarge {
                                bytes: usize::MAX,
                                limit: self.service.as_ref().unwrap().limits.max_snapshot_bytes,
                            };
                            if let Err(rollback) = self.start_rollback(error, false) {
                                self.finish(MigrationOutcome::Failed);
                                return Poll::Ready(Err(rollback));
                            }
                            continue;
                        };
                        self.snapshot_bytes = snapshot_bytes;
                        self.packet = packet.clone();
                        let client = self.client.as_ref().unwrap();
                        match client.migration_step(
                            self.packet.destination,
                            MIGRATION_PREPARE_ACTION_ID,
                            packet,
                        ) {
                            Ok(future) => self.state = MigrationState::Prepare(future),
                            Err(error) => {
                                if let Err(rollback) = self.start_rollback(error, false) {
                                    self.finish(MigrationOutcome::Failed);
                                    return Poll::Ready(Err(rollback));
                                }
                            }
                        }
                    }
                    Poll::Ready(Err(RuntimeError::Moved { object, location }))
                        if object == self.packet.object
                            && location.epoch() > self.packet.from.epoch()
                            && self.redirects
                                < self.service.as_ref().unwrap().limits.max_redirects =>
                    {
                        self.redirects += 1;
                        self.packet.from = location;
                        self.original = location;
                        self.service
                            .as_ref()
                            .unwrap()
                            .cache_location(object, location);
                        match self.client.as_ref().unwrap().migration_step(
                            Place::new(location.locality(), location.domain()),
                            MIGRATION_FREEZE_ACTION_ID,
                            self.packet.clone(),
                        ) {
                            Ok(future) => self.state = MigrationState::Freeze(future),
                            Err(error) => {
                                self.finish(MigrationOutcome::RolledBack);
                                return Poll::Ready(Err(error));
                            }
                        }
                    }
                    Poll::Ready(Err(error)) => {
                        let error = self.migration_error(error);
                        self.finish(MigrationOutcome::RolledBack);
                        self.state = MigrationState::Done;
                        return Poll::Ready(Err(error));
                    }
                },
                MigrationState::Prepare(future) => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(packet)) => {
                        self.prepared = Some(packet.from);
                        self.packet = packet.clone();
                        let place = Place::new(self.authority.locality(), self.authority.domain());
                        match self.client.as_ref().unwrap().migration_step(
                            place,
                            MIGRATION_COMMIT_ACTION_ID,
                            packet,
                        ) {
                            Ok(future) => self.state = MigrationState::Commit(future),
                            Err(error) => {
                                if let Err(rollback) = self.start_rollback(error, true) {
                                    self.finish(MigrationOutcome::Failed);
                                    return Poll::Ready(Err(rollback));
                                }
                            }
                        }
                    }
                    Poll::Ready(Err(error)) => {
                        if let Err(rollback) = self.start_rollback(error, false) {
                            self.finish(MigrationOutcome::Failed);
                            return Poll::Ready(Err(rollback));
                        }
                    }
                },
                MigrationState::Commit(future) => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(packet)) => {
                        self.committed = true;
                        self.packet = packet.clone();
                        match self.client.as_ref().unwrap().migration_step(
                            self.packet.destination,
                            MIGRATION_ACTIVATE_ACTION_ID,
                            packet,
                        ) {
                            Ok(future) => self.state = MigrationState::Activate(future),
                            Err(error) => {
                                self.client.as_ref().unwrap().fail_committed_migration();
                                self.finish(MigrationOutcome::Failed);
                                return Poll::Ready(Err(RuntimeError::MigrationCommittedFailure {
                                    object: self.packet.object,
                                    message: error.to_string(),
                                }));
                            }
                        }
                    }
                    Poll::Ready(Err(error)) => {
                        if let Err(rollback) = self.start_rollback(error, true) {
                            self.finish(MigrationOutcome::Failed);
                            return Poll::Ready(Err(rollback));
                        }
                    }
                },
                MigrationState::Activate(future) => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(packet)) => {
                        self.packet = packet.clone();
                        if self.original.locality() == packet.from.locality() {
                            let to = self.prepared.expect("commit requires prepared location");
                            self.service
                                .as_ref()
                                .unwrap()
                                .cache_location(self.packet.object, to);
                            let report = MigrationReport {
                                object: self.packet.object,
                                from: self.original,
                                to,
                                snapshot_bytes: self.snapshot_bytes,
                            };
                            self.finish(MigrationOutcome::Completed);
                            self.state = MigrationState::Done;
                            return Poll::Ready(Ok(report));
                        }
                        let source = Place::new(self.original.locality(), self.original.domain());
                        match self.client.as_ref().unwrap().migration_step(
                            source,
                            MIGRATION_RETIRE_ACTION_ID,
                            packet,
                        ) {
                            Ok(future) => self.state = MigrationState::Retire(future),
                            Err(error) => {
                                self.client.as_ref().unwrap().fail_committed_migration();
                                self.finish(MigrationOutcome::Failed);
                                return Poll::Ready(Err(RuntimeError::MigrationCommittedFailure {
                                    object: self.packet.object,
                                    message: error.to_string(),
                                }));
                            }
                        }
                    }
                    Poll::Ready(Err(error)) => {
                        self.client.as_ref().unwrap().fail_committed_migration();
                        self.finish(MigrationOutcome::Failed);
                        return Poll::Ready(Err(RuntimeError::MigrationCommittedFailure {
                            object: self.packet.object,
                            message: error.to_string(),
                        }));
                    }
                },
                MigrationState::Retire(future) => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(_)) => {
                        let to = self.prepared.expect("commit requires prepared location");
                        self.service
                            .as_ref()
                            .unwrap()
                            .cache_location(self.packet.object, to);
                        let report = MigrationReport {
                            object: self.packet.object,
                            from: self.original,
                            to,
                            snapshot_bytes: self.snapshot_bytes,
                        };
                        self.finish(MigrationOutcome::Completed);
                        self.state = MigrationState::Done;
                        return Poll::Ready(Ok(report));
                    }
                    Poll::Ready(Err(error)) => {
                        self.client.as_ref().unwrap().fail_committed_migration();
                        self.finish(MigrationOutcome::Failed);
                        return Poll::Ready(Err(RuntimeError::MigrationCommittedFailure {
                            object: self.packet.object,
                            message: error.to_string(),
                        }));
                    }
                },
                MigrationState::RollbackSource {
                    future,
                    error,
                    destination,
                } => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(rollback)) => {
                        let message = format!("{}; source rollback failed: {rollback}", error);
                        self.finish(MigrationOutcome::Failed);
                        return Poll::Ready(Err(RuntimeError::Migration {
                            request: self.packet.migration,
                            message,
                        }));
                    }
                    Poll::Ready(Ok(_)) if *destination => {
                        let error = error.clone();
                        match self.client.as_ref().unwrap().migration_step(
                            self.packet.destination,
                            MIGRATION_ROLLBACK_ACTION_ID,
                            self.packet.clone(),
                        ) {
                            Ok(future) => {
                                self.state = MigrationState::RollbackDestination { future, error }
                            }
                            Err(_) => {
                                self.finish(MigrationOutcome::Failed);
                                return Poll::Ready(Err(error));
                            }
                        }
                    }
                    Poll::Ready(Ok(_)) => {
                        let error = error.clone();
                        self.finish(MigrationOutcome::RolledBack);
                        self.state = MigrationState::Done;
                        return Poll::Ready(Err(error));
                    }
                },
                MigrationState::RollbackDestination { future, error } => {
                    match Pin::new(future).poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Err(rollback)) => {
                            let message =
                                format!("{}; destination rollback failed: {rollback}", error);
                            self.finish(MigrationOutcome::Failed);
                            return Poll::Ready(Err(RuntimeError::Migration {
                                request: self.packet.migration,
                                message,
                            }));
                        }
                        Poll::Ready(Ok(_)) => {
                            let error = error.clone();
                            self.finish(MigrationOutcome::RolledBack);
                            self.state = MigrationState::Done;
                            return Poll::Ready(Err(error));
                        }
                    }
                }
                MigrationState::Done => {
                    return Poll::Ready(Err(RuntimeError::Protocol(
                        "migration future polled after completion".into(),
                    )));
                }
            }
        }
    }
}

impl<T: MobileObject> Unpin for MigrationFuture<T> {}

impl<T: MobileObject> Drop for MigrationFuture<T> {
    fn drop(&mut self) {
        if self.service.is_some() {
            if self.committed {
                if let Some(client) = &self.client {
                    client.fail_committed_migration();
                }
            }
            self.finish(if self.committed {
                MigrationOutcome::Failed
            } else {
                MigrationOutcome::RolledBack
            });
        }
    }
}

struct PreparedPermit<'a> {
    count: &'a AtomicU64,
    committed: bool,
}

impl PreparedPermit<'_> {
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for PreparedPermit<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.count.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl ObjectService {
    pub(super) fn migration_freeze_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            service
                .freeze_migration(MigrationPacket::decode(segments)?)?
                .encode()
        })
    }

    pub(super) fn migration_prepare_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            service
                .prepare_migration(MigrationPacket::decode(segments)?)?
                .encode()
        })
    }

    pub(super) fn migration_commit_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            service
                .commit_migration(MigrationPacket::decode(segments)?)?
                .encode()
        })
    }

    pub(super) fn migration_activate_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            service
                .activate_migration(MigrationPacket::decode(segments)?)?
                .encode()
        })
    }

    pub(super) fn migration_rollback_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            service
                .rollback_migration(MigrationPacket::decode(segments)?)?
                .encode()
        })
    }

    pub(super) fn migration_retire_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            service
                .retire_migration(MigrationPacket::decode(segments)?)?
                .encode()
        })
    }

    pub(crate) fn migration_metadata(
        &self,
        object_type: ObjectTypeId,
    ) -> Result<(Mobility, u32, u64), RuntimeError> {
        let registration = self
            .registry
            .types
            .get(&object_type)
            .ok_or(RuntimeError::UnknownObjectType(object_type))?;
        Ok((
            registration.mobility,
            registration.snapshot_version,
            registration.snapshot_schema,
        ))
    }

    pub(crate) fn begin_migration(&self, migration: RequestId) -> Result<(), RuntimeError> {
        let mut active = self.active_migrations.lock().unwrap();
        if active.len() >= self.limits.max_migrations {
            return Err(RuntimeError::ResourceExhausted {
                resource: crate::ResourceKind::Migrations,
                limit: self.limits.max_migrations,
            });
        }
        if !active.insert(migration) {
            return Err(RuntimeError::Protocol("duplicate migration id".into()));
        }
        Ok(())
    }

    pub(crate) fn finish_migration(
        &self,
        migration: RequestId,
        outcome: MigrationOutcome,
        bytes: usize,
    ) {
        self.active_migrations.lock().unwrap().remove(&migration);
        match outcome {
            MigrationOutcome::Completed => {
                self.completed_migrations.fetch_add(1, Ordering::Relaxed);
                self.migrated_bytes
                    .fetch_add(bytes as u64, Ordering::Relaxed);
            }
            MigrationOutcome::RolledBack => {
                self.rolled_back_migrations.fetch_add(1, Ordering::Relaxed);
            }
            MigrationOutcome::Failed => {
                self.failed_migrations.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn freeze_migration(
        &self,
        mut packet: MigrationPacket,
    ) -> Result<MigrationPacket, ActionError> {
        let entry = self.entry_for_migration(&packet)?;
        if entry.location() != packet.from || entry.authority != packet.authority {
            return Err(ActionError::user("migration source changed"));
        }
        let registration = self
            .registry
            .types
            .get(&packet.object_type)
            .ok_or_else(|| ActionError::user("unknown migration object type"))?;
        let freezer = registration
            .freezer
            .as_ref()
            .ok_or_else(|| ActionError::user("object is pinned"))?;
        if packet.snapshot_version != registration.snapshot_version
            || packet.snapshot_schema != registration.snapshot_schema
        {
            return Err(ActionError::user("migration snapshot schema mismatch"));
        }
        {
            let mut lifecycle = entry.lifecycle.lock().unwrap();
            match *lifecycle {
                ObjectLifecycle::Resident => {
                    *lifecycle = ObjectLifecycle::Freezing {
                        migration: packet.migration,
                        expires_at: Instant::now() + self.limits.forwarding_ttl,
                    }
                }
                ObjectLifecycle::Freezing { migration, .. } if migration == packet.migration => {}
                _ => return Err(ActionError::user("migration already in progress")),
            }
        }
        let state = entry.resident_state()?;
        let snapshot = match catch_unwind(AssertUnwindSafe(|| freezer(&state))) {
            Ok(Ok(snapshot)) => snapshot,
            Ok(Err(error)) => {
                *entry.lifecycle.lock().unwrap() = ObjectLifecycle::Resident;
                return Err(error);
            }
            Err(_) => {
                *entry.lifecycle.lock().unwrap() = ObjectLifecycle::Resident;
                return Err(ActionError::Panic);
            }
        };
        let bytes = match self.validate_snapshot(&snapshot) {
            Ok(bytes) => bytes,
            Err(error) => {
                *entry.lifecycle.lock().unwrap() = ObjectLifecycle::Resident;
                return Err(error);
            }
        };
        *entry.snapshot_bytes.lock().unwrap() = bytes;
        packet.snapshot = snapshot;
        Ok(packet)
    }

    fn prepare_migration(
        &self,
        mut packet: MigrationPacket,
    ) -> Result<MigrationPacket, ActionError> {
        if packet.destination.locality != self.local_id {
            return Err(ActionError::user("migration prepared at wrong locality"));
        }
        let bytes = self.validate_snapshot(&packet.snapshot)?;
        let registration = self
            .registry
            .types
            .get(&packet.object_type)
            .ok_or_else(|| ActionError::user("unknown migration object type"))?;
        if registration.mobility == Mobility::Pinned
            || packet.snapshot_version != registration.snapshot_version
            || packet.snapshot_schema != registration.snapshot_schema
        {
            return Err(ActionError::user("migration snapshot schema mismatch"));
        }
        self.prepared_count
            .try_update_compat(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < self.limits.max_prepared_migrations as u64).then_some(count + 1)
            })
            .map_err(|_| ActionError::user("prepared migration limit reached"))?;
        let permit = PreparedPermit {
            count: &self.prepared_count,
            committed: false,
        };
        {
            let entries = self.entries.lock().unwrap();
            if entries.len() >= self.limits.max_objects && !entries.contains_key(&packet.object) {
                return Err(ActionError::user("object store is full"));
            }
        }
        let epoch = packet
            .from
            .epoch()
            .checked_add(1)
            .ok_or_else(|| ActionError::user("object epoch overflow"))?;
        let mut slot = self.next_slot.lock().unwrap();
        let location =
            ObjectLocation::new(self.local_id, packet.destination.domain, *slot, 1, epoch)
                .map_err(|_| ActionError::user("invalid migration destination"))?;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| ActionError::user("object slot overflow"))?;
        drop(slot);
        let restore = registration
            .restorer
            .as_ref()
            .ok_or_else(|| ActionError::user("object is pinned"))?;
        let snapshot = std::mem::take(&mut packet.snapshot);
        let restored = Arc::new(restore(
            snapshot,
            RestoreContext {
                destination: packet.destination,
            },
        )?);
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.limits.max_objects && !entries.contains_key(&packet.object) {
            return Err(ActionError::user("object store is full"));
        }
        let entry = entries.entry(packet.object).or_insert_with(|| {
            Arc::new(ObjectEntry {
                object_type: packet.object_type,
                authority: packet.authority,
                location: Mutex::new(packet.from),
                state: Mutex::new(None),
                lifecycle: Mutex::new(ObjectLifecycle::Prepared {
                    migration: packet.migration,
                    expires_at: Instant::now() + self.limits.forwarding_ttl,
                }),
                prepared: Mutex::new(None),
                snapshot_bytes: Mutex::new(0),
                leases: Mutex::new(HashMap::new()),
                roots: Mutex::new(0),
                in_flight: Mutex::new(0),
                placement_tickets: Mutex::new(0),
                admission: Mutex::new(Admission {
                    readers: 0,
                    writer: false,
                    queue: VecDeque::new(),
                }),
            })
        });
        if entry.object_type != packet.object_type || entry.authority != packet.authority {
            return Err(ActionError::user("migration destination conflict"));
        }
        let mut prepared = entry.prepared.lock().unwrap();
        if let Some(existing) = prepared.as_ref() {
            if existing.migration != packet.migration || existing.location != location {
                return Err(ActionError::user("conflicting prepared migration"));
            }
        } else {
            *prepared = Some(PreparedState {
                migration: packet.migration,
                location,
                state: restored,
                snapshot_bytes: bytes,
                expires_at: Instant::now() + self.limits.forwarding_ttl,
            });
            permit.commit();
        }
        packet.from = location;
        Ok(packet)
    }

    fn commit_migration(&self, packet: MigrationPacket) -> Result<MigrationPacket, ActionError> {
        if packet.authority != self.local_id {
            return Err(ActionError::user("migration committed at wrong authority"));
        }
        let entry = self.entry_for_migration(&packet)?;
        // INVARIANT: the scan is capped by ObjectLimits::max_objects.
        let forwarders = self
            .entries
            .lock()
            .unwrap()
            .values()
            .filter(|entry| {
                matches!(
                    *entry.lifecycle.lock().unwrap(),
                    ObjectLifecycle::Forwarding { .. } | ObjectLifecycle::Retiring { .. }
                )
            })
            .count();
        if !matches!(
            *entry.lifecycle.lock().unwrap(),
            ObjectLifecycle::Forwarding { .. } | ObjectLifecycle::Retiring { .. }
        ) && forwarders >= self.limits.max_forwarders
        {
            return Err(ActionError::user("forwarding entry limit reached"));
        }
        let old = entry.location();
        if old.epoch().checked_add(1) != Some(packet.from.epoch()) {
            return Err(ActionError::user("migration epoch conflict"));
        }
        *entry.location.lock().unwrap() = packet.from;
        *entry.lifecycle.lock().unwrap() = ObjectLifecycle::Forwarding {
            location: packet.from,
            expires_at: Instant::now() + self.limits.forwarding_ttl,
        };
        self.cache_location(packet.object, packet.from);
        Ok(packet)
    }

    fn activate_migration(&self, packet: MigrationPacket) -> Result<MigrationPacket, ActionError> {
        let entry = self.entry_for_migration(&packet)?;
        let mut lifecycle = entry.lifecycle.lock().unwrap();
        let prepared = entry
            .prepared
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| ActionError::user("migration destination was not prepared"))?;
        self.prepared_count.fetch_sub(1, Ordering::AcqRel);
        if prepared.migration != packet.migration || prepared.location != packet.from {
            return Err(ActionError::user("prepared migration mismatch"));
        }
        *entry.location.lock().unwrap() = prepared.location;
        *entry.state.lock().unwrap() = Some(prepared.state);
        *entry.snapshot_bytes.lock().unwrap() = prepared.snapshot_bytes;
        *lifecycle = ObjectLifecycle::Resident;
        drop(lifecycle);
        self.cache_location(packet.object, prepared.location);
        Ok(packet)
    }

    fn rollback_migration(&self, packet: MigrationPacket) -> Result<MigrationPacket, ActionError> {
        let entry = self.entries.lock().unwrap().get(&packet.object).cloned();
        if let Some(entry) = entry {
            let mut prepared = entry.prepared.lock().unwrap();
            if prepared
                .as_ref()
                .is_some_and(|state| state.migration == packet.migration)
            {
                *prepared = None;
                self.prepared_count.fetch_sub(1, Ordering::AcqRel);
            }
            drop(prepared);
            let mut lifecycle = entry.lifecycle.lock().unwrap();
            if matches!(
                *lifecycle,
                ObjectLifecycle::Freezing { migration, .. } if migration == packet.migration
            ) {
                *lifecycle = ObjectLifecycle::Resident;
                *entry.snapshot_bytes.lock().unwrap() = 0;
            }
        }
        Ok(packet)
    }

    fn retire_migration(&self, packet: MigrationPacket) -> Result<MigrationPacket, ActionError> {
        let entry = self.entry_for_migration(&packet)?;
        if entry.location().epoch() > packet.from.epoch() {
            return Ok(packet);
        }
        *entry.state.lock().unwrap() = None;
        *entry.snapshot_bytes.lock().unwrap() = 0;
        *entry.location.lock().unwrap() = packet.from;
        *entry.lifecycle.lock().unwrap() = ObjectLifecycle::Forwarding {
            location: packet.from,
            expires_at: Instant::now() + self.limits.forwarding_ttl,
        };
        self.cache_location(packet.object, packet.from);
        self.collect(packet.object, Instant::now());
        Ok(packet)
    }

    fn entry_for_migration(
        &self,
        packet: &MigrationPacket,
    ) -> Result<Arc<ObjectEntry>, ActionError> {
        let entry = self
            .entries
            .lock()
            .unwrap()
            .get(&packet.object)
            .cloned()
            .ok_or_else(|| ActionError::user("migration object not found"))?;
        if entry.object_type != packet.object_type {
            return Err(ActionError::user("migration object type mismatch"));
        }
        Ok(entry)
    }

    pub(super) fn validate_snapshot(&self, snapshot: &Segments) -> Result<usize, ActionError> {
        if snapshot.len() > self.limits.max_snapshot_segments {
            return Err(ActionError::user(
                "migration snapshot segment limit exceeded",
            ));
        }
        let bytes = snapshot
            .iter()
            .try_fold(0_usize, |total, segment| total.checked_add(segment.len()))
            .ok_or_else(|| ActionError::user("migration snapshot size overflow"))?;
        if bytes > self.limits.max_snapshot_bytes {
            return Err(ActionError::user("migration snapshot byte limit exceeded"));
        }
        Ok(bytes)
    }
}
