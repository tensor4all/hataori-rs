use crate::{
    action::{Action, ActionRegistry, ActionValue, RegisteredAction, Segments},
    domain::{ActionJob, ObjectJob},
    ActionError, Place, RemoteFuture, RuntimeClient, RuntimeError, SpawnOptions, WireValue,
};
use hataori_runtime_foundation::protocol::{
    ActionId, DomainId, LocalityId, ObjectId, ObjectLocation, ObjectTypeId, RequestId, RunId,
};
use std::{
    any::Any,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    future::Future,
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock, Weak,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

const OBJECT_MAGIC: [u8; 4] = *b"HOBJ";
const OBJECT_VERSION: u8 = 1;
const OBJECT_HEADER_BYTES: usize = 76;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObjectConcurrency {
    Exclusive,
    ReadWrite,
}

pub trait DistributedObject: WireValue {
    const TYPE_ID: u128;
}

pub trait ObjectReadAction<T: DistributedObject>: WireValue {
    const ID: u128;
    type Output: WireValue;
    fn execute(self, state: &T) -> Result<Self::Output, ActionError>;
}

pub trait ObjectWriteAction<T: DistributedObject>: WireValue {
    const ID: u128;
    type Output: WireValue;
    fn execute(self, state: &mut T) -> Result<Self::Output, ActionError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Mobility {
    Pinned,
    Reconstructible,
    Migratable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RestoreContext {
    destination: Place,
}

impl RestoreContext {
    pub const fn destination(self) -> Place {
        self.destination
    }
}

/// Logical object state that can be frozen and restored at another `Place`.
pub trait MobileObject: DistributedObject {
    const MOBILITY: Mobility;
    const SNAPSHOT_VERSION: u32;
    type Snapshot: WireValue;

    /// Produces a versioned logical snapshot while runtime admission is closed.
    ///
    /// # Errors
    /// Returns `ActionError` when logical state cannot be snapshotted.
    fn freeze(&mut self) -> Result<Self::Snapshot, ActionError>;

    /// Restores logical state and rebuilds destination-local resources.
    ///
    /// # Errors
    /// Returns `ActionError` when the snapshot is invalid or a required
    /// destination-local resource cannot be reconstructed.
    fn restore(snapshot: Self::Snapshot, context: RestoreContext) -> Result<Self, ActionError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObjectAccess {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementFallback {
    Any,
    Place(Place),
    Reject,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectLimits {
    pub max_objects: usize,
    pub max_mailbox_per_object: usize,
    pub max_resolver_entries: usize,
    pub max_local_leases: usize,
    pub max_roots: usize,
    pub max_placement_tickets: usize,
    pub max_transfers: usize,
    pub max_migrations: usize,
    pub max_prepared_migrations: usize,
    pub max_forwarders: usize,
    pub max_redirects: usize,
    pub max_snapshot_bytes: usize,
    pub max_snapshot_segments: usize,
    pub forwarding_ttl: Duration,
    pub lease_renew_interval: Duration,
    pub lease_ttl: Duration,
    pub lease_grace: Duration,
}

impl Default for ObjectLimits {
    fn default() -> Self {
        Self {
            max_objects: 1024,
            max_mailbox_per_object: 64,
            max_resolver_entries: 1024,
            max_local_leases: 1024,
            max_roots: 1024,
            max_placement_tickets: 1024,
            max_transfers: 1024,
            max_migrations: 64,
            max_prepared_migrations: 64,
            max_forwarders: 1024,
            max_redirects: 8,
            max_snapshot_bytes: 8 * 1024 * 1024 - MIGRATION_HEADER_BYTES,
            max_snapshot_segments: 1023,
            forwarding_ttl: Duration::from_secs(60),
            lease_renew_interval: Duration::from_secs(10),
            lease_ttl: Duration::from_secs(30),
            lease_grace: Duration::from_secs(30),
        }
    }
}

impl ObjectLimits {
    pub fn validate(self) -> Result<Self, RuntimeError> {
        if self.max_objects == 0
            || self.max_mailbox_per_object == 0
            || self.max_resolver_entries == 0
            || self.max_local_leases == 0
            || self.max_roots == 0
            || self.max_placement_tickets == 0
            || self.max_transfers == 0
            || self.max_migrations == 0
            || self.max_prepared_migrations == 0
            || self.max_forwarders == 0
            || self.max_redirects == 0
            || self.max_snapshot_bytes == 0
            || self.max_snapshot_segments == 0
            || self.forwarding_ttl.is_zero()
            || self.lease_renew_interval.is_zero()
            || self.lease_ttl <= self.lease_renew_interval
            || self.lease_grace.is_zero()
        {
            return Err(RuntimeError::InvalidLimits("invalid object limits"));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MigrationReport {
    pub object: ObjectId,
    pub from: ObjectLocation,
    pub to: ObjectLocation,
    pub snapshot_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ObjectStats {
    pub live_objects: usize,
    pub leases: usize,
    pub suspect_leases: usize,
    pub roots: usize,
    pub queued_calls: usize,
    pub in_flight_calls: usize,
    pub placement_tickets: usize,
    pub resolver_entries: usize,
    pub resolver_bytes: usize,
    pub transfers: usize,
    pub active_migrations: usize,
    pub prepared_migrations: usize,
    pub forwarding_entries: usize,
    pub migration_snapshot_bytes: usize,
    pub transferred_snapshot_bytes: u64,
    pub completed_migrations: u64,
    pub rolled_back_migrations: u64,
    pub failed_migrations: u64,
    pub local_calls: u64,
    pub remote_calls: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ObjectEnvelope {
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) epoch: u64,
    pub(crate) lease_locality: LocalityId,
    pub(crate) domain: DomainId,
    pub(crate) rooted: bool,
}

impl ObjectEnvelope {
    pub(crate) fn encode(self, mut payload: Segments) -> Segments {
        let mut header = Vec::with_capacity(OBJECT_HEADER_BYTES);
        header.extend_from_slice(&OBJECT_MAGIC);
        header.push(OBJECT_VERSION);
        header.push(u8::from(self.rooted));
        header.extend_from_slice(&[0; 2]);
        header.extend_from_slice(&self.object.run().get().to_le_bytes());
        header.extend_from_slice(&self.object.unique().to_le_bytes());
        header.extend_from_slice(&self.object_type.get().to_le_bytes());
        header.extend_from_slice(&self.epoch.to_le_bytes());
        header.extend_from_slice(&self.lease_locality.get().to_le_bytes());
        header.extend_from_slice(&self.domain.get().to_le_bytes());
        debug_assert_eq!(header.len(), OBJECT_HEADER_BYTES);
        let mut result = Vec::with_capacity(payload.len() + 1);
        result.push(header);
        result.append(&mut payload);
        result
    }

    fn inspect(segments: &Segments) -> Result<Self, ActionError> {
        let header = segments
            .first()
            .ok_or_else(|| ActionError::codec("invalid object envelope length"))?;
        Self::decode_header(header)
    }

    fn decode(mut segments: Segments) -> Result<(Self, Segments), ActionError> {
        if segments.is_empty() || segments[0].len() != OBJECT_HEADER_BYTES {
            return Err(ActionError::codec("invalid object envelope length"));
        }
        let header = segments.remove(0);
        Ok((Self::decode_header(&header)?, segments))
    }

    fn decode_header(header: &[u8]) -> Result<Self, ActionError> {
        if header.len() != OBJECT_HEADER_BYTES {
            return Err(ActionError::codec("invalid object envelope length"));
        }
        if header[..4] != OBJECT_MAGIC || header[4] != OBJECT_VERSION || header[6..8] != [0; 2] {
            return Err(ActionError::codec("invalid object envelope header"));
        }
        let rooted = match header[5] {
            0 => false,
            1 => true,
            _ => return Err(ActionError::codec("invalid object root flag")),
        };
        let run = RunId::new(u128::from_le_bytes(header[8..24].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid object run id"))?;
        let object = ObjectId::new(run, u128::from_le_bytes(header[24..40].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid object id"))?;
        let object_type =
            ObjectTypeId::new(u128::from_le_bytes(header[40..56].try_into().unwrap()))
                .map_err(|_| ActionError::codec("invalid object type"))?;
        let epoch = u64::from_le_bytes(header[56..64].try_into().unwrap());
        if epoch == 0 {
            return Err(ActionError::codec("invalid object epoch"));
        }
        Ok(Self {
            object,
            object_type,
            epoch,
            lease_locality: LocalityId::new(u64::from_le_bytes(header[64..72].try_into().unwrap())),
            domain: DomainId::new(u32::from_le_bytes(header[72..76].try_into().unwrap())),
            rooted,
        })
    }
}

mod migration;
pub use migration::MigrationFuture;
use migration::MIGRATION_SCHEMA_ID;
pub(crate) use migration::{
    MigrationPacket, MIGRATION_ACTIVATE_ACTION_ID, MIGRATION_COMMIT_ACTION_ID,
    MIGRATION_FREEZE_ACTION_ID, MIGRATION_HEADER_BYTES, MIGRATION_PREPARE_ACTION_ID,
    MIGRATION_RETIRE_ACTION_ID, MIGRATION_ROLLBACK_ACTION_ID,
};

type Decoder = dyn Fn(Segments) -> Result<ErasedState, ActionError> + Send + Sync;
type LocalWrapper = dyn Fn(Box<dyn Any + Send>) -> Result<ErasedState, RuntimeError> + Send + Sync;
type ObjectHandler = dyn Fn(&ObjectEntry, Segments) -> Result<Segments, ActionError> + Send + Sync;
type Freezer = dyn Fn(&ErasedState) -> Result<Segments, ActionError> + Send + Sync;
type Restorer = dyn Fn(Segments, RestoreContext) -> Result<ErasedState, ActionError> + Send + Sync;

#[derive(Clone)]
struct ObjectActionRegistration {
    object_type: ObjectTypeId,
    input_schema: u64,
    output_schema: u64,
    access: ObjectAccess,
    handler: Arc<ObjectHandler>,
}

#[derive(Clone)]
struct MigrationRegistration {
    mobility: Mobility,
    snapshot_version: u32,
    snapshot_schema: u64,
    freezer: Option<Arc<Freezer>>,
    restorer: Option<Arc<Restorer>>,
}

#[derive(Clone)]
struct PlacementActionRegistration {
    input_schema: u64,
    output_schema: u64,
    handler: RegisteredAction,
}

#[derive(Clone)]
struct ObjectTypeRegistration {
    schema: u64,
    concurrency: ObjectConcurrency,
    mobility: Mobility,
    snapshot_version: u32,
    snapshot_schema: u64,
    decoder: Arc<Decoder>,
    local_wrapper: Arc<LocalWrapper>,
    freezer: Option<Arc<Freezer>>,
    restorer: Option<Arc<Restorer>>,
}

#[derive(Clone, Default)]
pub(crate) struct ObjectRegistry {
    types: BTreeMap<ObjectTypeId, ObjectTypeRegistration>,
    actions: BTreeMap<ActionId, ObjectActionRegistration>,
    placements: BTreeMap<ActionId, PlacementActionRegistration>,
}

impl std::fmt::Debug for ObjectRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectRegistry")
            .field("types", &self.types.len())
            .field("actions", &self.actions.len())
            .field("placements", &self.placements.len())
            .finish()
    }
}

impl ObjectRegistry {
    pub(crate) fn register_exclusive<T: DistributedObject>(&mut self) -> Result<(), RuntimeError> {
        self.register_type::<T>(
            ObjectConcurrency::Exclusive,
            MigrationRegistration {
                mobility: Mobility::Pinned,
                snapshot_version: 0,
                snapshot_schema: 0,
                freezer: None,
                restorer: None,
            },
            |segments| {
                Ok(ErasedState::Exclusive(Mutex::new(Box::new(T::decode(
                    segments,
                )?))))
            },
            |state| {
                let state = state.downcast::<T>().map_err(|_| {
                    RuntimeError::UnknownObjectType(ObjectTypeId::new(T::TYPE_ID).unwrap())
                })?;
                Ok(ErasedState::Exclusive(Mutex::new(state)))
            },
        )
    }

    pub(crate) fn register_read_write<T: DistributedObject + Sync>(
        &mut self,
    ) -> Result<(), RuntimeError> {
        self.register_type::<T>(
            ObjectConcurrency::ReadWrite,
            MigrationRegistration {
                mobility: Mobility::Pinned,
                snapshot_version: 0,
                snapshot_schema: 0,
                freezer: None,
                restorer: None,
            },
            |segments| {
                Ok(ErasedState::ReadWrite(RwLock::new(Box::new(T::decode(
                    segments,
                )?))))
            },
            |state| {
                let state = state.downcast::<T>().map_err(|_| {
                    RuntimeError::UnknownObjectType(ObjectTypeId::new(T::TYPE_ID).unwrap())
                })?;
                let state: Box<dyn Any + Send + Sync> = state;
                Ok(ErasedState::ReadWrite(RwLock::new(state)))
            },
        )
    }

    pub(crate) fn register_mobile_exclusive<T: MobileObject>(
        &mut self,
    ) -> Result<(), RuntimeError> {
        validate_mobile::<T>()?;
        self.register_type::<T>(
            ObjectConcurrency::Exclusive,
            MigrationRegistration {
                mobility: T::MOBILITY,
                snapshot_version: T::SNAPSHOT_VERSION,
                snapshot_schema: T::Snapshot::SCHEMA_ID,
                freezer: Some(Arc::new(|state| {
                    let ErasedState::Exclusive(state) = state else {
                        return Err(ActionError::user("object concurrency mismatch"));
                    };
                    let mut state = state
                        .try_lock()
                        .map_err(|_| ActionError::user("object admission saturated"))?;
                    let typed = state
                        .downcast_mut::<T>()
                        .ok_or_else(|| ActionError::user("object type mismatch"))?;
                    typed.freeze()?.encode()
                })),
                restorer: Some(Arc::new(|segments, context| {
                    let state = T::restore(T::Snapshot::decode(segments)?, context)?;
                    Ok(ErasedState::Exclusive(Mutex::new(Box::new(state))))
                })),
            },
            |segments| {
                Ok(ErasedState::Exclusive(Mutex::new(Box::new(T::decode(
                    segments,
                )?))))
            },
            |state| {
                let state = state.downcast::<T>().map_err(|_| {
                    RuntimeError::UnknownObjectType(ObjectTypeId::new(T::TYPE_ID).unwrap())
                })?;
                Ok(ErasedState::Exclusive(Mutex::new(state)))
            },
        )
    }

    pub(crate) fn register_mobile_read_write<T: MobileObject + Sync>(
        &mut self,
    ) -> Result<(), RuntimeError> {
        validate_mobile::<T>()?;
        self.register_type::<T>(
            ObjectConcurrency::ReadWrite,
            MigrationRegistration {
                mobility: T::MOBILITY,
                snapshot_version: T::SNAPSHOT_VERSION,
                snapshot_schema: T::Snapshot::SCHEMA_ID,
                freezer: Some(Arc::new(|state| {
                    let ErasedState::ReadWrite(state) = state else {
                        return Err(ActionError::user("object concurrency mismatch"));
                    };
                    let mut state = state
                        .try_write()
                        .map_err(|_| ActionError::user("object admission saturated"))?;
                    let typed = state
                        .downcast_mut::<T>()
                        .ok_or_else(|| ActionError::user("object type mismatch"))?;
                    typed.freeze()?.encode()
                })),
                restorer: Some(Arc::new(|segments, context| {
                    let state = T::restore(T::Snapshot::decode(segments)?, context)?;
                    let state: Box<dyn Any + Send + Sync> = Box::new(state);
                    Ok(ErasedState::ReadWrite(RwLock::new(state)))
                })),
            },
            |segments| {
                Ok(ErasedState::ReadWrite(RwLock::new(Box::new(T::decode(
                    segments,
                )?))))
            },
            |state| {
                let state = state.downcast::<T>().map_err(|_| {
                    RuntimeError::UnknownObjectType(ObjectTypeId::new(T::TYPE_ID).unwrap())
                })?;
                let state: Box<dyn Any + Send + Sync> = state;
                Ok(ErasedState::ReadWrite(RwLock::new(state)))
            },
        )
    }

    fn register_type<T: DistributedObject>(
        &mut self,
        concurrency: ObjectConcurrency,
        migration: MigrationRegistration,
        decoder: impl Fn(Segments) -> Result<ErasedState, ActionError> + Send + Sync + 'static,
        local_wrapper: impl Fn(Box<dyn Any + Send>) -> Result<ErasedState, RuntimeError>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), RuntimeError> {
        let id = ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        if T::SCHEMA_ID == 0 {
            return Err(RuntimeError::InvalidSchema);
        }
        if self.types.contains_key(&id) {
            return Err(RuntimeError::DuplicateObjectType(id));
        }
        let create = create_action_id(id)?;
        if self.reserves_action(create) {
            return Err(RuntimeError::DuplicateAction(create));
        }
        self.types.insert(
            id,
            ObjectTypeRegistration {
                schema: T::SCHEMA_ID,
                concurrency,
                mobility: migration.mobility,
                snapshot_version: migration.snapshot_version,
                snapshot_schema: migration.snapshot_schema,
                decoder: Arc::new(decoder),
                local_wrapper: Arc::new(local_wrapper),
                freezer: migration.freezer,
                restorer: migration.restorer,
            },
        );
        Ok(())
    }

    pub(crate) fn register_read<T, A>(&mut self) -> Result<(), RuntimeError>
    where
        T: DistributedObject + Sync,
        A: ObjectReadAction<T>,
    {
        let object_type = self.require_type::<T>(ObjectConcurrency::ReadWrite)?;
        self.insert_action(
            A::ID,
            A::SCHEMA_ID,
            A::Output::SCHEMA_ID,
            object_type,
            ObjectAccess::Read,
            |entry, input| {
                let action = A::decode(input)?;
                let erased = entry.resident_state()?;
                let ErasedState::ReadWrite(state) = &*erased else {
                    return Err(ActionError::user("object does not admit reads"));
                };
                let state = state
                    .try_read()
                    .map_err(|_| ActionError::user("object admission saturated"))?;
                let typed = state
                    .downcast_ref::<T>()
                    .ok_or_else(|| ActionError::user("object type mismatch"))?;
                action.execute(typed)?.encode()
            },
        )
    }

    pub(crate) fn register_write<T, A>(&mut self) -> Result<(), RuntimeError>
    where
        T: DistributedObject,
        A: ObjectWriteAction<T>,
    {
        let object_type =
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        if !self.types.contains_key(&object_type) {
            return Err(RuntimeError::UnknownObjectType(object_type));
        }
        self.insert_action(
            A::ID,
            A::SCHEMA_ID,
            A::Output::SCHEMA_ID,
            object_type,
            ObjectAccess::Write,
            |entry, input| {
                let action = A::decode(input)?;
                let erased = entry.resident_state()?;
                match &*erased {
                    ErasedState::Exclusive(state) => {
                        let mut state = state
                            .try_lock()
                            .map_err(|_| ActionError::user("object admission saturated"))?;
                        let typed = state
                            .downcast_mut::<T>()
                            .ok_or_else(|| ActionError::user("object type mismatch"))?;
                        action.execute(typed)?.encode()
                    }
                    ErasedState::ReadWrite(state) => {
                        let mut state = state
                            .try_write()
                            .map_err(|_| ActionError::user("object admission saturated"))?;
                        let typed = state
                            .downcast_mut::<T>()
                            .ok_or_else(|| ActionError::user("object type mismatch"))?;
                        action.execute(typed)?.encode()
                    }
                }
            },
        )
    }

    fn require_type<T: DistributedObject>(
        &self,
        concurrency: ObjectConcurrency,
    ) -> Result<ObjectTypeId, RuntimeError> {
        let id = ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        let registration = self
            .types
            .get(&id)
            .ok_or(RuntimeError::UnknownObjectType(id))?;
        if registration.concurrency != concurrency {
            return Err(RuntimeError::InvalidObjectConcurrency);
        }
        Ok(id)
    }

    fn insert_action(
        &mut self,
        raw_id: u128,
        input_schema: u64,
        output_schema: u64,
        object_type: ObjectTypeId,
        access: ObjectAccess,
        handler: impl Fn(&ObjectEntry, Segments) -> Result<Segments, ActionError>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), RuntimeError> {
        let id = ActionId::new(raw_id).map_err(|_| RuntimeError::InvalidActionId)?;
        if input_schema == 0 || output_schema == 0 || self.reserves_action(id) {
            return Err(RuntimeError::DuplicateAction(id));
        }
        self.actions.insert(
            id,
            ObjectActionRegistration {
                object_type,
                input_schema,
                output_schema,
                access,
                handler: Arc::new(handler),
            },
        );
        Ok(())
    }

    pub(crate) fn register_placement<A: Action>(
        &mut self,
        handler: RegisteredAction,
    ) -> Result<ActionId, RuntimeError> {
        let id =
            placement_action_id(ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?)?;
        if A::SCHEMA_ID == 0 || A::Output::SCHEMA_ID == 0 || self.reserves_action(id) {
            return Err(RuntimeError::DuplicateAction(id));
        }
        self.placements.insert(
            id,
            PlacementActionRegistration {
                input_schema: A::SCHEMA_ID,
                output_schema: A::Output::SCHEMA_ID,
                handler,
            },
        );
        Ok(id)
    }

    pub(crate) fn reserves_action(&self, id: ActionId) -> bool {
        matches!(
            id.get(),
            LEASE_RENEW_ACTION_ID
                | LEASE_RELEASE_ACTION_ID
                | LEASE_ACQUIRE_ACTION_ID
                | ROOT_RELEASE_ACTION_ID
                | TRANSFER_ACK_ACTION_ID
                | MIGRATION_FREEZE_ACTION_ID
                | MIGRATION_PREPARE_ACTION_ID
                | MIGRATION_COMMIT_ACTION_ID
                | MIGRATION_ACTIVATE_ACTION_ID
                | MIGRATION_ROLLBACK_ACTION_ID
                | MIGRATION_RETIRE_ACTION_ID
        ) || self.actions.contains_key(&id)
            || self.placements.contains_key(&id)
            || self
                .types
                .keys()
                .any(|object_type| create_action_id(*object_type).is_ok_and(|create| create == id))
    }

    pub(crate) fn install(
        &self,
        service: &Arc<ObjectService>,
        actions: &mut ActionRegistry,
    ) -> Result<(), RuntimeError> {
        actions.insert_erased(
            ActionId::new(LEASE_RENEW_ACTION_ID).unwrap(),
            1,
            1,
            service.lease_handler(true),
        )?;
        actions.insert_erased(
            ActionId::new(LEASE_RELEASE_ACTION_ID).unwrap(),
            1,
            1,
            service.lease_handler(false),
        )?;
        actions.insert_erased(
            ActionId::new(LEASE_ACQUIRE_ACTION_ID).unwrap(),
            1,
            1,
            service.acquire_lease_handler(),
        )?;
        actions.insert_erased(
            ActionId::new(ROOT_RELEASE_ACTION_ID).unwrap(),
            1,
            1,
            service.root_release_handler(),
        )?;
        actions.insert_erased(
            ActionId::new(TRANSFER_ACK_ACTION_ID).unwrap(),
            1,
            1,
            service.transfer_ack_handler(),
        )?;
        for (id, handler) in [
            (
                MIGRATION_FREEZE_ACTION_ID,
                service.migration_freeze_handler(),
            ),
            (
                MIGRATION_PREPARE_ACTION_ID,
                service.migration_prepare_handler(),
            ),
            (
                MIGRATION_COMMIT_ACTION_ID,
                service.migration_commit_handler(),
            ),
            (
                MIGRATION_ACTIVATE_ACTION_ID,
                service.migration_activate_handler(),
            ),
            (
                MIGRATION_ROLLBACK_ACTION_ID,
                service.migration_rollback_handler(),
            ),
            (
                MIGRATION_RETIRE_ACTION_ID,
                service.migration_retire_handler(),
            ),
        ] {
            actions.insert_erased(
                ActionId::new(id).unwrap(),
                MIGRATION_SCHEMA_ID,
                MIGRATION_SCHEMA_ID,
                handler,
            )?;
        }
        for (object_type, registration) in &self.types {
            actions.insert_erased(
                create_action_id(*object_type)?,
                registration.schema,
                LOCATION_SCHEMA_ID,
                service.create_handler(*object_type)?,
            )?;
        }
        for (id, registration) in &self.placements {
            let handler = registration.handler.clone();
            actions.insert_erased(
                *id,
                registration.input_schema,
                registration.output_schema,
                RegisteredAction::new(move |segments| {
                    let (_, payload) = ObjectEnvelope::decode(segments)?;
                    handler.execute_encoded(payload)
                }),
            )?;
        }
        for (id, registration) in &self.actions {
            actions.insert_erased(
                *id,
                registration.input_schema,
                registration.output_schema,
                service.action_handler(*id)?,
            )?;
        }
        Ok(())
    }

    pub(crate) fn fingerprint(&self) -> [u8; 32] {
        let mut hashes = [
            0xcbf2_9ce4_8422_2325_u64,
            0x8422_2325_cbf2_9ce4,
            0x9e37_79b9_7f4a_7c15,
            0xd6e8_feb8_6659_fd93,
        ];
        for (id, registration) in &self.types {
            hash_bytes(&mut hashes, &id.get().to_le_bytes());
            hash_bytes(&mut hashes, &registration.schema.to_le_bytes());
            hash_bytes(&mut hashes, &[registration.concurrency as u8]);
            hash_bytes(&mut hashes, &[registration.mobility as u8]);
            hash_bytes(&mut hashes, &registration.snapshot_version.to_le_bytes());
            hash_bytes(&mut hashes, &registration.snapshot_schema.to_le_bytes());
        }
        for (id, registration) in &self.placements {
            hash_bytes(&mut hashes, &id.get().to_le_bytes());
            hash_bytes(&mut hashes, &registration.input_schema.to_le_bytes());
            hash_bytes(&mut hashes, &registration.output_schema.to_le_bytes());
        }
        for (id, registration) in &self.actions {
            hash_bytes(&mut hashes, &id.get().to_le_bytes());
            hash_bytes(&mut hashes, &registration.object_type.get().to_le_bytes());
            hash_bytes(&mut hashes, &registration.input_schema.to_le_bytes());
            hash_bytes(&mut hashes, &registration.output_schema.to_le_bytes());
            hash_bytes(&mut hashes, &[registration.access as u8]);
        }
        let mut result = [0; 32];
        for (index, hash) in hashes.into_iter().enumerate() {
            result[index * 8..index * 8 + 8].copy_from_slice(&hash.to_le_bytes());
        }
        result
    }
}

fn validate_mobile<T: MobileObject>() -> Result<(), RuntimeError> {
    if T::MOBILITY == Mobility::Pinned {
        return Err(RuntimeError::PinnedObjectType(
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?,
        ));
    }
    if T::SNAPSHOT_VERSION == 0 || T::Snapshot::SCHEMA_ID == 0 {
        return Err(RuntimeError::InvalidSchema);
    }
    Ok(())
}

const PLACEMENT_ACTION_MASK: u128 = 0xc4ea_7e03_0000_0000_0000_0000_0000_0000;

pub(crate) fn placement_action_id(action: ActionId) -> Result<ActionId, RuntimeError> {
    ActionId::new(action.get() ^ PLACEMENT_ACTION_MASK).map_err(|_| RuntimeError::InvalidActionId)
}

const CREATE_ACTION_MASK: u128 = 0xc4ea_7e00_0000_0000_0000_0000_0000_0000;
const LOCATION_SCHEMA_ID: u64 = 3;
pub(crate) const LEASE_RENEW_ACTION_ID: u128 = 0xc4ea_7e01_0000_0000_0000_0000_0000_0001;
pub(crate) const LEASE_RELEASE_ACTION_ID: u128 = 0xc4ea_7e01_0000_0000_0000_0000_0000_0002;
pub(crate) const LEASE_ACQUIRE_ACTION_ID: u128 = 0xc4ea_7e01_0000_0000_0000_0000_0000_0003;
pub(crate) const ROOT_RELEASE_ACTION_ID: u128 = 0xc4ea_7e01_0000_0000_0000_0000_0000_0004;
pub(crate) const TRANSFER_ACK_ACTION_ID: u128 = 0xc4ea_7e01_0000_0000_0000_0000_0000_0005;

pub(crate) fn create_action_id(object_type: ObjectTypeId) -> Result<ActionId, RuntimeError> {
    ActionId::new(object_type.get() ^ CREATE_ACTION_MASK).map_err(|_| RuntimeError::InvalidActionId)
}

fn hash_bytes(hashes: &mut [u64; 4], bytes: &[u8]) {
    for byte in bytes {
        for (index, hash) in hashes.iter_mut().enumerate() {
            *hash ^= u64::from(*byte).wrapping_add(index as u64);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
}

enum ErasedState {
    Exclusive(Mutex<Box<dyn Any + Send>>),
    ReadWrite(RwLock<Box<dyn Any + Send + Sync>>),
}

struct LocalAdmissionPin<'a> {
    service: &'a ObjectService,
    job: ObjectJob,
}

impl Drop for LocalAdmissionPin<'_> {
    fn drop(&mut self) {
        self.service.complete(self.job);
        self.service.collect(self.job.object, Instant::now());
    }
}

struct LeaseRecord {
    expires_at: Instant,
    suspect_since: Option<Instant>,
}

struct Admission {
    readers: usize,
    writer: bool,
    queue: VecDeque<ActionJob>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MigrationOutcome {
    Completed,
    RolledBack,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObjectLifecycle {
    Resident,
    Freezing {
        migration: RequestId,
        expires_at: Instant,
    },
    Prepared {
        migration: RequestId,
        expires_at: Instant,
    },
    Forwarding {
        location: ObjectLocation,
        expires_at: Instant,
    },
    Retiring {
        location: ObjectLocation,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Retirement {
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) authority: LocalityId,
    pub(crate) location: ObjectLocation,
    pub(crate) snapshot_version: u32,
    pub(crate) snapshot_schema: u64,
}

struct PreparedState {
    migration: RequestId,
    location: ObjectLocation,
    state: Arc<ErasedState>,
    snapshot_bytes: usize,
    expires_at: Instant,
}

struct ObjectEntry {
    object_type: ObjectTypeId,
    authority: LocalityId,
    location: Mutex<ObjectLocation>,
    state: Mutex<Option<Arc<ErasedState>>>,
    lifecycle: Mutex<ObjectLifecycle>,
    prepared: Mutex<Option<PreparedState>>,
    snapshot_bytes: Mutex<usize>,
    leases: Mutex<HashMap<LocalityId, LeaseRecord>>,
    roots: Mutex<usize>,
    in_flight: Mutex<usize>,
    placement_tickets: Mutex<usize>,
    admission: Mutex<Admission>,
}

impl ObjectEntry {
    fn location(&self) -> ObjectLocation {
        *self.location.lock().unwrap()
    }

    fn resident_state(&self) -> Result<Arc<ErasedState>, ActionError> {
        self.state
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| ActionError::user("object moved"))
    }
}

pub(crate) struct ObjectService {
    run_id: RunId,
    local_id: LocalityId,
    pub(crate) limits: ObjectLimits,
    registry: ObjectRegistry,
    entries: Mutex<HashMap<ObjectId, Arc<ObjectEntry>>>,
    next_slot: Mutex<u64>,
    resolver: Mutex<(HashMap<ObjectId, ObjectLocation>, VecDeque<ObjectId>)>,
    local_calls: AtomicU64,
    remote_calls: AtomicU64,
    root_count: AtomicU64,
    transfer_count: AtomicU64,
    placement_count: AtomicU64,
    pending_transfers: Mutex<HashSet<(ObjectId, LocalityId)>>,
    active_migrations: Mutex<HashSet<RequestId>>,
    prepared_count: AtomicU64,
    completed_migrations: AtomicU64,
    rolled_back_migrations: AtomicU64,
    failed_migrations: AtomicU64,
    migrated_bytes: AtomicU64,
    retirements: Mutex<VecDeque<Retirement>>,
}

impl std::fmt::Debug for ObjectService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats(Instant::now());
        f.debug_struct("ObjectService")
            .field("local_id", &self.local_id)
            .field("stats", &stats)
            .finish()
    }
}

impl ObjectService {
    pub(crate) fn new(
        run_id: RunId,
        local_id: LocalityId,
        limits: ObjectLimits,
        registry: ObjectRegistry,
    ) -> Self {
        Self {
            run_id,
            local_id,
            limits,
            registry,
            entries: Mutex::new(HashMap::new()),
            next_slot: Mutex::new(1),
            resolver: Mutex::new((HashMap::new(), VecDeque::new())),
            local_calls: AtomicU64::new(0),
            remote_calls: AtomicU64::new(0),
            root_count: AtomicU64::new(0),
            transfer_count: AtomicU64::new(0),
            placement_count: AtomicU64::new(0),
            pending_transfers: Mutex::new(HashSet::new()),
            active_migrations: Mutex::new(HashSet::new()),
            prepared_count: AtomicU64::new(0),
            completed_migrations: AtomicU64::new(0),
            rolled_back_migrations: AtomicU64::new(0),
            failed_migrations: AtomicU64::new(0),
            migrated_bytes: AtomicU64::new(0),
            retirements: Mutex::new(VecDeque::new()),
        }
    }

    pub(crate) fn create_local<T: DistributedObject>(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        domain: DomainId,
        state: T,
        lease_locality: LocalityId,
        rooted: bool,
    ) -> Result<ObjectLocation, RuntimeError> {
        let registration = self
            .registry
            .types
            .get(&object_type)
            .ok_or(RuntimeError::UnknownObjectType(object_type))?;
        let state = (registration.local_wrapper)(Box::new(state))?;
        self.insert(
            ObjectEnvelope {
                object,
                object_type,
                epoch: 1,
                lease_locality,
                domain,
                rooted,
            },
            state,
            Instant::now(),
        )
        .map_err(|error| RuntimeError::ObjectAction(error.message()))
    }

    pub(crate) fn create_handler(
        self: &Arc<Self>,
        object_type: ObjectTypeId,
    ) -> Result<RegisteredAction, RuntimeError> {
        let registration = self
            .registry
            .types
            .get(&object_type)
            .cloned()
            .ok_or(RuntimeError::UnknownObjectType(object_type))?;
        let service = Arc::clone(self);
        Ok(RegisteredAction::new(move |segments| {
            let (envelope, state) = ObjectEnvelope::decode(segments)?;
            if envelope.object_type != object_type || envelope.object.run() != service.run_id {
                return Err(ActionError::user("object identity mismatch"));
            }
            let state = (registration.decoder)(state)?;
            let location = service.insert(envelope, state, Instant::now())?;
            encode_location(location)
        }))
    }

    pub(crate) fn acquire_lease_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            let (envelope, payload) = ObjectEnvelope::decode(segments)?;
            let acquired = if envelope.rooted {
                service.begin_transfer(envelope.object, envelope.lease_locality, Instant::now())
            } else {
                service.acquire(envelope.object, envelope.lease_locality, Instant::now())
            };
            if !payload.is_empty() || !acquired {
                return Err(ActionError::user("object lease is unavailable"));
            }
            Ok(Vec::new())
        })
    }

    pub(crate) fn transfer_ack_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            let (envelope, payload) = ObjectEnvelope::decode(segments)?;
            if !payload.is_empty() {
                return Err(ActionError::codec(
                    "transfer acknowledgement payload must be empty",
                ));
            }
            service.ack_transfer(envelope.object, envelope.lease_locality);
            Ok(Vec::new())
        })
    }

    pub(crate) fn root_release_handler(self: &Arc<Self>) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            let (envelope, payload) = ObjectEnvelope::decode(segments)?;
            if !payload.is_empty() {
                return Err(ActionError::codec("root control payload must be empty"));
            }
            service.release_root(envelope.object);
            Ok(Vec::new())
        })
    }

    pub(crate) fn lease_handler(self: &Arc<Self>, renew: bool) -> RegisteredAction {
        let service = Arc::clone(self);
        RegisteredAction::new(move |segments| {
            let (envelope, payload) = ObjectEnvelope::decode(segments)?;
            if !payload.is_empty() {
                return Err(ActionError::codec("lease control payload must be empty"));
            }
            if renew {
                if !service.renew(envelope.object, envelope.lease_locality, Instant::now()) {
                    return Err(ActionError::user("object lease is unavailable"));
                }
            } else {
                service.release(envelope.object, envelope.lease_locality);
            }
            Ok(Vec::new())
        })
    }

    pub(crate) fn is_local_resident(&self, object: ObjectId, location: ObjectLocation) -> bool {
        self.entries
            .lock()
            .unwrap()
            .get(&object)
            .is_some_and(|entry| {
                let lifecycle = entry.lifecycle.lock().unwrap();
                matches!(*lifecycle, ObjectLifecycle::Resident)
                    && entry.location() == location
                    && entry.state.lock().unwrap().is_some()
            })
    }

    pub(crate) fn request_targets_current(&self, action: ActionId, payload: &Segments) -> bool {
        if !self.registry.actions.contains_key(&action)
            && !self.registry.placements.contains_key(&action)
        {
            return false;
        }
        let Ok(envelope) = ObjectEnvelope::inspect(payload) else {
            return false;
        };
        self.entries
            .lock()
            .unwrap()
            .get(&envelope.object)
            .is_some_and(|entry| {
                entry.location().epoch() == envelope.epoch
                    && matches!(*entry.lifecycle.lock().unwrap(), ObjectLifecycle::Resident)
            })
    }

    // `fetch_update` is deprecated on newer toolchains in favor of `try_update`,
    // which does not exist at the crate's MSRV (1.85).
    #[allow(deprecated)]
    pub(crate) fn admit(
        &self,
        action: ActionId,
        mut job: ActionJob,
        domain_available: bool,
    ) -> Result<Option<ActionJob>, RuntimeError> {
        let registration = self.registry.actions.get(&action);
        let placement = self.registry.placements.contains_key(&action);
        let freeze = action.get() == MIGRATION_FREEZE_ACTION_ID;
        if registration.is_none() && !placement && !freeze {
            return Ok(Some(job));
        }
        let encoded = match &job.input {
            ActionValue::Encoded(segments) => segments,
            ActionValue::Typed(_) => {
                return Err(RuntimeError::Protocol(
                    "object admission requires encoded input".into(),
                ));
            }
        };
        let (object, object_type, epoch, migration) = if freeze {
            let packet = MigrationPacket::decode(encoded.clone())
                .map_err(|error| RuntimeError::Protocol(error.message()))?;
            (
                packet.object,
                packet.object_type,
                packet.from.epoch(),
                Some(packet.migration),
            )
        } else {
            let envelope = ObjectEnvelope::inspect(encoded)
                .map_err(|error| RuntimeError::Protocol(error.message()))?;
            (envelope.object, envelope.object_type, envelope.epoch, None)
        };
        let entry = self
            .entries
            .lock()
            .unwrap()
            .get(&object)
            .cloned()
            .ok_or(RuntimeError::ObjectCollected(object))?;
        if entry.object_type != object_type {
            return Err(RuntimeError::UnknownObjectType(object_type));
        }
        let location = entry.location();
        if location.epoch() != epoch {
            return Err(RuntimeError::Moved { object, location });
        }
        {
            let mut lifecycle = entry.lifecycle.lock().unwrap();
            if let Some(migration) = migration {
                match *lifecycle {
                    ObjectLifecycle::Resident => {
                        *lifecycle = ObjectLifecycle::Freezing {
                            migration,
                            expires_at: Instant::now() + self.limits.forwarding_ttl,
                        };
                    }
                    ObjectLifecycle::Freezing {
                        migration: existing,
                        ..
                    } if existing == migration => {}
                    _ => return Err(RuntimeError::MigrationInProgress(object)),
                }
            } else {
                match *lifecycle {
                    ObjectLifecycle::Resident => {}
                    ObjectLifecycle::Forwarding { location, .. }
                    | ObjectLifecycle::Retiring { location } => {
                        return Err(RuntimeError::Moved { object, location });
                    }
                    ObjectLifecycle::Freezing { .. } | ObjectLifecycle::Prepared { .. } => {
                        return Err(RuntimeError::MigrationInProgress(object));
                    }
                }
            }
        }
        if placement {
            if !domain_available {
                return Err(RuntimeError::Placement {
                    request: job.request,
                    message: "colocated domain is saturated".into(),
                });
            }
            let lifecycle = entry.lifecycle.lock().unwrap();
            if !matches!(*lifecycle, ObjectLifecycle::Resident) {
                return Err(RuntimeError::MigrationInProgress(object));
            }
            self.placement_count
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                    (count < self.limits.max_placement_tickets as u64).then_some(count + 1)
                })
                .map_err(|_| RuntimeError::ResourceExhausted {
                    resource: crate::ResourceKind::PlacementTickets,
                    limit: self.limits.max_placement_tickets,
                })?;
            *entry.placement_tickets.lock().unwrap() += 1;
            *entry.in_flight.lock().unwrap() += 1;
            drop(lifecycle);
            job.object = Some(ObjectJob {
                object,
                read: false,
                placement: true,
            });
            return Ok(Some(job));
        }
        let read =
            registration.is_some_and(|registration| registration.access == ObjectAccess::Read);
        let mut admission = entry.admission.lock().unwrap();
        let available = domain_available
            && admission.queue.is_empty()
            && (!freeze || *entry.placement_tickets.lock().unwrap() == 0)
            && if read {
                !admission.writer
            } else {
                !admission.writer && admission.readers == 0
            };
        job.object = Some(ObjectJob {
            object,
            read,
            placement: false,
        });
        if available {
            if read {
                admission.readers += 1;
            } else {
                admission.writer = true;
            }
            *entry.in_flight.lock().unwrap() += 1;
            return Ok(Some(job));
        }
        if admission.queue.len() >= self.limits.max_mailbox_per_object {
            return Err(RuntimeError::ResourceExhausted {
                resource: crate::ResourceKind::ObjectMailbox,
                limit: self.limits.max_mailbox_per_object,
            });
        }
        admission.queue.push_back(job);
        Ok(None)
    }

    pub(crate) fn complete(&self, completed: ObjectJob) -> Vec<ActionJob> {
        let Some(entry) = self.entries.lock().unwrap().get(&completed.object).cloned() else {
            return Vec::new();
        };
        let mut admission = entry.admission.lock().unwrap();
        if completed.placement {
            let mut tickets = entry.placement_tickets.lock().unwrap();
            *tickets = tickets.saturating_sub(1);
            self.placement_count.fetch_sub(1, Ordering::AcqRel);
        } else if completed.read {
            admission.readers = admission.readers.saturating_sub(1);
        } else {
            admission.writer = false;
        }
        let mut in_flight = entry.in_flight.lock().unwrap();
        *in_flight = in_flight.saturating_sub(1);
        Vec::new()
    }

    pub(crate) fn rollback(&self, job: ObjectJob) -> Vec<ActionJob> {
        self.complete(job)
    }

    pub(crate) fn retry_ready(
        &self,
        max: usize,
        mut domain_available: impl FnMut(DomainId) -> bool,
    ) -> Vec<ActionJob> {
        let entries: Vec<_> = self.entries.lock().unwrap().values().cloned().collect();
        let mut ready = Vec::new();
        for entry in entries {
            if ready.len() >= max {
                break;
            }
            let mut admission = entry.admission.lock().unwrap();
            if admission.writer || admission.readers != 0 {
                continue;
            }
            loop {
                if ready.len() >= max {
                    break;
                }
                let Some(front) = admission.queue.front() else {
                    break;
                };
                if !domain_available(front.domain)
                    || (front.action_id.get() == MIGRATION_FREEZE_ACTION_ID
                        && *entry.placement_tickets.lock().unwrap() != 0)
                {
                    break;
                }
                let object = front
                    .object
                    .expect("queued object job has admission metadata");
                if object.read {
                    admission.readers += 1;
                } else {
                    admission.writer = true;
                }
                *entry.in_flight.lock().unwrap() += 1;
                ready.push(admission.queue.pop_front().unwrap());
                if !object.read {
                    break;
                }
            }
        }
        ready
    }

    fn admit_local<'a>(
        &'a self,
        entry: &ObjectEntry,
        object: ObjectId,
        read: bool,
    ) -> Result<LocalAdmissionPin<'a>, RuntimeError> {
        let lifecycle = entry.lifecycle.lock().unwrap();
        match *lifecycle {
            ObjectLifecycle::Resident => {}
            ObjectLifecycle::Forwarding { location, .. }
            | ObjectLifecycle::Retiring { location } => {
                return Err(RuntimeError::Moved { object, location });
            }
            ObjectLifecycle::Freezing { .. } | ObjectLifecycle::Prepared { .. } => {
                return Err(RuntimeError::MigrationInProgress(object));
            }
        }
        let mut admission = entry.admission.lock().unwrap();
        drop(lifecycle);
        let available = admission.queue.is_empty()
            && if read {
                !admission.writer
            } else {
                !admission.writer && admission.readers == 0
            };
        if !available {
            return Err(RuntimeError::ResourceExhausted {
                resource: crate::ResourceKind::ObjectMailbox,
                limit: self.limits.max_mailbox_per_object,
            });
        }
        if read {
            admission.readers += 1;
        } else {
            admission.writer = true;
        }
        *entry.in_flight.lock().unwrap() += 1;
        Ok(LocalAdmissionPin {
            service: self,
            job: ObjectJob {
                object,
                read,
                placement: false,
            },
        })
    }

    pub(crate) fn call_read_local<T, A>(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
        action: A,
    ) -> Result<A::Output, RuntimeError>
    where
        T: DistributedObject + Sync,
        A: ObjectReadAction<T>,
    {
        let entry = self.local_entry(object, object_type, location)?;
        let _pin = self.admit_local(&entry, object, true)?;
        let erased = entry
            .resident_state()
            .map_err(|error| RuntimeError::ObjectAction(error.message()))?;
        let ErasedState::ReadWrite(state) = &*erased else {
            return Err(RuntimeError::InvalidObjectConcurrency);
        };
        let state = state
            .try_read()
            .map_err(|_| RuntimeError::ResourceExhausted {
                resource: crate::ResourceKind::ObjectMailbox,
                limit: self.limits.max_mailbox_per_object,
            })?;
        let typed = state
            .downcast_ref::<T>()
            .ok_or(RuntimeError::UnknownObjectType(object_type))?;
        self.local_calls.fetch_add(1, Ordering::Relaxed);
        match catch_unwind(AssertUnwindSafe(|| action.execute(typed))) {
            Ok(result) => result.map_err(|error| RuntimeError::ObjectAction(error.message())),
            Err(_) => Err(RuntimeError::ObjectAction("object action panicked".into())),
        }
    }

    pub(crate) fn call_write_local<T, A>(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
        action: A,
    ) -> Result<A::Output, RuntimeError>
    where
        T: DistributedObject,
        A: ObjectWriteAction<T>,
    {
        let entry = self.local_entry(object, object_type, location)?;
        let _pin = self.admit_local(&entry, object, false)?;
        let erased = entry
            .resident_state()
            .map_err(|error| RuntimeError::ObjectAction(error.message()))?;
        let output = match &*erased {
            ErasedState::Exclusive(state) => {
                let mut state = state
                    .try_lock()
                    .map_err(|_| RuntimeError::ResourceExhausted {
                        resource: crate::ResourceKind::ObjectMailbox,
                        limit: self.limits.max_mailbox_per_object,
                    })?;
                let typed = state
                    .downcast_mut::<T>()
                    .ok_or(RuntimeError::UnknownObjectType(object_type))?;
                catch_unwind(AssertUnwindSafe(|| action.execute(typed)))
            }
            ErasedState::ReadWrite(state) => {
                let mut state = state
                    .try_write()
                    .map_err(|_| RuntimeError::ResourceExhausted {
                        resource: crate::ResourceKind::ObjectMailbox,
                        limit: self.limits.max_mailbox_per_object,
                    })?;
                let typed = state
                    .downcast_mut::<T>()
                    .ok_or(RuntimeError::UnknownObjectType(object_type))?;
                catch_unwind(AssertUnwindSafe(|| action.execute(typed)))
            }
        };
        self.local_calls.fetch_add(1, Ordering::Relaxed);
        match output {
            Ok(result) => result.map_err(|error| RuntimeError::ObjectAction(error.message())),
            Err(_) => Err(RuntimeError::ObjectAction("object action panicked".into())),
        }
    }

    fn local_entry(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
    ) -> Result<Arc<ObjectEntry>, RuntimeError> {
        let entry = self
            .entries
            .lock()
            .unwrap()
            .get(&object)
            .cloned()
            .ok_or(RuntimeError::ObjectCollected(object))?;
        if entry.object_type != object_type {
            return Err(RuntimeError::UnknownObjectType(object_type));
        }
        if entry.location().epoch() != location.epoch() {
            return Err(RuntimeError::StaleObjectLocation(object));
        }
        Ok(entry)
    }

    pub(crate) fn action_handler(
        self: &Arc<Self>,
        action: ActionId,
    ) -> Result<RegisteredAction, RuntimeError> {
        let registration = self
            .registry
            .actions
            .get(&action)
            .cloned()
            .ok_or(RuntimeError::UnknownAction(action))?;
        let service = Arc::clone(self);
        Ok(RegisteredAction::new(move |segments| {
            let (envelope, input) = ObjectEnvelope::decode(segments)?;
            if envelope.object_type != registration.object_type {
                return Err(ActionError::user("object type mismatch"));
            }
            service.remote_calls.fetch_add(1, Ordering::Relaxed);
            let entry = service
                .entries
                .lock()
                .unwrap()
                .get(&envelope.object)
                .cloned()
                .ok_or_else(|| ActionError::user("object collected"))?;
            if entry.object_type != envelope.object_type
                || entry.location().epoch() != envelope.epoch
            {
                return Err(ActionError::user("stale object location"));
            }
            (registration.handler)(&entry, input)
        }))
    }

    // `fetch_update` is deprecated on newer toolchains in favor of `try_update`,
    // which does not exist at the crate's MSRV (1.85).
    #[allow(deprecated)]
    fn insert(
        &self,
        envelope: ObjectEnvelope,
        state: ErasedState,
        now: Instant,
    ) -> Result<ObjectLocation, ActionError> {
        let object = envelope.object;
        let object_type = envelope.object_type;
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.limits.max_objects {
            return Err(ActionError::user("object store is full"));
        }
        if entries.contains_key(&object) {
            return Err(ActionError::user("duplicate object id"));
        }
        let mut slot = self.next_slot.lock().unwrap();
        let current = *slot;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| ActionError::user("object slot overflow"))?;
        let location = ObjectLocation::new(self.local_id, envelope.domain, current, 1, 1)
            .map_err(|_| ActionError::user("invalid object location"))?;
        if envelope.rooted
            && self
                .root_count
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                    (count < self.limits.max_roots as u64).then_some(count + 1)
                })
                .is_err()
        {
            return Err(ActionError::user("object root limit reached"));
        }
        let mut leases = HashMap::new();
        leases.insert(
            envelope.lease_locality,
            LeaseRecord {
                expires_at: now + self.limits.lease_ttl,
                suspect_since: None,
            },
        );
        entries.insert(
            object,
            Arc::new(ObjectEntry {
                object_type,
                authority: self.local_id,
                location: Mutex::new(location),
                state: Mutex::new(Some(Arc::new(state))),
                lifecycle: Mutex::new(ObjectLifecycle::Resident),
                prepared: Mutex::new(None),
                snapshot_bytes: Mutex::new(0),
                leases: Mutex::new(leases),
                roots: Mutex::new(usize::from(envelope.rooted)),
                in_flight: Mutex::new(0),
                placement_tickets: Mutex::new(0),
                admission: Mutex::new(Admission {
                    readers: 0,
                    writer: false,
                    queue: VecDeque::new(),
                }),
            }),
        );
        drop(entries);
        self.cache_location(object, location);
        Ok(location)
    }

    pub(crate) fn cache_location(&self, object: ObjectId, location: ObjectLocation) {
        let mut resolver = self.resolver.lock().unwrap();
        if !resolver.0.contains_key(&object) {
            if resolver.0.len() >= self.limits.max_resolver_entries {
                if let Some(oldest) = resolver.1.pop_front() {
                    resolver.0.remove(&oldest);
                }
            }
            resolver.1.push_back(object);
        }
        resolver.0.insert(object, location);
    }

    pub(crate) fn resolve(&self, object: ObjectId) -> Option<ObjectLocation> {
        self.resolver
            .lock()
            .unwrap()
            .0
            .get(&object)
            .copied()
            .or_else(|| {
                self.entries
                    .lock()
                    .unwrap()
                    .get(&object)
                    .map(|entry| entry.location())
            })
    }

    pub(crate) fn clear_all(&self) {
        self.entries.lock().unwrap().clear();
        self.root_count.store(0, Ordering::Release);
        self.transfer_count.store(0, Ordering::Release);
        self.placement_count.store(0, Ordering::Release);
        self.pending_transfers.lock().unwrap().clear();
        self.active_migrations.lock().unwrap().clear();
        self.prepared_count.store(0, Ordering::Release);
        self.retirements.lock().unwrap().clear();
        self.clear_resolver();
    }

    pub(crate) fn clear_resolver(&self) {
        let mut resolver = self.resolver.lock().unwrap();
        resolver.0.clear();
        resolver.1.clear();
    }

    // `fetch_update` is deprecated on newer toolchains in favor of `try_update`,
    // which does not exist at the crate's MSRV (1.85).
    #[allow(deprecated)]
    pub(crate) fn begin_transfer(
        &self,
        object: ObjectId,
        locality: LocalityId,
        now: Instant,
    ) -> bool {
        let key = (object, locality);
        if self
            .entries
            .lock()
            .unwrap()
            .get(&object)
            .is_none_or(|entry| entry.leases.lock().unwrap().contains_key(&locality))
        {
            return false;
        }
        {
            let mut transfers = self.pending_transfers.lock().unwrap();
            if transfers.contains(&key) {
                return false;
            }
            if self
                .transfer_count
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                    (count < self.limits.max_transfers as u64).then_some(count + 1)
                })
                .is_err()
            {
                return false;
            }
            transfers.insert(key);
        }
        if self.acquire(object, locality, now) {
            true
        } else {
            self.pending_transfers.lock().unwrap().remove(&key);
            self.transfer_count.fetch_sub(1, Ordering::AcqRel);
            false
        }
    }

    pub(crate) fn ack_transfer(&self, object: ObjectId, locality: LocalityId) -> bool {
        if self
            .pending_transfers
            .lock()
            .unwrap()
            .remove(&(object, locality))
        {
            self.transfer_count.fetch_sub(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    pub(crate) fn acquire(&self, object: ObjectId, locality: LocalityId, now: Instant) -> bool {
        let Some(entry) = self.entries.lock().unwrap().get(&object).cloned() else {
            return false;
        };
        let mut leases = entry.leases.lock().unwrap();
        if leases.len() >= self.limits.max_local_leases && !leases.contains_key(&locality) {
            return false;
        }
        leases.insert(
            locality,
            LeaseRecord {
                expires_at: now + self.limits.lease_ttl,
                suspect_since: None,
            },
        );
        true
    }

    pub(crate) fn release_root(&self, object: ObjectId) -> bool {
        let Some(entry) = self.entries.lock().unwrap().get(&object).cloned() else {
            return false;
        };
        let mut roots = entry.roots.lock().unwrap();
        if *roots == 0 {
            return false;
        }
        *roots -= 1;
        self.root_count.fetch_sub(1, Ordering::AcqRel);
        drop(roots);
        self.collect(object, Instant::now());
        true
    }

    pub(crate) fn renew(&self, object: ObjectId, locality: LocalityId, now: Instant) -> bool {
        let Some(entry) = self.entries.lock().unwrap().get(&object).cloned() else {
            return false;
        };
        let mut leases = entry.leases.lock().unwrap();
        let Some(lease) = leases.get_mut(&locality) else {
            return false;
        };
        lease.expires_at = now + self.limits.lease_ttl;
        lease.suspect_since = None;
        true
    }

    pub(crate) fn release(&self, object: ObjectId, locality: LocalityId) -> bool {
        let Some(entry) = self.entries.lock().unwrap().get(&object).cloned() else {
            return false;
        };
        let removed = entry.leases.lock().unwrap().remove(&locality).is_some();
        self.ack_transfer(object, locality);
        self.collect(object, Instant::now());
        removed
    }

    pub(crate) fn expire(&self, now: Instant) {
        let ids: Vec<_> = self.entries.lock().unwrap().keys().copied().collect();
        for id in ids {
            let Some(entry) = self.entries.lock().unwrap().get(&id).cloned() else {
                continue;
            };
            {
                let mut lifecycle = entry.lifecycle.lock().unwrap();
                match *lifecycle {
                    ObjectLifecycle::Freezing { expires_at, .. } if expires_at <= now => {
                        *lifecycle = ObjectLifecycle::Resident;
                        *entry.snapshot_bytes.lock().unwrap() = 0;
                    }
                    ObjectLifecycle::Prepared { expires_at, .. } if expires_at <= now => {
                        if entry.prepared.lock().unwrap().take().is_some() {
                            self.prepared_count.fetch_sub(1, Ordering::AcqRel);
                        }
                        *lifecycle = ObjectLifecycle::Forwarding {
                            location: entry.location(),
                            expires_at: now + self.limits.forwarding_ttl,
                        };
                    }
                    ObjectLifecycle::Forwarding { expires_at, .. }
                        if expires_at <= now && entry.authority != self.local_id =>
                    {
                        *entry.state.lock().unwrap() = None;
                    }
                    _ => {}
                }
                let expired_prepared = entry
                    .prepared
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|prepared| prepared.expires_at <= now);
                if expired_prepared && entry.prepared.lock().unwrap().take().is_some() {
                    self.prepared_count.fetch_sub(1, Ordering::AcqRel);
                }
            }
            let mut leases = entry.leases.lock().unwrap();
            for lease in leases.values_mut() {
                if lease.expires_at <= now && lease.suspect_since.is_none() {
                    lease.suspect_since = Some(now);
                }
            }
            leases.retain(|_, lease| {
                lease.suspect_since.is_none_or(|since| {
                    now.saturating_duration_since(since) < self.limits.lease_grace
                })
            });
            let active: HashSet<_> = leases.keys().copied().collect();
            drop(leases);
            let mut transfers = self.pending_transfers.lock().unwrap();
            let before = transfers.len();
            transfers.retain(|(object, locality)| *object != id || active.contains(locality));
            let removed = before - transfers.len();
            if removed != 0 {
                self.transfer_count
                    .fetch_sub(removed as u64, Ordering::AcqRel);
            }
            drop(transfers);
            self.collect(id, now);
        }
    }

    fn collect(&self, object: ObjectId, _now: Instant) {
        let Some(entry) = self.entries.lock().unwrap().get(&object).cloned() else {
            return;
        };
        let idle = *entry.in_flight.lock().unwrap() == 0
            && *entry.placement_tickets.lock().unwrap() == 0
            && entry.admission.lock().unwrap().queue.is_empty()
            && entry.prepared.lock().unwrap().is_none();
        if !idle {
            return;
        }
        if entry.authority != self.local_id {
            let lifecycle = entry.lifecycle.lock().unwrap();
            if matches!(
                *lifecycle,
                ObjectLifecycle::Forwarding { .. } | ObjectLifecycle::Retiring { .. }
            ) && entry.state.lock().unwrap().is_none()
            {
                drop(lifecycle);
                self.entries.lock().unwrap().remove(&object);
                self.resolver.lock().unwrap().0.remove(&object);
            }
            return;
        }
        if !entry.leases.lock().unwrap().is_empty() || *entry.roots.lock().unwrap() != 0 {
            return;
        }
        let lifecycle = *entry.lifecycle.lock().unwrap();
        match lifecycle {
            ObjectLifecycle::Resident if entry.state.lock().unwrap().is_some() => {
                self.entries.lock().unwrap().remove(&object);
                self.resolver.lock().unwrap().0.remove(&object);
            }
            ObjectLifecycle::Forwarding { location, .. }
                if entry.state.lock().unwrap().is_none() =>
            {
                if location.locality() == self.local_id {
                    self.entries.lock().unwrap().remove(&object);
                    self.resolver.lock().unwrap().0.remove(&object);
                    return;
                }
                let registration = self.registry.types.get(&entry.object_type).unwrap();
                let mut retirements = self.retirements.lock().unwrap();
                if retirements.len() >= self.limits.max_migrations {
                    return;
                }
                *entry.lifecycle.lock().unwrap() = ObjectLifecycle::Retiring { location };
                retirements.push_back(Retirement {
                    object,
                    object_type: entry.object_type,
                    authority: entry.authority,
                    location,
                    snapshot_version: registration.snapshot_version,
                    snapshot_schema: registration.snapshot_schema,
                });
            }
            ObjectLifecycle::Retiring { .. } => {}
            _ => {}
        }
    }

    pub(crate) fn take_retirements(&self, max: usize) -> Vec<Retirement> {
        let mut queue = self.retirements.lock().unwrap();
        (0..max).filter_map(|_| queue.pop_front()).collect()
    }

    pub(crate) fn requeue_retirement(&self, retirement: Retirement) {
        self.retirements.lock().unwrap().push_front(retirement);
    }

    pub(crate) fn finish_retirement(&self, object: ObjectId) {
        self.entries.lock().unwrap().remove(&object);
        self.resolver.lock().unwrap().0.remove(&object);
    }

    pub(crate) fn stats(&self, now: Instant) -> ObjectStats {
        let entries = self.entries.lock().unwrap();
        let resolver_entries = self.resolver.lock().unwrap().0.len();
        let mut stats = ObjectStats {
            live_objects: entries.len(),
            resolver_entries,
            resolver_bytes: resolver_entries
                .saturating_mul(std::mem::size_of::<(ObjectId, ObjectLocation)>()),
            local_calls: self.local_calls.load(Ordering::Relaxed),
            remote_calls: self.remote_calls.load(Ordering::Relaxed),
            transfers: self.transfer_count.load(Ordering::Relaxed) as usize,
            placement_tickets: self.placement_count.load(Ordering::Relaxed) as usize,
            active_migrations: self.active_migrations.lock().unwrap().len(),
            prepared_migrations: self.prepared_count.load(Ordering::Relaxed) as usize,
            completed_migrations: self.completed_migrations.load(Ordering::Relaxed),
            rolled_back_migrations: self.rolled_back_migrations.load(Ordering::Relaxed),
            failed_migrations: self.failed_migrations.load(Ordering::Relaxed),
            transferred_snapshot_bytes: self.migrated_bytes.load(Ordering::Relaxed),
            ..ObjectStats::default()
        };
        for entry in entries.values() {
            let leases = entry.leases.lock().unwrap();
            stats.leases += leases.len();
            stats.suspect_leases += leases
                .values()
                .filter(|lease| lease.expires_at <= now || lease.suspect_since.is_some())
                .count();
            stats.roots += *entry.roots.lock().unwrap();
            stats.in_flight_calls += *entry.in_flight.lock().unwrap();
            stats.queued_calls += entry.admission.lock().unwrap().queue.len();
            stats.migration_snapshot_bytes = stats
                .migration_snapshot_bytes
                .saturating_add(*entry.snapshot_bytes.lock().unwrap());
            if matches!(
                *entry.lifecycle.lock().unwrap(),
                ObjectLifecycle::Forwarding { .. } | ObjectLifecycle::Retiring { .. }
            ) {
                stats.forwarding_entries += 1;
            }
        }
        stats
    }
}

pub(crate) fn decode_location(bytes: Vec<u8>) -> Result<ObjectLocation, RuntimeError> {
    if bytes.len() != 32 {
        return Err(RuntimeError::Protocol(
            "invalid object location payload".into(),
        ));
    }
    ObjectLocation::new(
        LocalityId::new(u64::from_le_bytes(bytes[0..8].try_into().unwrap())),
        DomainId::new(u32::from_le_bytes(bytes[8..12].try_into().unwrap())),
        u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
        u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    )
    .map_err(|error| RuntimeError::Protocol(format!("invalid object location: {error}")))
}

pub(crate) struct LocalLease {
    object: ObjectId,
    object_type: ObjectTypeId,
    pub(crate) location: ObjectLocation,
    client: RuntimeClient,
    active: AtomicBool,
    next_renewal: Mutex<Instant>,
}

impl LocalLease {
    fn new(
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
        client: RuntimeClient,
        renew_interval: Duration,
    ) -> Self {
        Self {
            object,
            object_type,
            location,
            client,
            active: AtomicBool::new(true),
            next_renewal: Mutex::new(Instant::now() + renew_interval),
        }
    }

    pub(crate) fn renew(&self) {
        self.client
            .renew_object(self.object, self.object_type, self.location);
    }

    pub(crate) fn deactivate(&self) {
        if self.active.swap(false, Ordering::AcqRel) {
            self.client
                .release_object(self.object, self.object_type, self.location);
        }
    }
}

impl Drop for LocalLease {
    fn drop(&mut self) {
        if self.active.swap(false, Ordering::AcqRel) {
            self.client
                .release_object(self.object, self.object_type, self.location);
        }
    }
}

pub(crate) struct LocalLeaseManager {
    limit: usize,
    renew_interval: Duration,
    entries: Mutex<HashMap<ObjectId, Weak<LocalLease>>>,
}

impl LocalLeaseManager {
    pub(crate) fn new(limit: usize, renew_interval: Duration) -> Self {
        Self {
            limit,
            renew_interval,
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn attach<T: DistributedObject>(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
        client: RuntimeClient,
    ) -> Result<Remote<T>, RuntimeError> {
        let mut entries = self.entries.lock().unwrap();
        if let Some(existing) = entries.get(&object).and_then(Weak::upgrade) {
            return Ok(Remote {
                object,
                object_type,
                location,
                client,
                lease: existing,
                marker: PhantomData,
            });
        }
        entries.retain(|_, lease| lease.strong_count() != 0);
        if entries.len() >= self.limit {
            return Err(RuntimeError::ResourceExhausted {
                resource: crate::ResourceKind::LocalityLeases,
                limit: self.limit,
            });
        }
        let lease = Arc::new(LocalLease::new(
            object,
            object_type,
            location,
            client.clone(),
            self.renew_interval,
        ));
        entries.insert(object, Arc::downgrade(&lease));
        Ok(Remote {
            object,
            object_type,
            location,
            client,
            lease,
            marker: PhantomData,
        })
    }

    pub(crate) fn due(&self, now: Instant) -> Vec<Arc<LocalLease>> {
        let mut entries = self.entries.lock().unwrap();
        let mut due = Vec::new();
        entries.retain(|_, weak| {
            let Some(lease) = weak.upgrade() else {
                return false;
            };
            let mut next = lease.next_renewal.lock().unwrap();
            if lease.active.load(Ordering::Acquire) && *next <= now {
                *next = now + self.renew_interval;
                due.push(Arc::clone(&lease));
            }
            true
        });
        due
    }

    pub(crate) fn clear(&self) {
        let leases: Vec<_> = self
            .entries
            .lock()
            .unwrap()
            .drain()
            .filter_map(|(_, lease)| lease.upgrade())
            .collect();
        for lease in leases {
            lease.deactivate();
        }
    }
}

#[must_use = "a colocated future must be driven or dropped"]
pub struct ColocatedFuture<T: WireValue> {
    inner: RemoteObjectCall<T>,
    _lease: Arc<LocalLease>,
}

impl<T: WireValue> ColocatedFuture<T> {
    pub(crate) fn new(inner: RemoteObjectCall<T>, lease: Arc<LocalLease>) -> Self {
        Self {
            inner,
            _lease: lease,
        }
    }
}

impl<T: WireValue> Future for ColocatedFuture<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.inner).poll(context)
    }
}

impl<T: WireValue> Unpin for ColocatedFuture<T> {}

enum PlacementState<T: WireValue> {
    Preferred(ColocatedFuture<T>),
    Fallback(RemoteFuture<T>),
    Done,
}

#[must_use = "a placement future must be driven or dropped"]
pub struct PlacementFuture<T: WireValue> {
    state: PlacementState<T>,
    fallback: PlacementFallback,
    client: RuntimeClient,
    action: ActionId,
    payload: Option<Segments>,
    options: SpawnOptions,
}

impl<T: WireValue> PlacementFuture<T> {
    pub(crate) fn new(
        preferred: ColocatedFuture<T>,
        fallback: PlacementFallback,
        client: RuntimeClient,
        action: ActionId,
        payload: Segments,
        options: SpawnOptions,
    ) -> Self {
        Self {
            state: PlacementState::Preferred(preferred),
            fallback,
            client,
            action,
            payload: Some(payload),
            options,
        }
    }
}

impl<T: WireValue> Future for PlacementFuture<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match &mut self.state {
                PlacementState::Preferred(future) => match Pin::new(future).poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(value)) => {
                        self.state = PlacementState::Done;
                        return Poll::Ready(Ok(value));
                    }
                    Poll::Ready(Err(error)) if placement_saturated(&error) => {
                        let place = match self.fallback {
                            PlacementFallback::Any => {
                                Place::new(self.client.local_id(), DomainId::DEFAULT)
                            }
                            PlacementFallback::Place(place) => place,
                            PlacementFallback::Reject => {
                                self.state = PlacementState::Done;
                                return Poll::Ready(Err(error));
                            }
                        };
                        let payload = self.payload.take().expect("placement fallback used once");
                        match self
                            .client
                            .spawn_encoded(place, self.action, payload, self.options)
                        {
                            Ok(future) => self.state = PlacementState::Fallback(future),
                            Err(error) => {
                                self.state = PlacementState::Done;
                                return Poll::Ready(Err(error));
                            }
                        }
                    }
                    Poll::Ready(Err(error)) => {
                        self.state = PlacementState::Done;
                        return Poll::Ready(Err(error));
                    }
                },
                PlacementState::Fallback(future) => {
                    let result = Pin::new(future).poll(context);
                    if result.is_ready() {
                        self.state = PlacementState::Done;
                    }
                    return result;
                }
                PlacementState::Done => {
                    return Poll::Ready(Err(RuntimeError::Protocol(
                        "placement future polled after completion".into(),
                    )));
                }
            }
        }
    }
}

fn placement_saturated(error: &RuntimeError) -> bool {
    matches!(
        error,
        RuntimeError::ResourceExhausted { .. } | RuntimeError::Placement { .. }
    ) || matches!(error, RuntimeError::RemoteAction { message, .. }
            if message.contains("ResourceExhausted")
                || message.contains("saturated")
                || message.contains("queue")
                || message.contains("MigrationInProgress")
                || message.contains("migration"))
}

impl<T: WireValue> Unpin for PlacementFuture<T> {}

pub(crate) struct RemoteCallSpec {
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) action: ActionId,
    pub(crate) payload: Segments,
    pub(crate) last_epoch: u64,
    pub(crate) max_redirects: usize,
    pub(crate) deadline: Duration,
}

pub(crate) struct RemoteObjectCall<T: WireValue> {
    future: RemoteFuture<T>,
    client: RuntimeClient,
    object: ObjectId,
    object_type: ObjectTypeId,
    action: ActionId,
    payload: Segments,
    request: RequestId,
    last_epoch: u64,
    redirects: usize,
    max_redirects: usize,
    deadline: Duration,
}

impl<T: WireValue> RemoteObjectCall<T> {
    pub(crate) fn new(
        future: RemoteFuture<T>,
        client: RuntimeClient,
        spec: RemoteCallSpec,
    ) -> Self {
        let request = future.request_id();
        Self {
            future,
            client,
            object: spec.object,
            object_type: spec.object_type,
            action: spec.action,
            payload: spec.payload,
            request,
            last_epoch: spec.last_epoch,
            redirects: 0,
            max_redirects: spec.max_redirects,
            deadline: spec.deadline,
        }
    }
}

impl<T: WireValue> Future for RemoteObjectCall<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.future).poll(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(RuntimeError::Moved { object, location })) => {
                    if object != self.object || location.epoch() <= self.last_epoch {
                        return Poll::Ready(Err(RuntimeError::Protocol(format!(
                            "invalid object redirect: expected {:?} after epoch {}, got {:?} at epoch {}",
                            self.object,
                            self.last_epoch,
                            object,
                            location.epoch()
                        ))));
                    }
                    if self.redirects >= self.max_redirects {
                        return Poll::Ready(Err(RuntimeError::RedirectLimit(self.object)));
                    }
                    self.redirects += 1;
                    self.last_epoch = location.epoch();
                    self.client.cache_object_location(self.object, location);
                    let payload = ObjectEnvelope {
                        object: self.object,
                        object_type: self.object_type,
                        epoch: location.epoch(),
                        lease_locality: self.client.local_id(),
                        domain: location.domain(),
                        rooted: false,
                    }
                    .encode(self.payload.clone());
                    match self.client.spawn_encoded_retry(
                        Place::new(location.locality(), location.domain()),
                        self.action,
                        payload,
                        SpawnOptions {
                            deadline: self.deadline,
                            trace_id: None,
                        },
                        self.request,
                    ) {
                        Ok(future) => self.future = future,
                        Err(error) => return Poll::Ready(Err(error)),
                    }
                }
                Poll::Ready(result) => return Poll::Ready(result),
            }
        }
    }
}

impl<T: WireValue> Unpin for RemoteObjectCall<T> {}

enum ObjectCallState<T: WireValue> {
    Local(Option<Result<T, RuntimeError>>),
    Remote(RemoteObjectCall<T>),
}

#[must_use = "an object call future must be driven or dropped"]
pub struct ObjectCallFuture<T: WireValue> {
    state: ObjectCallState<T>,
}

impl<T: WireValue> ObjectCallFuture<T> {
    pub(crate) fn local(result: Result<T, RuntimeError>) -> Self {
        Self {
            state: ObjectCallState::Local(Some(result)),
        }
    }

    pub(crate) fn remote(future: RemoteObjectCall<T>) -> Self {
        Self {
            state: ObjectCallState::Remote(future),
        }
    }
}

impl<T: WireValue> std::fmt::Debug for ObjectCallFuture<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectCallFuture").finish_non_exhaustive()
    }
}

impl<T: WireValue> Future for ObjectCallFuture<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.state {
            ObjectCallState::Local(result) => Poll::Ready(result.take().unwrap_or_else(|| {
                Err(RuntimeError::Protocol(
                    "object future polled after completion".into(),
                ))
            })),
            ObjectCallState::Remote(future) => Pin::new(future).poll(context),
        }
    }
}

impl<T: WireValue> Unpin for ObjectCallFuture<T> {}

#[must_use = "an object creation future must be driven or dropped"]
pub struct CreateFuture<T: DistributedObject> {
    inner: Option<RemoteFuture<Vec<u8>>>,
    ready: Option<Result<Remote<T>, RuntimeError>>,
    object: ObjectId,
    object_type: ObjectTypeId,
    client: RuntimeClient,
}

impl<T: DistributedObject> std::fmt::Debug for CreateFuture<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateFuture")
            .field("object", &self.object)
            .finish_non_exhaustive()
    }
}

impl<T: DistributedObject> CreateFuture<T> {
    pub(crate) fn new(
        inner: RemoteFuture<Vec<u8>>,
        object: ObjectId,
        object_type: ObjectTypeId,
        client: RuntimeClient,
    ) -> Self {
        Self {
            inner: Some(inner),
            ready: None,
            object,
            object_type,
            client,
        }
    }

    pub(crate) fn ready(
        result: Result<Remote<T>, RuntimeError>,
        object: ObjectId,
        object_type: ObjectTypeId,
        client: RuntimeClient,
    ) -> Self {
        Self {
            inner: None,
            ready: Some(result),
            object,
            object_type,
            client,
        }
    }
}

impl<T: DistributedObject> Future for CreateFuture<T> {
    type Output = Result<Remote<T>, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(result) = self.ready.take() {
            return Poll::Ready(result);
        }
        let poll = Pin::new(
            self.inner
                .as_mut()
                .expect("create future polled after completion"),
        )
        .poll(context);
        match poll {
            Poll::Ready(Ok(bytes)) => {
                self.inner = None;
                Poll::Ready(decode_location(bytes).and_then(|location| {
                    self.client
                        .attach_object(self.object, self.object_type, location)
                }))
            }
            Poll::Ready(Err(error)) => {
                self.inner = None;
                Poll::Ready(Err(error))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: DistributedObject> Unpin for CreateFuture<T> {}

#[must_use = "a weak upgrade future must be driven or dropped"]
pub struct UpgradeFuture<T: DistributedObject> {
    inner: RemoteFuture<()>,
    object: ObjectId,
    object_type: ObjectTypeId,
    location: ObjectLocation,
    client: RuntimeClient,
    marker: PhantomData<T>,
}

impl<T: DistributedObject> UpgradeFuture<T> {
    pub(crate) fn new(
        inner: RemoteFuture<()>,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
        client: RuntimeClient,
    ) -> Self {
        Self {
            inner,
            object,
            object_type,
            location,
            client,
            marker: PhantomData,
        }
    }
}

impl<T: DistributedObject> Future for UpgradeFuture<T> {
    type Output = Result<Remote<T>, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.inner).poll(context) {
            Poll::Ready(Ok(())) => Poll::Ready(self.client.attach_object(
                self.object,
                self.object_type,
                self.location,
            )),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: DistributedObject> Unpin for UpgradeFuture<T> {}

pub struct LeaseTransfer<T: DistributedObject> {
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) location: ObjectLocation,
    pub(crate) destination: LocalityId,
    pub(crate) expires_at: Instant,
    source_client: RuntimeClient,
    pub(crate) active: bool,
    marker: PhantomData<T>,
}

impl<T: DistributedObject> std::fmt::Debug for LeaseTransfer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeaseTransfer")
            .field("object", &self.object)
            .field("destination", &self.destination)
            .finish_non_exhaustive()
    }
}

impl<T: DistributedObject> Drop for LeaseTransfer<T> {
    fn drop(&mut self) {
        if self.active {
            self.source_client.release_for_locality(
                self.object,
                self.object_type,
                self.location,
                self.destination,
            );
        }
    }
}

#[must_use = "a lease transfer future must be driven or dropped"]
pub struct TransferFuture<T: DistributedObject> {
    inner: RemoteFuture<()>,
    object: ObjectId,
    object_type: ObjectTypeId,
    location: ObjectLocation,
    destination: LocalityId,
    expires_at: Instant,
    source_client: RuntimeClient,
    marker: PhantomData<T>,
}

impl<T: DistributedObject> TransferFuture<T> {
    pub(crate) fn new(
        inner: RemoteFuture<()>,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: ObjectLocation,
        destination: LocalityId,
        expires_at: Instant,
        source_client: RuntimeClient,
    ) -> Self {
        Self {
            inner,
            object,
            object_type,
            location,
            destination,
            expires_at,
            source_client,
            marker: PhantomData,
        }
    }
}

impl<T: DistributedObject> Future for TransferFuture<T> {
    type Output = Result<LeaseTransfer<T>, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.inner).poll(context) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(LeaseTransfer {
                object: self.object,
                object_type: self.object_type,
                location: self.location,
                destination: self.destination,
                expires_at: self.expires_at,
                source_client: self.source_client.clone(),
                active: true,
                marker: PhantomData,
            })),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: DistributedObject> Unpin for TransferFuture<T> {}

pub(crate) fn encode_location(location: ObjectLocation) -> Result<Segments, ActionError> {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(&location.locality().get().to_le_bytes());
    bytes.extend_from_slice(&location.domain().get().to_le_bytes());
    bytes.extend_from_slice(&location.slot().to_le_bytes());
    bytes.extend_from_slice(&location.generation().to_le_bytes());
    bytes.extend_from_slice(&location.epoch().to_le_bytes());
    Ok(vec![bytes])
}

pub struct Remote<T: DistributedObject> {
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) location: ObjectLocation,
    pub(crate) client: RuntimeClient,
    pub(crate) lease: Arc<LocalLease>,
    marker: PhantomData<T>,
}

impl<T: DistributedObject> Clone for Remote<T> {
    fn clone(&self) -> Self {
        Self {
            object: self.object,
            object_type: self.object_type,
            location: self.location,
            client: self.client.clone(),
            lease: Arc::clone(&self.lease),
            marker: PhantomData,
        }
    }
}

impl<T: DistributedObject> std::fmt::Debug for Remote<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Remote")
            .field("object", &self.object)
            .field("object_type", &self.object_type)
            .finish_non_exhaustive()
    }
}

impl<T: DistributedObject> Remote<T> {
    pub const fn object_id(&self) -> ObjectId {
        self.object
    }

    pub fn observed_location(&self) -> ObjectLocation {
        self.client
            .resolve_object(self.object)
            .unwrap_or(self.location)
    }

    /// Releases this strong handle. The locality lease is released when this
    /// is the last local clone.
    pub fn release(self) {
        drop(self);
    }

    pub fn call_read<A: ObjectReadAction<T>>(
        &self,
        action: A,
    ) -> Result<ObjectCallFuture<A::Output>, RuntimeError>
    where
        T: Sync,
    {
        self.client.call_object_read(self, action)
    }

    pub fn call_write<A: ObjectWriteAction<T>>(
        &self,
        action: A,
    ) -> Result<ObjectCallFuture<A::Output>, RuntimeError> {
        self.client.call_object_write(self, action)
    }

    pub fn transfer_to(&self, destination: LocalityId) -> Result<TransferFuture<T>, RuntimeError> {
        self.client.transfer_object(self, destination)
    }

    /// Starts explicit migration to an exact logical place.
    ///
    /// # Errors
    /// Returns typed placement, mobility, snapshot, resource, transport,
    /// rollback, post-commit, or shutdown errors.
    pub fn migrate_to(&self, destination: Place) -> Result<MigrationFuture<T>, RuntimeError>
    where
        T: MobileObject,
    {
        self.client.migrate(self, destination)
    }

    pub fn downgrade(&self) -> WeakRemote<T> {
        WeakRemote {
            object: self.object,
            object_type: self.object_type,
            location: self.lease.location,
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct WeakRemote<T: DistributedObject> {
    pub(crate) object: ObjectId,
    pub(crate) object_type: ObjectTypeId,
    pub(crate) location: ObjectLocation,
    marker: PhantomData<T>,
}

impl<T: DistributedObject> Copy for WeakRemote<T> {}

impl<T: DistributedObject> Clone for WeakRemote<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: DistributedObject> WireValue for WeakRemote<T> {
    const SCHEMA_ID: u64 = T::SCHEMA_ID ^ 0x5745_414b_4f42_4a00;

    fn encode(self) -> Result<Segments, ActionError> {
        let mut bytes = Vec::with_capacity(80);
        bytes.extend_from_slice(&self.object.run().get().to_le_bytes());
        bytes.extend_from_slice(&self.object.unique().to_le_bytes());
        bytes.extend_from_slice(&self.object_type.get().to_le_bytes());
        bytes.extend_from_slice(&self.location.locality().get().to_le_bytes());
        bytes.extend_from_slice(&self.location.domain().get().to_le_bytes());
        bytes.extend_from_slice(&self.location.slot().to_le_bytes());
        bytes.extend_from_slice(&self.location.generation().to_le_bytes());
        bytes.extend_from_slice(&self.location.epoch().to_le_bytes());
        Ok(vec![bytes])
    }

    fn decode(mut segments: Segments) -> Result<Self, ActionError> {
        if segments.len() != 1 || segments[0].len() != 80 {
            return Err(ActionError::codec("invalid weak remote payload"));
        }
        let bytes = segments.pop().unwrap();
        let run = RunId::new(u128::from_le_bytes(bytes[0..16].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid weak remote run"))?;
        let object = ObjectId::new(run, u128::from_le_bytes(bytes[16..32].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid weak remote object"))?;
        let object_type = ObjectTypeId::new(u128::from_le_bytes(bytes[32..48].try_into().unwrap()))
            .map_err(|_| ActionError::codec("invalid weak remote type"))?;
        if object_type.get() != T::TYPE_ID {
            return Err(ActionError::codec("weak remote type mismatch"));
        }
        let location = ObjectLocation::new(
            LocalityId::new(u64::from_le_bytes(bytes[48..56].try_into().unwrap())),
            DomainId::new(u32::from_le_bytes(bytes[56..60].try_into().unwrap())),
            u64::from_le_bytes(bytes[60..68].try_into().unwrap()),
            u32::from_le_bytes(bytes[68..72].try_into().unwrap()),
            u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
        )
        .map_err(|_| ActionError::codec("invalid weak remote location"))?;
        Ok(Self {
            object,
            object_type,
            location,
            marker: PhantomData,
        })
    }
}

impl<T: DistributedObject> WeakRemote<T> {
    pub const fn object_id(self) -> ObjectId {
        self.object
    }

    pub fn upgrade(&self, client: &RuntimeClient) -> Result<UpgradeFuture<T>, RuntimeError> {
        client.upgrade_object(self)
    }
}

pub struct ObjectRoot<T: DistributedObject> {
    object: ObjectId,
    object_type: ObjectTypeId,
    location: ObjectLocation,
    client: RuntimeClient,
    active: bool,
    marker: PhantomData<T>,
}

impl<T: DistributedObject> std::fmt::Debug for ObjectRoot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectRoot")
            .field("object", &self.object)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

impl<T: DistributedObject> ObjectRoot<T> {
    pub(crate) fn new(remote: &Remote<T>) -> Self {
        Self {
            object: remote.object,
            object_type: remote.object_type,
            location: remote.location,
            client: remote.client.clone(),
            active: true,
            marker: PhantomData,
        }
    }

    pub const fn object_id(&self) -> ObjectId {
        self.object
    }

    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        if self.active {
            self.active = false;
            self.client
                .release_root(self.object, self.object_type, self.location);
        }
    }
}

impl<T: DistributedObject> Drop for ObjectRoot<T> {
    fn drop(&mut self) {
        self.release_inner();
    }
}

#[must_use = "a rooted object creation future must be driven or dropped"]
pub struct RootedCreateFuture<T: DistributedObject> {
    inner: CreateFuture<T>,
}

impl<T: DistributedObject> RootedCreateFuture<T> {
    pub(crate) fn new(inner: CreateFuture<T>) -> Self {
        Self { inner }
    }
}

impl<T: DistributedObject> Future for RootedCreateFuture<T> {
    type Output = Result<(Remote<T>, ObjectRoot<T>), RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.inner).poll(context) {
            Poll::Ready(Ok(remote)) => {
                let root = ObjectRoot::new(&remote);
                Poll::Ready(Ok((remote, root)))
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: DistributedObject> Unpin for RootedCreateFuture<T> {}

#[cfg(test)]
mod tests;
